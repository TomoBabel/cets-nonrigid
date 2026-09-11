"""Immutable local CETS JSON + Zarr publication, with document-last commit."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import urlsplit, unquote
import uuid

import numpy as np
import torch
import zarr
from cets_data_model.models import models as m
from cets_data_model.utils.references import document_to_dict

from cets_nonrigid.context import PROFILE_VERSION, alignment_contexts
from cets_nonrigid.samples import AlignmentBundle, DeformationSamples, SampleBlock

_ARRAYS = (
    "points",
    "projected_residual",
    "sample_valid",
    "observation_valid",
    "projection_valid",
    "weights",
    "displacement_3d",
    "displacement_valid",
    "ctf_depth",
    "ctf_depth_valid",
)


def resolve_payload(document_path, uri):
    parsed = urlsplit(uri)
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("profile 0.1 supports local filesystem payloads only")
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri)
    return path if path.is_absolute() else Path(document_path).parent / path


def _group_key(key):
    path = PurePosixPath(key)
    if not key or path.is_absolute() or any(p in {".", ".."} for p in key.split("/")) or "\\" in key:
        raise ValueError("payload_group must be a safe relative Zarr group key")
    return path.as_posix()


def _write_block(group, block, row_name):
    for name in _ARRAYS:
        value = getattr(block, name)
        if value is None:
            continue
        array = value.detach().cpu().numpy()
        if name == "points":
            dims = ("sample", "coordinate")
            chunks = (max(1, min(len(array), 4096)), array.shape[1])
        elif name == "sample_valid":
            dims, chunks = ("sample",), (max(1, min(len(array), 4096)),)
        else:
            dims = (row_name, "sample") + (("coordinate",) if array.ndim == 3 else ())
            chunks = (1, max(1, min(array.shape[1], 4096))) + ((array.shape[2],) if array.ndim == 3 else ())
        group.create_array(name, data=array, chunks=chunks, dimension_names=dims)
    if block.point_ids is not None:
        ids = group.create_array(
            "point_ids",
            shape=(block.count,),
            dtype="str",
            chunks=(max(1, min(block.count, 4096)),),
            dimension_names=("sample",),
        )
        if block.count:
            ids[:] = block.point_ids


def _read_block(group, row_name):
    values = {}
    for name in _ARRAYS:
        if name in group:
            array = group[name]
            expected = (
                ("sample", "coordinate")
                if name == "points"
                else ("sample",)
                if name == "sample_valid"
                else (row_name, "sample") + (("coordinate",) if array.ndim == 3 else ())
            )
            if array.metadata.dimension_names != expected:
                raise ValueError(f"payload {name} has incorrect dimension names")
            values[name] = torch.from_numpy(np.asarray(array[:]).copy())
        else:
            values[name] = None
    if "point_ids" in group and group["point_ids"].metadata.dimension_names != ("sample",):
        raise ValueError("payload point_ids has incorrect dimension names")
    point_ids = list(group["point_ids"][:].tolist()) if "point_ids" in group else None
    return SampleBlock(**values, point_ids=point_ids)


def _read_samples(group, descriptor):
    if "heldout" not in group:
        raise ValueError("payload requires an explicit held-out block, including for zero samples")
    row_name = "tilt_image" if descriptor.kind == "tilt-series-projection-residual" else "frame"
    return DeformationSamples(
        _read_block(group, row_name),
        _read_block(group["heldout"], row_name),
        descriptor.sampling,
        descriptor.heldout,
        descriptor.channels,
    )


def read_bundle(path) -> AlignmentBundle:
    path = Path(path)
    document = m.Dataset.model_validate_json(path.read_text())
    result = AlignmentBundle(document)
    stores = set()
    for context in alignment_contexts(document):
        descriptor = context.owner.non_rigid_alignment
        if descriptor is None:
            continue
        store_path = resolve_payload(path, descriptor.payload_uri)
        stores.add(store_path.resolve())
        if len(stores) > 1:
            raise ValueError("profile 0.1 uses one payload store per document")
        root = zarr.open_group(str(store_path), mode="r", use_consolidated=False)
        if root.attrs.get("profile_version") != PROFILE_VERSION or root.attrs.get("complete") is not True:
            raise ValueError("payload store is incomplete or uses an unsupported profile")
        group = root[_group_key(descriptor.payload_group)]
        expected = {
            "profile_version": PROFILE_VERSION,
            "context_digest": descriptor.context_digest,
            "alignment_id": context.alignment_id,
            "parent_id": context.parent.id,
            "units": "angstrom",
        }
        for key, value in expected.items():
            if group.attrs.get(key) != value:
                raise ValueError(f"payload attribute {key!r} disagrees with its CETS descriptor")
        result.deformations[context.key] = _read_samples(group, descriptor)
        if "reports" in group:
            result.reports["/".join(context.key)] = dict(group["reports"].attrs.get("diagnostics", {}))
        if "snapshots" in group:
            import hashlib

            files = {}
            for key in group["snapshots"].array_keys():
                array = group["snapshots"][key]
                raw = np.asarray(array[:], dtype=np.uint8).tobytes()
                if hashlib.sha256(raw).hexdigest() != array.attrs.get("sha256"):
                    raise ValueError("native snapshot integrity mismatch")
                files[array.attrs["role"]] = raw
            result.snapshots[context.key] = files
    result.validate()
    return result


def write_bundle(bundle: AlignmentBundle, path) -> Path:
    """Publish a new document version. Never mutate the input bundle or replace output.

    A newly reserved store can be visible before publication, but remains incomplete
    and unreferenced until every group is validated and the JSON is linked last.
    """
    bundle.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace CETS document {output}")
    stem = output.stem.removesuffix(".cets")
    store = output.with_name(stem + ".nonrigid.zarr")
    document = m.Dataset.model_validate(document_to_dict(bundle.document))
    staged = AlignmentBundle(document, dict(bundle.deformations), dict(bundle.reports), dict(bundle.snapshots))
    created_store, published = False, False
    temporary_json = None
    try:
        if staged.deformations:
            store.mkdir(exist_ok=False)  # atomic reservation: another writer cannot own this store
            created_store = True
            root = zarr.open_group(str(store), mode="w")
            root.attrs.update({"profile_version": PROFILE_VERSION, "complete": False})
            for context in alignment_contexts(document):
                descriptor = context.owner.non_rigid_alignment
                if descriptor is None:
                    continue
                data = staged.deformations[context.key]
                prefix = "movies" if context.ndim == 2 else "alignments"
                group_id = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(context.key)).hex
                key = prefix + "/" + group_id
                descriptor.payload_uri = store.name
                descriptor.payload_group = key
                root.require_group(prefix)
                temporary_key = prefix + "/.staging-" + uuid.uuid4().hex
                group = root.create_group(temporary_key)
                group.attrs.update(
                    {
                        "profile_version": PROFILE_VERSION,
                        "context_digest": descriptor.context_digest,
                        "alignment_id": context.alignment_id,
                        "parent_id": context.parent.id,
                        "units": "angstrom",
                    }
                )
                row_name = "frame" if context.ndim == 2 else "tilt_image"
                _write_block(group, data.training, row_name)
                _write_block(group.create_group("heldout"), data.heldout, row_name)
                if "/".join(context.key) in staged.reports:
                    group.create_group("reports").attrs["diagnostics"] = staged.reports["/".join(context.key)]
                if context.key in staged.snapshots:
                    import hashlib

                    snapshots = group.create_group("snapshots")
                    for role, raw in staged.snapshots[context.key].items():
                        array = snapshots.create_array(
                            hashlib.sha256(role.encode()).hexdigest(),
                            data=np.frombuffer(raw, dtype=np.uint8),
                            dimension_names=("byte",),
                        )
                        array.attrs.update({"role": role, "sha256": hashlib.sha256(raw).hexdigest()})
                _read_samples(group, descriptor).validate(context)
                os.rename(store / temporary_key, store / key)
            root.attrs["complete"] = True
        staged.validate()
        fd, name = tempfile.mkstemp(prefix="." + output.name + ".", suffix=".tmp", dir=output.parent)
        temporary_json = Path(name)
        with os.fdopen(fd, "w") as handle:
            json.dump(document_to_dict(document), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if the final document already exists.
        os.link(temporary_json, output)
        published = True
        return output
    finally:
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)
        if created_store and not published:
            shutil.rmtree(store)
