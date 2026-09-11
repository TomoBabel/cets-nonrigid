"""CETS-owned geometry and deterministic context binding.

No native file is consulted here. Globals are evaluated from the ordinary core
models, using the shared rigid codec's folding and frame interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from cets_data_model.models import models as m
from cets_data_model.utils.references import document_to_dict, validate_document_references
from cryoet_alignment.io.cets.alignment import fold_projection
from cryoet_alignment.io.cets.frames import image_frame

PROFILE_VERSION = "cets-nonrigid/0.1"
DIGEST_VERSION = 1


def canonical_json(value: Any) -> bytes:
    """Digest v1: finite float64 hexadecimal strings, sorted keys and UTF-8."""

    def normalize(item):
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("context digest does not accept nonfinite values")
            return (0.0 if number == 0 else number).hex()
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            return int(item)
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("canonical context object keys must be strings")
            return {key: normalize(value) for key, value in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if item is None or isinstance(item, (str, bool)):
            return item
        if hasattr(item, "model_dump"):
            return normalize(document_to_dict(item))
        raise TypeError(f"unsupported canonical context value {type(item).__name__}")

    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def movie_stacks(region):
    collection = region.movie_stack_collection
    for series in (collection.movie_stacks or []) if collection is not None else []:
        yield from series.stacks or []


def alignment_contexts(document):
    for region in document.regions or []:
        for alignment in region.alignments or []:
            if alignment.id:
                yield CetsContext(document, alignment.id, region.id)
        for stack in movie_stacks(region):
            for alignment in stack.alignments or []:
                if alignment.id:
                    yield CetsContext(document, alignment.id, region.id)


@dataclass(frozen=True)
class CetsContext:
    document: m.Dataset
    alignment_id: str
    region_id: str | None = None

    def resolve(self):
        matches = []
        for region in self.document.regions or []:
            if self.region_id is not None and region.id != self.region_id:
                continue
            for owner in region.alignments or []:
                if owner.id == self.alignment_id:
                    parents = [s for s in region.tilt_series or [] if s.id == owner.tilt_series_id]
                    if len(parents) != 1:
                        raise ValueError("alignment requires one resolved tilt series")
                    matches.append((region, owner, parents[0]))
            for stack in movie_stacks(region):
                for owner in stack.alignments or []:
                    if owner.id == self.alignment_id:
                        matches.append((region, owner, stack))
        if len(matches) != 1:
            raise ValueError(
                f"alignment {self.alignment_id!r} resolves to {len(matches)} objects; supply unique IDs and region"
            )
        return matches[0]

    @property
    def key(self):
        return self.resolve()[0].id, self.alignment_id

    @property
    def owner(self):
        return self.resolve()[1]

    @property
    def parent(self):
        return self.resolve()[2]

    @property
    def kind(self):
        return "movie-frame-residual" if isinstance(self.owner, m.MovieAlignment) else "tilt-series-projection-residual"

    @property
    def ndim(self):
        return 2 if self.kind == "movie-frame-residual" else 3

    @property
    def row_ids(self):
        owner = self.owner
        if self.ndim == 2:
            return list(owner.frame_ids or [])
        descriptor = owner.non_rigid_alignment
        return list(descriptor.tilt_image_ids if descriptor is not None else [i.id for i in self.parent.images or []])

    @property
    def rows(self):
        images = {image.id: image for image in self.parent.images or []}
        if len(images) != len(self.parent.images or []) or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("image/frame identities must be unique")
        if set(images) != set(self.row_ids) or None in images:
            raise ValueError("payload rows must cover all owning images with stable identities")
        return [images[key] for key in self.row_ids]

    @property
    def reference_entity(self):
        if self.ndim == 2:
            return self.parent
        region, owner, _ = self.resolve()
        volumes = [v for v in region.tomograms or [] if v.id == owner.reference_volume_id]
        if len(volumes) != 1:
            raise ValueError("alignment requires a resolved reference_volume_id -> Tomogram")
        return volumes[0]

    @property
    def reference_frame(self):
        frame = image_frame(self.reference_entity)
        _ = frame.isotropic_spacing
        if tuple(frame.origin_index) != tuple(n // 2 for n in frame.size_px):
            raise ValueError("processing requires the CETS floor(N/2) physical origin")
        if frame.ndim != self.ndim:
            raise ValueError("reference geometry has incorrect dimensionality")
        return frame

    @property
    def image_frames(self):
        frames = []
        for row in self.rows:
            entity = row
            if self.ndim == 2:
                entity = row.model_copy(deep=True)
                for name in ("width", "height", "coordinate_systems", "coordinate_transformations"):
                    if not getattr(entity, name):
                        setattr(entity, name, getattr(self.parent, name))
            frame = image_frame(entity)
            _ = frame.isotropic_spacing
            if tuple(frame.origin_index) != tuple(n // 2 for n in frame.size_px):
                raise ValueError("processing requires the CETS floor(N/2) physical origin")
            frames.append(frame)
        return frames

    @property
    def reference_center_a(self):
        frame = self.reference_frame
        return np.asarray(frame.origin_index) * np.asarray(frame.spacing_a)

    @property
    def image_centers_a(self):
        return np.asarray([np.asarray(f.origin_index) * np.asarray(f.spacing_a) for f in self.image_frames])

    def operators(self):
        """Ordered R,t and row activity. Inactive slots have identity/zero placeholders."""
        count, dim = len(self.row_ids), self.ndim
        matrices = np.repeat(np.eye(dim, dtype=np.float64)[None], count, axis=0)
        shifts = np.zeros((count, 2), dtype=np.float64)
        active = np.zeros(count, dtype=bool)
        owner = self.owner
        entries = owner.frame_alignments if dim == 2 else owner.projection_alignments
        associations = {}
        for entry in entries or []:
            key = entry.frame_id if dim == 2 else entry.tilt_image_id
            if key not in self.row_ids or key in associations:
                raise ValueError("global operator has a missing or duplicate image association")
            associations[key] = entry
        for index, key in enumerate(self.row_ids):
            if key not in associations:
                continue
            if dim == 3:
                matrices[index], shifts[index] = fold_projection(associations[key])
            else:
                shift = associations[key].transform.translation
                if shift is None or len(shift) != 2:
                    raise ValueError("movie global transforms must be two-component Translations")
                shifts[index] = shift
            active[index] = True
        if not (np.isfinite(matrices).all() and np.isfinite(shifts).all()):
            raise ValueError("global operators must be finite")
        return matrices, shifts, active

    def evaluate_global(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.as_tensor(points, dtype=torch.float64)
        if points.ndim != 2 or points.shape[1] != self.ndim or not torch.isfinite(points).all():
            raise ValueError("sample points must be a finite (N,D) array in reference physical Angstrom")
        matrices, shifts, active = self.operators()
        rotation = torch.as_tensor(matrices, dtype=torch.float64, device=points.device)
        translation = torch.as_tensor(shifts, dtype=torch.float64, device=points.device)
        result = torch.einsum("tij,nj->tni", rotation, points)[..., :2] + translation[:, None]
        result[~torch.as_tensor(active, device=points.device)] = 0
        return result

    def validate(self):
        validate_document_references(self.document)
        _ = self.reference_frame, self.image_frames, self.operators()
        orders = [row.acquisition_order for row in self.rows if row.acquisition_order is not None]
        if len(orders) != len(set(orders)):
            raise ValueError("known acquisition-order values must be unique within a series")
        for row in self.rows:
            for name in ("accumulated_dose", "exposure_dose", "exposure_time"):
                value = getattr(row, name, None)
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError(f"{name} must be finite and nonnegative when supplied")
            if row.ctf_metadata is not None:
                for name in ("defocus_u", "defocus_v", "defocus_angle", "phase_shift", "fit_score", "fit_resolution"):
                    value = getattr(row.ctf_metadata, name)
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"CTF {name} must be finite when supplied")

    def digest_record(self, *, point_ids=None, heldout_point_ids=None):
        owner = self.owner
        matrices, shifts, active = self.operators()

        def frame_record(frame):
            return {
                "size_px": list(frame.size_px),
                "spacing_a": list(frame.spacing_a),
                "origin_index": list(frame.origin_index),
            }

        record = {
            "digest_version": DIGEST_VERSION,
            "profile_version": PROFILE_VERSION,
            "region_id": self.resolve()[0].id,
            "alignment_id": owner.id,
            "alignment_name": owner.name,
            "kind": self.kind,
            "parent_id": self.parent.id,
            "row_ids": self.row_ids,
            "activity": active.tolist(),
            "rotation": matrices,
            "shift": shifts,
            "reference_id": self.reference_entity.id,
            "reference_frame": frame_record(self.reference_frame),
            "image_frames": [frame_record(frame) for frame in self.image_frames],
            "acquisition": [
                {
                    name: getattr(row, name, None)
                    for name in (
                        "acquisition_order",
                        "exposure_dose",
                        "accumulated_dose",
                        "exposure_time",
                        "source_start_index",
                        "source_frame_count",
                    )
                }
                for row in self.rows
            ],
        }
        if self.ndim == 3:
            record["ctf_convention"] = {
                "handedness": self.parent.defocus_handedness,
                "slope": self.parent.defocus_slope,
            }
        else:
            record["gauge"] = owner.gauge
            record["reference_frame_id"] = owner.reference_frame_id
            record["raw_frame_count"] = self.parent.raw_frame_count
        descriptor = owner.non_rigid_alignment
        if descriptor is not None:
            record["sampling"] = document_to_dict(descriptor.sampling)
            record["heldout"] = document_to_dict(descriptor.heldout)
            record["channels"] = document_to_dict(descriptor.channels)
            if descriptor.sampling.kind == "particles":
                annotations = {a.id: a for a in self.resolve()[0].annotations or []}
                annotation = annotations[descriptor.sampling.annotation_id]
                record["particle_binding"] = {
                    "reference_id": annotation.source_tomogram_id,
                    "point_ids": annotation.point_ids,
                    "coordinates": annotation.origin3D,
                    "coordinate_systems": document_to_dict(annotation.coordinate_systems),
                    "coordinate_transformations": document_to_dict(annotation.coordinate_transformations),
                    "training_ids": point_ids,
                    "heldout_ids": heldout_point_ids,
                }
        return record

    def digest(self, **kwargs):
        return hashlib.sha256(canonical_json(self.digest_record(**kwargs))).hexdigest()
