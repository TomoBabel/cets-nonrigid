"""Payload integrity and channel tests replacing standalone-IR wire tests."""

import json
from dataclasses import replace
import pytest
import torch
import zarr
from cets_nonrigid import api
from cets_nonrigid.io.store import DeformationStore
from test_cets_exchange import warp_source as source_fixture, warp_bundle as bundle_fixture


@pytest.fixture(scope="module")
def warp_source():
    return source_fixture.__wrapped__()


@pytest.fixture(scope="module")
def warp_bundle(warp_source):
    return bundle_fixture.__wrapped__(warp_source)


def stored(bundle, tmp_path):
    path = tmp_path / "bundle.cets.json"
    api.write_bundle(bundle, path)
    back = api.read_bundle(path)
    root = zarr.open_group(str(tmp_path / "bundle.nonrigid.zarr"), mode="r+")
    return path, back, root, root[back.context.owner.non_rigid_alignment.payload_group]


def test_optional_arrays_and_dimension_names(warp_bundle, tmp_path):
    _, back, _, group = stored(warp_bundle, tmp_path)
    for block in ("training", "heldout"):
        for name in ("points", "projected_residual", "displacement_3d", "ctf_depth", "weights", "observation_valid"):
            torch.testing.assert_close(
                getattr(getattr(back.samples, block), name),
                getattr(getattr(warp_bundle.samples, block), name),
                rtol=0,
                atol=0,
            )
    assert group["displacement_3d"].metadata.dimension_names == ("tilt_image", "sample", "coordinate")
    assert group["ctf_depth"].metadata.dimension_names == ("tilt_image", "sample")


def test_incomplete_store_refused(warp_bundle, tmp_path):
    path, _, root, _ = stored(warp_bundle, tmp_path)
    root.attrs["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        api.read_bundle(path)


def test_unknown_kind_refused(warp_bundle, tmp_path):
    path, _, _, _ = stored(warp_bundle, tmp_path)
    doc = json.loads(path.read_text())
    doc["regions"][0]["alignments"][0]["non_rigid_alignment"]["kind"] = "bogus"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        api.read_bundle(path)


def test_no_legacy_ir_serialization(warp_source, tmp_path):
    with pytest.raises(ValueError, match="standalone numerical IR"):
        DeformationStore.write(tmp_path / "legacy.zarr", warp_source.ir)
    assert not (tmp_path / "legacy.zarr").exists()


@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "missing", "flag"])
def test_optional_channel_validation(warp_bundle, tmp_path, corruption):
    path, back, _, group = stored(warp_bundle, tmp_path)
    if corruption == "shape":
        block = replace(back.samples.training, displacement_3d=back.samples.training.displacement_3d[..., :2])
        back.deformations[back.context.key] = replace(back.samples, training=block)
        with pytest.raises(ValueError):
            back.validate()
    elif corruption == "nonfinite":
        group["displacement_3d"][0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            api.read_bundle(path)
    elif corruption == "missing":
        del group["heldout/displacement_3d"]
        with pytest.raises(ValueError):
            api.read_bundle(path)
    else:
        back.samples.channels.displacement_3d = "none"
        with pytest.raises(ValueError):
            back.validate()


def test_partial_depth_does_not_disable_projection(warp_bundle):
    import copy

    bundle = copy.deepcopy(warp_bundle)
    block = bundle.samples.training
    before = block.observations(bundle.context).clone()
    block.ctf_depth_valid[0, 0] = False
    block.ctf_depth[0, 0] = 0
    bundle.context.owner.non_rigid_alignment.context_digest = bundle.context.digest()
    bundle.validate()
    torch.testing.assert_close(block.observations(bundle.context), before, equal_nan=True)
    from cets_nonrigid.runtime import runtime_ir

    assert runtime_ir(bundle).source_ctf_depth_a is None
