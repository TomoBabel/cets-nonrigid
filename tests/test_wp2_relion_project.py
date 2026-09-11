"""RELION project, source overrides, particles and multi-tomogram CETS workflows."""

import numpy as np
import pytest
import starfile
import torch
import test_r4_a2r as helpers
from click.testing import CliRunner
from test_wp1_cli import _synthetic_run, invoke, OPTICS
from cets_nonrigid.cli import main
from cets_nonrigid import api


def _picks_dir(tmp_path, stems, n=3, grid=False):
    d = tmp_path / "picks"
    d.mkdir()
    vol_a = np.array(helpers.TOMO) * helpers.PIX
    if grid:  # 27 spread picks: enough support for a movement-grid fit
        fracs = [[x, y, z] for x in (0.25, 0.5, 0.75) for y in (0.25, 0.5, 0.75) for z in (0.3, 0.5, 0.7)]
    else:
        fracs = [[0.3, 0.4, 0.5], [0.6, 0.5, 0.4], [0.5, 0.6, 0.6]][:n]
    for stem in stems:
        (d / f"{stem}.txt").write_text("\n".join(" ".join(f"{v:.1f}" for v in vol_a * f) for f in fracs))
    return d


def test_relion_source_overrides_gates(tmp_path):
    from cets_nonrigid.convert_relion import RelionSourceOverrides, apply_source_overrides
    from cets_nonrigid.io.relion_star import RelionTomogramData

    t = 4
    tomo = RelionTomogramData(
        name="T",
        voltage_kv=300,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        hand=1,
        pixel_size_a=2.0,
        tomo_dims_px=(48, 48, 12),
        image_dims_px=(96, 96),
        xtilt_deg=torch.zeros(t),
        ytilt_deg=torch.tensor([-30.0, -10.0, 10.0, 30.0]),
        zrot_deg=torch.zeros(t),
        xshift_a=torch.zeros(t),
        yshift_a=torch.zeros(t),
        pre_exposure=torch.arange(t, dtype=torch.float64),
        nominal_stage_angle_deg=torch.zeros(t),
    )
    d, notes = apply_source_overrides(tomo, RelionSourceOverrides(tomo_dims_px=(96, 96, 24), pad_px=4))
    assert d.tomo_dims_px == (104, 104, 24) and any("rlnTomoSize" in n for n in notes)
    with pytest.raises(ValueError, match="even"):
        apply_source_overrides(tomo, RelionSourceOverrides(pad_px=3))
    d, _ = apply_source_overrides(tomo, RelionSourceOverrides(nominal_from_ytilt=True))
    assert torch.equal(d.nominal_stage_angle_deg, tomo.ytilt_deg.to(torch.float64))
    with pytest.raises(ValueError, match="ascending"):
        apply_source_overrides(tomo, RelionSourceOverrides(nominal_angles=[30.0, 10.0, -10.0, -30.0]))
    with pytest.raises(ValueError, match="std"):
        apply_source_overrides(tomo, RelionSourceOverrides(nominal_angles=[-30.0, -10.0, 15.0, 30.0]))


def test_dims_override_loader(tmp_path, ts1_xml_path):
    from cets_nonrigid.io.warp_xml import DimsOverride, load_warp_tiltseries

    text = ts1_xml_path.read_text()
    import re

    zeroed = re.sub(r'ImageDimensionsAngstrom="[^"]*"', 'ImageDimensionsAngstrom="0, 0"', text, count=1)
    zeroed = re.sub(r'VolumeDimensionsAngstrom="[^"]*"', 'VolumeDimensionsAngstrom="0, 0, 0"', zeroed, count=1)
    p = tmp_path / "zero.xml"
    p.write_text(zeroed)
    with pytest.raises(ValueError, match="bad ImageDimensionsAngstrom"):
        load_warp_tiltseries(p)
    ref = load_warp_tiltseries(ts1_xml_path).ts
    got = load_warp_tiltseries(
        p,
        dims_override=DimsOverride(
            image_a=tuple(ref.image_dimensions_physical.tolist()),
            volume_a=tuple(ref.volume_dimensions_physical.tolist()),
        ),
    ).ts
    assert torch.equal(got.image_dimensions_physical, ref.image_dimensions_physical)
    assert torch.equal(got.volume_dimensions_physical, ref.volume_dimensions_physical)
    assert p.read_text() == zeroed  # nothing edited on disk


def test_batch_command_removed():
    res = CliRunner().invoke(main, ["batch", "--help"])
    assert res.exit_code != 0


def make_project(tmp_path, count=2, **target):
    run, stems = _synthetic_run(tmp_path, n=count)
    picks = _picks_dir(tmp_path, stems, grid=True)
    output = tmp_path / "relion"
    result = invoke(
        tmp_path,
        "aretomo3-to-relion",
        run,
        output,
        source_options={"tomo_size_px": [96, 96, 24], "particles": str(picks), "defocus_handedness": -1, **OPTICS},
        target_options=target,
    )
    assert result.exit_code == 0, result.output
    return run, stems, output


def test_multi_series_relion_and_particle_attributes(tmp_path):
    _, stems, output = make_project(tmp_path)
    global_table = starfile.read(output / "tomograms.star")
    assert list(global_table.rlnTomoName) == stems
    parts = starfile.read(output / "particles.star", always_dict=True)
    assert len(parts["particles"]) == 54
    assert len(set(parts["particles"].rlnTomoParticleName)) == 54
    parts["particles"]["rlnRandomSubset"] = [1, 2] * 27
    parts["particles"]["rlnClassNumber"] = [3] * 54
    starfile.write(parts, output / "particles.star", overwrite=True)
    bundle = api.to_cets("relion", output / "optimisation_set.star", tomo_name="TS_01", image_size_px=(96, 96))
    annotation = bundle.context.resolve()[0].annotations[0]
    assert {a.name for a in annotation.point_attributes} == {"half_set", "class_number"}
    restored = api.fit(bundle, "relion")
    api.export_native(restored, tmp_path / "restored")
    newparts = starfile.read(tmp_path / "restored/particles.star", always_dict=True)["particles"]
    assert list(newparts.rlnClassNumber) == [3] * 27
    assert list(newparts.rlnRandomSubset) == [1, 2] * 13 + [1]


@pytest.mark.parametrize("target", ["warp", "aretomo3"])
def test_all_relion_tomograms_to_native_projects(tmp_path, target):
    _, stems, output = make_project(tmp_path)
    destination = tmp_path / target
    result = invoke(
        tmp_path,
        "relion-to-" + target,
        output / "optimisation_set.star",
        destination,
        source_options={"image_size_px": [96, 96]},
        target_options={"movement_grid": [2, 2]} if target == "warp" else {"patch_grid": [2, 2]},
    )
    assert result.exit_code == 0, result.output
    files = list(destination.rglob("*.xml" if target == "warp" else "*.aln"))
    assert sorted(p.stem for p in files) == stems


def test_global_only_and_py2rely_placeholder(tmp_path):
    run, _ = _synthetic_run(tmp_path, n=1)
    output = tmp_path / "stream"
    result = invoke(
        tmp_path,
        "aretomo3-to-relion",
        run,
        output,
        source_options={"tomo_size_px": [96, 96, 24], "defocus_handedness": 1, **OPTICS},
        target_options={"no_particles": True, "placeholder_stack": True, "tilt_series_uri": "s3://bucket/TS_01.zarr"},
    )
    assert result.exit_code == 0, result.output
    assert not (output / "particles.star").exists()
    assert (output / "tilt_series/TS_01_placeholder.mrcs").exists()
    assert "s3://bucket/TS_01.zarr" in (output / "tilt_series/TS_01.star").read_text()


def test_ndjson_and_text_voxel_inputs(tmp_path):
    run, _ = _synthetic_run(tmp_path, n=1)
    text = tmp_path / "picks.txt"
    text.write_text("12 14 6\n20 18 5\n")
    ndjson = tmp_path / "picks.ndjson"
    ndjson.write_text('{"location":{"x":12,"y":14,"z":6}}\n{"location":{"x":20,"y":18,"z":5}}\n')
    bundles = [
        api.to_cets("aretomo3", run / "TS_01.aln", tomo_size_px=(96, 96, 24), particles=str(p), particles_voxel="4.0")
        for p in (text, ndjson)
    ]
    torch.testing.assert_close(bundles[0].samples.training.points, bundles[1].samples.training.points)
