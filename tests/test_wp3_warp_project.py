"""Warp frame materialization and project-readiness gates."""
import numpy as np
import pytest
from test_wp1_cli import _synthetic_run, invoke, OPTICS
from test_wp2_relion_project import _picks_dir
from cets_nonrigid import api

def test_frames_helpers(tmp_path):
    import mrcfile

    from cets_nonrigid.io.frames import frames_from_images, frames_from_stack, resolve_average_paths, stack_from_frames

    stack = tmp_path / "s.mrc"
    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with mrcfile.new(str(stack)) as m:
        m.set_data(data)
        m.voxel_size = 2.0
    fdir = tmp_path / "frames"
    out = frames_from_stack(stack, fdir, ["a", "b", "c"], order=[2, 0, 1], pixel_size_a=2.0)
    assert [p.name for p in out] == ["a.mrc", "b.mrc", "c.mrc"] and out[0].is_symlink()
    with mrcfile.open(str(fdir / "average" / "a.mrc")) as m:
        np.testing.assert_array_equal(m.data, data[2])
        assert float(m.voxel_size.x) == 2.0
    # re-run reuses, wrong shape refuses
    frames_from_stack(stack, fdir, ["a"], order=[0], pixel_size_a=2.0)
    with mrcfile.new(str(fdir / "average" / "d.mrc")) as m:
        m.set_data(np.zeros((9, 9), dtype=np.float32))
    with pytest.raises(FileExistsError):
        frames_from_stack(stack, fdir, ["d"], order=[0])
    rebuilt = stack_from_frames([fdir / "average" / n for n in ("a.mrc", "b.mrc", "c.mrc")], tmp_path / "r.mrc",
                                pixel_size_a=2.0, order=[1, 2, 0])
    with mrcfile.open(str(rebuilt)) as m:
        np.testing.assert_array_equal(m.data, data)  # b=slice0, c=slice1, a=slice2
    xml = tmp_path / "proj" / "warp_tiltseries" / "x.xml"
    (tmp_path / "proj" / "tomostar").mkdir(parents=True)
    xml.parent.mkdir()
    paths = resolve_average_paths(xml, ["../frames/a.mrc", ""], frames_dir=None)
    assert paths[0] == (tmp_path / "proj" / "frames" / "average" / "a.mrc").resolve() and paths[1] is None
    linked = frames_from_images([fdir / "average" / "a.mrc"], tmp_path / "f2", ["z"])
    assert linked[0].is_symlink() and (tmp_path / "f2" / "average" / "z.mrc").is_symlink()


def test_warp_project_images_feed_relion(tmp_path):
    run, stems = _synthetic_run(tmp_path, n=1)
    warp = tmp_path/'warp'
    result = invoke(tmp_path, 'aretomo3-to-warp', run, warp,
        source_options={'tomo_size_px':[96,96,24], 'defocus_handedness':-1, **OPTICS})
    assert result.exit_code == 0, result.output
    # Warp C# splits MoviePath on newlines without dropping empty entries.
    from lxml import etree
    node = etree.parse(str(warp / "warp_tiltseries/TS_01.xml")).getroot().find("MoviePath")
    assert len(node.text.split("\n")) == 9
    assert all(node.text.split("\n"))
    picks = _picks_dir(tmp_path, stems, grid=True)
    result = invoke(tmp_path, 'warp-to-relion', warp/'warp_tiltseries', tmp_path/'relion',
        source_options={'particles':str(picks), 'tilt_stack_dir':str(run)})
    assert result.exit_code == 0, result.output
    assert (tmp_path/'relion/particles.star').is_file()

def test_zero_warp_dimensions_restored_without_source_edit(tmp_path):
    import re
    from test_r4_w2r import _write_synthetic_xml, PIX, IMG_A, VOL_A
    from cets_nonrigid.io.warp_xml import DimsOverride
    xml = _write_synthetic_xml(tmp_path/'source.xml')
    original = re.sub(r'ImageDimensionsAngstrom="[^"]*"', 'ImageDimensionsAngstrom="0, 0"', xml.read_text())
    original = re.sub(r'VolumeDimensionsAngstrom="[^"]*"', 'VolumeDimensionsAngstrom="0, 0, 0"', original)
    xml.write_text(original)
    native = api.load_native('warp', xml, pixel_size_a=PIX, discover_adjacent=False,
        dims_override=DimsOverride(image_a=IMG_A, volume_a=VOL_A))
    assert native.context.reference_frame.size_px[2] > 0
    assert xml.read_text() == original
