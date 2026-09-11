"""M0 smoke tests: the dependency stack works on this Python version and the
golden fixtures load through the depended-upon packages."""

import torch


def test_imports():
    import lxml  # noqa: F401
    import numpy  # noqa: F401
    import pydantic
    import zarr

    import cets_nonrigid
    from cets_nonrigid.conventions import sign_conventions

    assert cets_nonrigid.__version__
    assert zarr.__version__.startswith("3")
    assert pydantic.VERSION.startswith("2")
    assert sign_conventions()["warp_tilt_angle_sign"] == -1


def test_warpylib_loads_ts1(ts1_xml_path):
    from warpylib import TiltSeries

    ts = TiltSeries(path=str(ts1_xml_path))
    assert ts.n_tilts == 41
    # Populated movement grids (6x4x41 in this fixture)
    assert tuple(ts.grid_movement_x.dimensions)[-1] == ts.n_tilts
    assert ts.grid_movement_x.values.abs().max() > 0
    # Nonzero level angles — the reason w2a globals are fitted
    assert ts.level_angle_x != 0.0 or ts.level_angle_y != 0.0
    # Forward model runs
    pts = torch.tensor([[100.0, 200.0, 300.0]])
    out = ts.get_position_in_all_tilts_single(pts)
    assert out.shape == (1, ts.n_tilts, 3)
    assert torch.isfinite(out).all()


def test_cryoet_alignment_loads_aln(test_aln_path):
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    aln = AreTomo3ALN.from_file(str(test_aln_path))
    assert aln.NumPatches == 2
    assert len(aln.DarkFrames) == 2
    assert aln.z_indices() == [0, 2, 3, 4, 5, 7]
    # NOTE: this fixture is a truncated example (4 global rows, 6 local rows) —
    # it does NOT satisfy the real-output invariant
    # len(locals) == len(globals) * NumPatches. cets_nonrigid's io.aln adapter will
    # enforce that invariant on real files; complete-.aln tests need real data (M1+).
    assert aln.LocalAlignments is not None and len(aln.LocalAlignments) > 0
    la = aln.LocalAlignments[0]
    assert la.is_reliable == 1.0 and la.center_x != 0.0
