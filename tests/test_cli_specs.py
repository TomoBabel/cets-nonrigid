"""The CLI's spec grammars (SOURCE tokens, --particles, --particles-voxel),
frame-extension inference, symlink safety, and the option-surface guard."""

from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

from cets_nonrigid.cli import main
from cets_nonrigid.cli.specs import (
    ParticleSpec,
    PortalSource,
    expand_sources,
    infer_frames_ext,
    parse_particles_spec,
    parse_particles_voxel,
    parse_portal_source,
)

TILT_SERIES_COMMANDS = ("a2w", "a2r", "w2a", "w2r", "r2w", "r2a")


# ---------------------------------------------------------------------------
# SOURCE tokens
# ---------------------------------------------------------------------------


def test_portal_source_forms():
    assert parse_portal_source("portal:10445") == PortalSource(10445)
    assert parse_portal_source("portal:10445/TS_105_5") == PortalSource(10445, "TS_105_5")
    s = parse_portal_source("portal:10445/TS_1-a@alignment=100,voxel=4.99")
    assert s == PortalSource(10445, "TS_1-a", 100, 4.99) and str(s) == "portal:10445/TS_1-a@alignment=100,voxel=4.99"
    for bad in ("portal:", "portal:abc", "portal:10445@foo=1", "portal:10445/x@alignment=z"):
        with pytest.raises(click.UsageError):
            parse_portal_source(bad)


def test_expand_sources_files_dirs_portal(tmp_path):
    (tmp_path / "a.aln").write_text("")
    (tmp_path / "b.aln").write_text("")
    (tmp_path / "b_EVN.mrc").write_text("")
    got = expand_sources([str(tmp_path), str(tmp_path / "a.aln"), "portal:10445/TS_1"], "aln", portal=True)
    assert [g.name for g in got if isinstance(g, Path)] == ["a.aln", "b.aln"]  # deduplicated, globbed by kind
    assert got[-1] == PortalSource(10445, "TS_1")
    with pytest.raises(click.UsageError, match="AreTomo3 sources"):
        expand_sources(["portal:10445"], "xml")
    with pytest.raises(click.UsageError, match="not a file"):
        expand_sources([str(tmp_path / "missing.aln")], "aln")
    with pytest.raises(click.UsageError, match="no files match"):
        expand_sources([str(tmp_path)], "xml")


# ---------------------------------------------------------------------------
# --particles / --particles-voxel
# ---------------------------------------------------------------------------


def test_particles_spec_forms(tmp_path):
    f = tmp_path / "picks.star"
    f.write_text("")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    assert parse_particles_spec(str(f)) == ParticleSpec("file", path=f)
    assert parse_particles_spec(str(tmp_path)) == ParticleSpec("dir", path=tmp_path)
    c = parse_particles_spec(f"copick:{cfg}#ribosome:alice/manual-001")
    assert c == ParticleSpec("copick", config=cfg, uri="ribosome:alice/manual-001")
    # only the first '#' splits: copick regex/glob patterns survive
    c = parse_particles_spec(f"copick:{cfg}#re:mt.*:u*/s#1")
    assert c.uri == "re:mt.*:u*/s#1"
    assert parse_particles_spec("portal:cytosolic ribosome") == ParticleSpec("portal", object_name="cytosolic ribosome")
    assert parse_particles_spec("portal:69549") == ParticleSpec("portal", annotation_id=69549)
    for bad in ("copick:nope.json#a:b/c", f"copick:{cfg}", f"copick:{cfg}#", "portal:", str(tmp_path / "missing.star")):
        with pytest.raises(click.UsageError):
            parse_particles_spec(bad)


def test_particles_voxel_forms(tmp_path):
    assert parse_particles_voxel("4.99") == 4.99
    assert parse_particles_voxel(str(tmp_path)) == tmp_path
    for bad in ("0", "-1", str(tmp_path / "nope")):
        with pytest.raises(click.UsageError):
            parse_particles_voxel(bad)


def test_dose_per_tilt_accepts_a_number_or_file(tmp_path):
    from cets_nonrigid.cli._series_options import _dose_per_tilt_callback

    assert _dose_per_tilt_callback(None, None, "3.87") == 3.87
    assert _dose_per_tilt_callback(None, None, "FILE") == "file"
    assert _dose_per_tilt_callback(None, None, None) is None
    with pytest.raises(click.BadParameter):
        _dose_per_tilt_callback(None, None, "lots")


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def test_infer_frames_ext(tmp_path):
    names = ["TS_01_001", "TS_01_002"]
    assert infer_frames_ext(tmp_path, names) is None
    assert infer_frames_ext(tmp_path / "missing", names) is None
    (tmp_path / "TS_01_001.eer").write_text("")
    assert infer_frames_ext(tmp_path, names) == ".eer"  # a partial set still names the extension
    (tmp_path / "TS_01_001.mrc").write_text("")
    assert infer_frames_ext(tmp_path, names) == ".mrc"  # .mrc wins when both exist


def test_frames_from_stack_never_replaces_a_regular_file(tmp_path):
    import mrcfile

    from cets_nonrigid.io.frames import frames_from_stack

    stack = tmp_path / "stack.mrc"
    with mrcfile.new(str(stack)) as m:
        m.set_data(np.zeros((2, 8, 8), dtype=np.float32))
        m.voxel_size = 2.0
    frames = tmp_path / "frames"
    frames.mkdir()
    real = frames / "TS_01_001.eer"
    real.write_bytes(b"real movie")
    out = frames_from_stack(stack, frames, ["TS_01_001", "TS_01_002"], ext=".eer", pixel_size_a=2.0)
    assert real.read_bytes() == b"real movie" and not real.is_symlink()  # untouched
    assert (frames / "TS_01_002.eer").is_symlink() and (frames / "average" / "TS_01_001.mrc").exists()
    assert out == [real, frames / "TS_01_002.eer"]
    # a stale symlink of ours is refreshed
    (frames / "TS_01_002.eer").unlink()
    (frames / "TS_01_002.eer").symlink_to("nowhere")
    frames_from_stack(stack, frames, ["TS_01_002"], ext=".eer", pixel_size_a=2.0)
    assert (frames / "TS_01_002.eer").resolve() == (frames / "average" / "TS_01_002.mrc").resolve()


# ---------------------------------------------------------------------------
# the option surface
# ---------------------------------------------------------------------------


def _options(*command):
    result = CliRunner().invoke(main, [*command, "--help"])
    assert result.exit_code == 0, result.output
    return [ln.strip().split()[0] for ln in result.output.splitlines() if ln.startswith("  -")]


@pytest.mark.parametrize("group", ["to-cets", "from-cets", "convert", "fit"])
def test_per_format_commands_stay_small(group):
    leaves = main.commands[group].commands
    assert len(leaves) == (12 if group == "convert" else 6)
    for name in leaves:
        opts = _options(group, name)
        assert len(opts) <= 25, (group, name, opts)
        assert "--config" in opts and "--fail-fast" in opts
        help_result = CliRunner().invoke(main, [group, name, "--help"])
        assert "SOURCES" in help_result.output.splitlines()[0]


def test_directional_aliases_and_legacy_store_commands_are_absent():
    for name in (*TILT_SERIES_COMMANDS, "m2w", "w2m", "dump-ir", "batch"):
        assert name not in main.commands
    assert "--particles" in _options("to-cets", "warp")
    assert "--particles" in _options("convert", "aretomo3-to-relion")
    assert "--overwrite" not in _options("convert", "aretomo3-to-warp")
