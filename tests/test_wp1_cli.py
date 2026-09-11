"""Native project CLI workflows through the CETS public API."""
import json
import numpy as np
from click.testing import CliRunner
from cets_nonrigid.cli import main

def _synthetic_run(tmp_path, n=2):
    """n synthetic AreTomo3 series with stem-adjacent stack/_TLT/_CTF files."""
    import mrcfile
    import test_r4_a2r as helpers

    run = tmp_path / "run"
    run.mkdir(parents=True)
    stems = []
    for i in range(n):
        stem = f"TS_{i + 1:02d}"
        _, tilts_raw = helpers._write_synthetic_aln(run / f"{stem}.aln")
        acq = np.argsort(np.argsort(np.abs(tilts_raw), kind="stable")) + 1
        (run / f"{stem}_TLT.txt").write_text(
            "".join(f"{tilts_raw[k]:8.2f}  {int(acq[k]):4d}  {1.5:8.2f}\n" for k in range(helpers.T_RAW))
        )
        (run / f"{stem}_CTF.txt").write_text(
            "".join(f"{k + 1:4d} {21000 + 100 * k:8.2f} {20500 + 100 * k:8.2f} {30.0:8.2f} "
                    f"{0.0:9.4f} {0.1:8.4f} {5.0:8.4f} {1:3d}\n" for k in range(helpers.T_RAW))
        )
        with mrcfile.new(run / f"{stem}.mrc") as m:
            m.set_data(np.zeros((helpers.T_RAW, helpers.IMG[1], helpers.IMG[0]), dtype=np.float32))
            m.voxel_size = helpers.PIX
        stems.append(stem)
    return run, stems


OPTICS = {"voltage_kv":300, "cs_mm":2.7, "amplitude_contrast":0.07}
def invoke(tmp_path, pair, source, output, *, source_options=None, target_options=None, project_options=None):
    config = tmp_path / (output.name + '.config.json')
    config.write_text(json.dumps({'source':source_options or {}, 'target':target_options or {}, 'project':project_options or {}}))
    return CliRunner().invoke(main, ['convert', pair, str(source), '-o', str(output), '--config', str(config)])

def test_multiple_aretomo_to_warp_project(tmp_path):
    run, stems = _synthetic_run(tmp_path)
    output = tmp_path/'warp'
    result = invoke(tmp_path, 'aretomo3-to-warp', run, output, source_options={'tomo_size_px':[96,96,24], **OPTICS})
    assert result.exit_code == 0, result.output
    assert sorted(p.stem for p in (output/'warp_tiltseries').glob('*.xml')) == stems
    assert (output/'cets-nonrigid-report.json').is_file()
    before = (output/'warp_tiltseries/TS_01.xml').read_bytes()
    again = invoke(tmp_path, 'aretomo3-to-warp', run, output, source_options={'tomo_size_px':[96,96,24], **OPTICS})
    assert again.exit_code != 0 and 'refusing to replace' in again.output
    assert before == (output/'warp_tiltseries/TS_01.xml').read_bytes()

def test_single_native_file_and_missing_geometry(tmp_path):
    run, _ = _synthetic_run(tmp_path, n=1)
    output = tmp_path/'one.xml'
    missing = invoke(tmp_path, 'aretomo3-to-warp', run, output)
    assert missing.exit_code != 0 and ('tomo_size' in missing.output or 'tomo-size' in missing.output)
    result = invoke(tmp_path, 'aretomo3-to-warp', run, output, source_options={'tomo_size_px':[96,96,24]})
    assert result.exit_code == 0, result.output
    assert output.is_file()
    run2, _ = _synthetic_run(tmp_path/'b')
    bad = invoke(tmp_path, 'aretomo3-to-warp', run2, tmp_path/'many.xml', source_options={'tomo_size_px':[96,96,24]})
    assert bad.exit_code != 0 and 'single file' in bad.output

def test_warp_back_to_aretomo_project(tmp_path):
    import mrcfile
    run, _ = _synthetic_run(tmp_path, n=1)
    warp = tmp_path/'warp'
    first = invoke(tmp_path, 'aretomo3-to-warp', run, warp, source_options={'tomo_size_px':[96,96,24], **OPTICS, 'defocus_handedness':-1})
    assert first.exit_code == 0, first.output
    output = tmp_path/'aretomo'
    second = invoke(tmp_path, 'warp-to-aretomo3', warp/'warp_tiltseries', output,
        source_options={'tilt_stack_dir':str(run)}, target_options={'patch_grid':[2,2]})
    assert second.exit_code == 0, second.output
    assert (output/'TS_01.aln').is_file() and (output/'TS_01_TLT.txt').is_file()
    with mrcfile.open(output/'TS_01.mrc') as stack:
        assert stack.data.shape[1:] == (96,96)
