"""Apply CETS scientific metadata after the validated numerical target fit."""

from __future__ import annotations

from pathlib import Path
import torch

from cets_nonrigid.metadata import get_ctf, get_optics


def apply_scientific_metadata(bundle, target, result, output, options):
    context = bundle.context
    if context.ndim == 2:
        if target == "mcaln":
            from cets_nonrigid.io.motion_txt import McAln, McAlnFrame

            parsed = McAln.from_file(output)
            active = context.operators()[2]
            indices = [i for i, value in enumerate(active) if value]
            parsed.raw_frame_count = context.parent.raw_frame_count or len(context.rows)
            parsed.integrated_frame_count = len(context.rows)
            parsed.frames = [
                McAlnFrame(
                    integrated_index=i,
                    source_start=row.source_start_index if row.source_start_index is not None else i,
                    source_count=row.source_frame_count or 1,
                    included=bool(active[i]),
                    aligned_index=indices.index(i) if active[i] else -1,
                )
                for i, row in enumerate(context.rows)
            ]
            parsed.validate_consistency()
            Path(output).write_text(parsed.to_string())
        return {
            "ctf_disposition": "frame motion only",
            "n_frames": len(context.rows),
            "fraction_frames": options.get("fraction_frames", 1.0) if target == "warp-movie" else None,
            "excluded_frame_ids": [key for key, active in zip(context.row_ids, context.operators()[2]) if not active],
        }
    mapping = result.source_row_map
    emitted = sorted((target_i, source_i) for source_i, target_i in enumerate(mapping) if target_i >= 0)
    metrics = {"source_to_target_rows": mapping}
    ctf = get_ctf(context, [src for _, src in emitted])
    if target == "warp":
        from cets_nonrigid.io.warp_xml import load_warp_tiltseries, write_alignment_into_template
        from cets_nonrigid.ctf import tiltctf_from_warp, tiltctf_to_warp

        ts = result.fit.ts
        optics = get_optics(context)
        for name, key in (("voltage", "voltage_kv"), ("cs", "cs_mm"), ("amplitude", "amplitude_contrast")):
            if optics[key] is not None:
                setattr(ts.ctf, name, optics[key])
        hand = context.parent.defocus_handedness
        if "angles_inverted" not in options and hand is not None:
            ts.are_angles_inverted = hand == -1
        if ctf is not None:
            current = tiltctf_from_warp(ts)
            if current is None:
                from cets_nonrigid.ctf import TiltCtf

                current = TiltCtf(
                    torch.full((ts.n_tilts,), 20000.0),
                    torch.full((ts.n_tilts,), 20000.0),
                    torch.zeros(ts.n_tilts),
                    torch.zeros(ts.n_tilts),
                )
            for k, (target_i, _) in enumerate(emitted):
                for name in ("defocus_u_a", "defocus_v_a", "angle_deg", "phase_deg"):
                    getattr(current, name)[target_i] = getattr(ctf, name)[k]
            tiltctf_to_warp(ts, current)
        # This is an explicit output being built in private scratch space, never a source snapshot.
        data = Path(output).read_bytes()
        temporary = Path(output).with_suffix(".ctf.xml")
        write_alignment_into_template(data, ts, temporary, with_ctf=True)
        load_warp_tiltseries(temporary)
        temporary.replace(output)
        metrics["ctf_disposition"] = (
            "CETS CTF on mapped rows" if ctf is not None else "target template or explicit synthesis placeholders"
        )
    elif target == "aretomo3" and ctf is not None:
        from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile, AreTomoCtfRow

        if len(emitted) != result.aln.RawSize[2]:
            metrics["ctf_disposition"] = "CTF not emitted: target has unmapped raw rows; supply complete row metadata"
        else:
            hand = context.parent.defocus_handedness
            if hand is None:
                metrics["ctf_disposition"] = "CTF not emitted: defocus handedness is unknown"
            else:
                rows = [
                    AreTomoCtfRow(
                        micrograph=i + 1,
                        df_max_a=float(ctf.defocus_u_a[i]),
                        df_min_a=float(ctf.defocus_v_a[i]),
                        azimuth_deg=float(ctf.angle_deg[i]),
                        phase_rad=float(torch.deg2rad(ctf.phase_deg[i])),
                        score=float(ctf.score[i]) if ctf.score is not None else 0.0,
                        res_a=float(ctf.res_a[i]) if ctf.res_a is not None else 999.99,
                        df_hand=hand,
                    )
                    for i in range(len(emitted))
                ]
                AreTomoCtfFile(rows=rows).to_file(Path(output).with_name(Path(output).stem + "_CTF.txt"))
                metrics["ctf_disposition"] = "CETS CTF in emitted raw-row order"
    else:
        metrics["ctf_disposition"] = "no complete CETS per-image CTF"
    return metrics


def depth_comparison(bundle, result):
    """Compare emitted Warp depth with CETS observations using independent masks."""
    from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

    context, data = bundle.context, bundle.samples
    if data.channels.ctf_depth != "present":
        return {"ctf_depth_validation": "not_evaluated", "ctf_depth_deviation_rms_a": None}
    model = WarpTiltSeriesModel(result.fit.ts)
    mapping = result.source_row_map
    pairs = [(src, dst) for src, dst in enumerate(mapping) if dst >= 0 and context.operators()[2][src]]
    metrics = {}
    for label, block in (("training", data.training), ("heldout", data.heldout)):
        if not block.count or not pairs:
            metrics[label] = {"status": "not_evaluated", "rms_a": None, "maximum_a": None}
            continue
        points = block.points + torch.tensor(context.reference_center_a, dtype=torch.float64)
        predicted = model.ctf_depth(points).to(torch.float64)
        src, dst = zip(*pairs)
        valid = block.ctf_depth_valid[list(src)] & block.sample_valid[None]
        error = (predicted[list(dst)] - block.ctf_depth[list(src)].to(torch.float64))[valid]
        metrics[label] = {
            "status": "evaluated" if error.numel() else "not_evaluated",
            "rms_a": float(torch.sqrt(torch.mean(error.square()))) if error.numel() else None,
            "maximum_a": float(error.abs().max()) if error.numel() else None,
        }
    return {"ctf_depth_validation": metrics}
