"""Scientific metadata between native readers and first-class CETS fields."""

from __future__ import annotations

import torch
from cets_data_model.models import models as m
from cets_nonrigid.ctf import TiltCtf


def set_optics(document, parent, *, voltage_kv=None, cs_mm=None, amplitude_contrast=None):
    if all(v is None for v in (voltage_kv, cs_mm, amplitude_contrast)):
        return
    iid, sid = parent.id + "_instrument", parent.id + "_session"
    document.instruments.append(m.Instrument(id=iid, voltage=voltage_kv))
    document.acquisition_sessions.append(
        m.AcquisitionSession(
            id=sid, instrument_id=iid, spherical_aberration=cs_mm, amplitude_contrast=amplitude_contrast
        )
    )
    parent.acquisition_session_id = sid


def get_optics(context):
    if context.ndim == 3:
        sid = context.parent.acquisition_session_id
    else:
        sid = next(
            s.acquisition_session_id
            for s in context.resolve()[0].movie_stack_collection.movie_stacks
            if any(stack.id == context.parent.id for stack in s.stacks)
        )
    session = next((s for s in context.document.acquisition_sessions if s.id == sid), None)
    instrument = next((i for i in context.document.instruments if session and i.id == session.instrument_id), None)
    return dict(
        voltage_kv=instrument.voltage if instrument else None,
        cs_mm=session.spherical_aberration if session else None,
        amplitude_contrast=session.amplitude_contrast if session else None,
    )


def set_ctf(context, ctf, *, row_indices=None):
    if ctf is None:
        return
    row_indices = row_indices if row_indices is not None else list(range(len(context.rows)))
    if ctf.n_tilts != len(row_indices):
        raise ValueError("CTF count disagrees with its image row mapping")
    for i, index in enumerate(row_indices):
        context.rows[index].ctf_metadata = m.CTFMetadata(
            defocus_u=float(ctf.defocus_u_a[i]),
            defocus_v=float(ctf.defocus_v_a[i]),
            defocus_angle=float(ctf.angle_deg[i]),
            phase_shift=float(ctf.phase_deg[i]),
            fit_score=float(ctf.score[i]) if ctf.score is not None else None,
            fit_resolution=float(ctf.res_a[i]) if ctf.res_a is not None else None,
        )


def get_ctf(context, rows=None):
    rows = list(range(len(context.rows))) if rows is None else rows
    entries = [context.rows[i].ctf_metadata for i in rows]
    if any(
        e is None or any(getattr(e, key) is None for key in ("defocus_u", "defocus_v", "defocus_angle", "phase_shift"))
        for e in entries
    ):
        return None
    return TiltCtf(
        **{
            name: torch.tensor([getattr(e, field) for e in entries], dtype=torch.float64)
            for name, field in (
                ("defocus_u_a", "defocus_u"),
                ("defocus_v_a", "defocus_v"),
                ("angle_deg", "defocus_angle"),
                ("phase_deg", "phase_shift"),
            )
        },
        score=torch.tensor([e.fit_score for e in entries], dtype=torch.float64)
        if all(e.fit_score is not None for e in entries)
        else None,
        res_a=torch.tensor([e.fit_resolution for e in entries], dtype=torch.float64)
        if all(e.fit_resolution is not None for e in entries)
        else None,
        **get_optics(context),
    )


def discover_native_metadata(native, extra):
    context, model = native.context, native.ir.native_model
    if context.ndim != 3:
        import json

        motion = native.ir.native_data if native.source == "relion-motion" else None

        def value(key):
            return extra[key] if extra.get(key) is not None else getattr(motion, key, None)

        dose, pre = value("dose_rate"), value("pre_exposure")
        for i, frame in enumerate(context.rows):
            frame.exposure_dose = dose
            frame.accumulated_dose = pre + i * dose if pre is not None and dose is not None else None
        series = next(
            s
            for s in context.resolve()[0].movie_stack_collection.movie_stacks
            if any(stack.id == context.parent.id for stack in s.stacks)
        )
        set_optics(context.document, series, voltage_kv=value("voltage_kv"))
        for key in ("eer_grouping", "eer_upsampling"):
            if value(key) is not None:
                context.owner.provenance.parameters.append(
                    m.NativeParameter(name=key, value_json=json.dumps(value(key)))
                )
        if motion is not None and motion.movie_name:
            context.parent.path = motion.movie_name
        return
    ctf, optics = None, {}
    if native.source == "warp":
        from cets_nonrigid.ctf import tiltctf_from_warp

        ctf = tiltctf_from_warp(model.ts)
        optics = dict(
            voltage_kv=float(model.ts.ctf.voltage),
            cs_mm=float(model.ts.ctf.cs),
            amplitude_contrast=float(model.ts.ctf.amplitude),
        )
        context.owner.provenance.dropped_information.extend(
            [
                "GridAngle* is outside the validated position model",
                "MagnificationCorrection is outside the validated position model",
                "non-unit runtime SizeRoundingFactors is unsupported",
            ]
        )
    elif native.source == "relion":
        data = native.ir.native_data
        ctf = data.ctf
        optics = dict(voltage_kv=data.voltage_kv, cs_mm=data.cs_mm, amplitude_contrast=data.amplitude_contrast)
    elif extra.get("ctf_file") is not None:
        from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile
        from cets_nonrigid.ctf import tiltctf_from_aretomo

        parsed = AreTomoCtfFile.from_file(extra["ctf_file"])
        # Validated helper maps sorted AreTomo raw rows into the source's Warp row order.
        ctf = tiltctf_from_aretomo(parsed, torch.tensor(native.ir.meta.projection_angle_deg, dtype=torch.float64))
    for key in ("voltage_kv", "cs_mm", "amplitude_contrast"):
        if extra.get(key) is not None:
            optics[key] = extra[key]
    set_ctf(context, ctf)
    set_optics(context.document, context.parent, **optics)
    if extra.get("defocus_handedness") is not None:
        context.parent.defocus_handedness = extra["defocus_handedness"]
    if extra.get("defocus_slope") is not None:
        context.parent.defocus_slope = extra["defocus_slope"]
