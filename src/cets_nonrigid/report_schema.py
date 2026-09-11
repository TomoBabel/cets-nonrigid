"""Frozen machine-readable results: ``cets_nonrigid.report/1``.

Every conversion writes ``<root>/cets_nonrigid_report.json`` (one entry per
series; re-runs into the same root replace that series' entry) plus a flat
``cets_nonrigid_report.tsv``. ``report --json`` embeds the same ``summary``
object beside the raw ``fit`` attributes, so downstream gating never needs to
scrape stdout.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cets_nonrigid import __version__
from cets_nonrigid.project.common import Gate

REPORT_SCHEMA = "cets_nonrigid.report/1"
REPORT_JSON = "cets_nonrigid_report.json"
REPORT_TSV = "cets_nonrigid_report.tsv"


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _get(obj, *path, default=None):
    cur = obj
    for name in path:
        if cur is None:
            return default
        cur = getattr(cur, name, None) if not isinstance(cur, dict) else cur.get(name)
    return default if cur is None else cur


def _heldout(fit, pix: float | None, unit: str) -> dict | None:
    if fit is None or _get(fit, "heldout_status") != "evaluated":
        return None
    if unit == "a":
        if not pix:
            return None
        return {
            "rms_px": _num(_get(fit, "rms_a_heldout")) / pix,
            "p95_px": _num(_get(fit, "p95_a_heldout")) / pix if _get(fit, "p95_a_heldout") is not None else None,
            "max_px": _num(_get(fit, "max_a_heldout")) / pix if _get(fit, "max_a_heldout") is not None else None,
            "coverage": _num(_get(fit, "coverage_heldout")),
        }
    return {
        "rms_px": _num(_get(fit, "rms_px_heldout")),
        "p95_px": _num(_get(fit, "p95_px_heldout")),
        "max_px": _num(_get(fit, "max_px_heldout")),
        "coverage": _num(_get(fit, "coverage_heldout")),
    }


def _volume_warp_summary(vw, meta) -> dict:
    """Warp-target volume warp: fitted grid + the two DISTINCT depth metrics
    (displacement-fit beam residual in the target frame; source->target
    CTF-depth deviation, only when the IR carried the source's depths)."""
    if vw is None:
        return {"fitted": False, "state": str((meta or {}).get("volume_warp", "zero"))}
    evaluated = _get(vw, "heldout_status") == "evaluated"
    dev = _num(_get(vw, "ctf_depth_deviation_rms_a_heldout"))
    return {
        "fitted": True,
        "grid": list(_get(vw, "grid") or []),
        "n_params_supported": _get(vw, "n_params_supported"),
        "data_rank": _get(vw, "data_rank"),
        "data_condition": _num(_get(vw, "data_condition")),
        "node_support": _num(_get(vw, "node_support")),
        "n_unsupported_slices": _get(vw, "n_unsupported_slices"),
        "dose_collisions": _get(vw, "dose_collisions"),
        "heldout_status": _get(vw, "heldout_status"),
        "displacement_fit_rms_a_heldout": _num(_get(vw, "rms_a_heldout")) if evaluated else None,
        "displacement_fit_beam_rms_a_heldout": _num(_get(vw, "rms_a_heldout_beam")) if evaluated else None,
        "ctf_depth_deviation_rms_a_heldout": dev if evaluated else None,
        "ctf_depth_deviation_rms_um_heldout": (dev * 1e-4) if (evaluated and dev is not None) else None,
        "ctf_depth_deviation_status": (_get(vw, "meta") or {}).get("ctf_depth_deviation"),
    }


def summary_for(direction: str, result: Any, *, pixel_size_a: float | None = None) -> dict:
    """Normalised summary (px units) for any conversion result object."""
    pix = pixel_size_a if pixel_size_a is not None else _num(_get(result, "pixel_size_a"))
    s: dict[str, Any] = {
        "direction": direction,
        "pixel_size_a": pix,
        "heldout_status": None,
        "heldout": None,
        "train": None,
        "global": None,
        "lift_max_residual_px": None,
        "defocus_hand": _get(result, "hand"),
        "alpha_offset_deg": _num(_get(result, "alpha_offset_deg")),
        "n_emitted_rows": None,
        "n_particles": _get(result, "n_particles"),
        "template_source": _get(result, "template_source"),
        "defaulted_fields": list(_get(result, "defaulted_fields") or []),
    }
    if direction == "a2w" or direction == "r2w":
        fit = _get(result, "fit")
        s["heldout_status"] = _get(fit, "heldout_status")
        s["heldout"] = _heldout(fit, pix, "a")
        train = _num(_get(fit, "rms_a_train"))
        s["train"] = {"rms_px": train / pix} if (train is not None and pix) else None
        gc = _num(_get(result, "global_check_rms_a"))
        gcp = _num(_get(result, "global_check_rms_px"))
        s["global"] = {"rms_px": (gc / pix if (gc is not None and pix) else gcp), "exact": None,
                       "used_fallback": None}
        match = _get(result, "match")
        if match is not None:
            s["n_emitted_rows"] = len(_get(match, "aln_to_warp") or [])
        vw = _get(fit, "volume_warp")
        s["volume_warp"] = _volume_warp_summary(vw, _get(fit, "meta"))
    elif direction in ("w2a", "r2a", "fit:aretomo"):
        lfit = _get(result, "local_fit")
        gfit = _get(result, "global_fit")
        s["heldout_status"] = _get(lfit, "heldout_status")
        s["heldout"] = _heldout(lfit, pix, "px")
        tr = _num(_get(lfit, "rms_px_train"))
        s["train"] = {"rms_px": tr} if tr is not None else None
        s["global"] = {
            "rms_px": _num(_get(gfit, "rms_px_heldout")),
            "train_rms_px": _num(_get(gfit, "rms_px_train")),
            "exact": None,
            "used_fallback": None,
            "validation_status": _get(result, "global_validation_status"),
        }
        check = _get(result, "aln_check")
        if check is not None:
            s["n_emitted_rows"] = _get(check, "n_rows")
    elif direction in ("a2r", "w2r"):
        g = _get(result, "global_result")
        s["heldout_status"] = "not_applicable"
        s["global"] = {
            "rms_px": _num(_get(g, "rms_px")),
            "exact": _get(g, "global_exact"),
            "used_fallback": _get(g, "used_fallback"),
        }
        s["lift_max_residual_px"] = _num(_get(result, "lift", "max_residual_px"))
        rows = _get(result, "sec_1b") or _get(result, "rows")
        s["n_emitted_rows"] = len(rows) if rows is not None else None
    return s


class ProjectReport:
    """Append-or-replace per-series entries in ``<root>/cets_nonrigid_report.json``."""

    def __init__(self, root: str | Path, direction: str, *, command_line: str | None = None):
        self.root = Path(root)
        self.direction = direction
        self.command_line = command_line if command_line is not None else " ".join(sys.argv)
        self.path = self.root / REPORT_JSON
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                if data.get("schema") == REPORT_SCHEMA:
                    data.setdefault("series", [])
                    return data
            except (OSError, ValueError):
                pass
        return {
            "schema": REPORT_SCHEMA,
            "cets_nonrigid_version": __version__,
            "output_root": str(self.root),
            "runs": [],
            "series": [],
        }

    def add_series(
        self,
        name: str,
        *,
        status: str,
        outputs: dict | None = None,
        summary: dict | None = None,
        gates: list[Gate] | None = None,
        meta: dict | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
        hint: str | None = None,
    ) -> dict:
        entry = {
            "name": name,
            "direction": self.direction,
            "status": status,
            "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "outputs": outputs or {},
            "summary": summary,
            "gates": [g.model_dump() for g in (gates or [])],
            "meta": meta,
            "warnings": list(warnings or []),
            "error": error,
            "hint": hint,
        }
        self.data["series"] = [e for e in self.data["series"] if e.get("name") != name]
        self.data["series"].append(entry)
        return entry

    def write(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.data["runs"].append(
            {"direction": self.direction, "command_line": self.command_line,
             "finished_utc": datetime.now(UTC).isoformat(timespec="seconds")}
        )
        self.data["n_ok"] = sum(1 for e in self.data["series"] if e["status"] == "ok")
        self.data["n_failed"] = sum(1 for e in self.data["series"] if e["status"] != "ok")
        self.path.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        self._write_tsv()
        return self.path

    def _write_tsv(self) -> None:
        lines = ["series\tdirection\tstatus\theldout_status\theldout_rms_px\tglobal_rms_px\tglobal_exact\tn_failed_gates\terror"]
        for e in self.data["series"]:
            s = e.get("summary") or {}
            ho = s.get("heldout") or {}
            g = s.get("global") or {}
            nfail = sum(1 for x in e.get("gates", []) if x.get("status") == "fail")
            lines.append("\t".join(str(x) if x is not None else "" for x in (
                e["name"], e["direction"], e["status"], s.get("heldout_status"),
                ho.get("rms_px"), g.get("rms_px"), g.get("exact"), nfail, e.get("error"),
            )))
        (self.root / REPORT_TSV).write_text("\n".join(lines) + "\n")
