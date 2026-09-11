"""Shared CLI plumbing for the batch-capable conversions: source expansion
(local files, directories, portal runs), output-root vs legacy single-file
targets, AreTomo3-source metadata resolution (discovery + CLI precedence),
cloup option groups and per-series reporting."""

from __future__ import annotations

from pathlib import Path

import click
import cloup

from cets_nonrigid.cli.specs import PortalSource, expand_sources
from cets_nonrigid.meta import MetaConflictError, SeriesMeta, resolve_series
from cets_nonrigid.meta.aretomo_run import discover_aretomo_series
from cets_nonrigid.meta.model import FIELD_FLAGS

__all__ = [
    "PortalSeries", "SeriesRunner", "batch_options", "echo_meta", "expand_sources", "output_target",
    "require_meta", "resolve_aretomo_source", "series_options", "series_sources", "single_only",
    "store_target", "warp_dims_override", "warp_xml_ctf_params", "warp_xml_pixel_size",
]


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def output_target(out: Path, stem: str, ext: str, *, n_sources: int, subdir: str | None = None):
    """``-o x<ext>`` (single source) writes that file; anything else is a
    project root and the series file goes to ``root/[subdir/]<stem><ext>``."""
    if out.suffix.lower() == ext:
        if n_sources != 1:
            raise click.UsageError(
                f"-o {out} names a single file but {n_sources} sources were given; "
                "pass an output directory instead"
            )
        return None, out
    target = (out / subdir if subdir else out) / f"{stem}{ext}"
    return out, target


def single_only(name: str, value, n_sources: int) -> None:
    if value is not None and value is not False and n_sources != 1:
        raise click.UsageError(f"{name} applies to a single source; {n_sources} were given")


def store_target(store: bool, beside: Path, stem: str) -> Path | None:
    """``--store`` writes ``<stem>.cets_nonrigid.zarr`` beside the output file."""
    return beside.with_name(f"{stem}.cets_nonrigid.zarr") if store else None


# ---------------------------------------------------------------------------
# option groups
# ---------------------------------------------------------------------------


def _dose_per_tilt_callback(ctx, param, value):
    if value is None:
        return None
    if str(value).strip().lower() == "file":
        return "file"
    try:
        return float(value)
    except ValueError:
        raise click.BadParameter("a dose in e/A^2 or the word 'file'") from None


def batch_options():
    return cloup.option_group(
        "Batch",
        cloup.option("--overwrite", is_flag=True, help="Replace a series that already exists in the output."),
        cloup.option("--fail-fast", is_flag=True, help="Abort on the first failed series."),
    )


def series_options():
    """Metadata of an AreTomo3-sourced series: every value defaults to what is
    discovered next to the .aln (stack header, _TLT.txt, mdoc, _CTF.txt,
    _Vol.mrc) or comes from the portal; a CLI value always wins."""
    return cloup.option_group(
        "Series metadata",
        "Discovered next to each .aln (<stem>.mrc header, _TLT.txt, mdoc, _CTF.txt, _Vol.mrc) or from the "
        "portal API; a value given here wins and its provenance is printed.",
        cloup.option("--pix", "pixel_size_a", type=float, default=None,
                     help="Pixel size (A/px) [default: tilt-stack header, else mdoc PixelSpacing]."),
        cloup.option("--tomo-size", "tomo_size", default=None,
                     help="Bin-1 tomogram dims XxYxZ, or just Z (X,Y from RawSize) [default: <stem>_Vol.mrc header x bin]."),
        cloup.option("--voltage", "voltage_kv", type=float, default=None, help="kV [default: mdoc Voltage]."),
        cloup.option("--cs", "cs_mm", type=float, default=None, help="Spherical aberration (mm)."),
        cloup.option("--amp-contrast", "amplitude_contrast", type=float, default=None,
                     help="Amplitude contrast (no file carries it)."),
        cloup.option("--mdoc-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
                     help="Directory holding <stem>.mdoc files [default: next to the .aln]."),
        cloup.option("--tilt-stack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
                     help="Directory holding <stem>.mrc stacks (raw sorted order incl. darks) [default: next to the .aln]."),
        cloup.option("--dose-per-tilt", default=None, metavar="D|file", callback=_dose_per_tilt_callback,
                     help="Per-tilt dose (e/A^2) [default: the constant per-image dose of _TLT.txt/mdoc/portal]; "
                          "'file' uses the per-image doses of that file even when they vary."),
        cloup.option("--dose-convention", type=click.Choice(["exclusive", "inclusive"]), default="exclusive",
                     show_default=True, help="Pre-exposure = dose BEFORE the image (exclusive) or including it."),
        cloup.option("--no-ctf", "no_ctf", is_flag=True, help="Ignore any _CTF.txt / portal CTF."),
    )


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


class PortalSeries:
    def __init__(self, spec: PortalSource, data, aln_path: Path, cache_dir: Path, *, stack_path=None, client=None):
        self.spec = spec
        self.data = data
        self.aln_path = aln_path
        self.cache_dir = cache_dir
        self.stack_path = stack_path
        self.client = client

    @property
    def stem(self) -> str:
        return self.data.run_name


def portal_series(specs: list, cache_root: Path, *, need_stack: bool) -> list:
    """Resolve ``portal:`` sources into downloaded .aln files + API metadata.
    The tilt stack is downloaded only when the target needs pixels."""
    if not specs:
        return []
    from cets_nonrigid.meta.portal import download_alignment, download_stack, fetch_run_data, find_runs, portal_client

    client = portal_client()
    out = []
    for spec in specs:
        runs = find_runs(client, dataset_id=spec.dataset_id, run_names=(spec.run_name,) if spec.run_name else ())
        for run in runs:
            data = fetch_run_data(client, run, alignment_id=spec.alignment_id)
            cache = Path(cache_root) / data.run_name
            click.echo(f"portal: run {data.run_name} (id {data.run_id}, dataset {data.dataset_id}), "
                       f"alignment {data.alignment_id} ({data.alignment_type}), {len(data.sections)} sections")
            aln = download_alignment(data, cache)
            stack = download_stack(data, cache, data.run_name) if need_stack else None
            out.append(PortalSeries(spec, data, aln, cache, stack_path=stack, client=client))
    return out


def series_sources(tokens, cache_root: Path, *, need_stack: bool) -> list:
    """[(stem, aln_path, PortalSeries|None)] from SOURCE tokens (files,
    directories, ``portal:`` specs)."""
    items = []
    portal_specs = []
    for src in expand_sources(tokens, "aln", portal=True):
        if isinstance(src, PortalSource):
            portal_specs.append(src)
        else:
            items.append((src.stem, src, None))
    for ps in portal_series(portal_specs, cache_root, need_stack=need_stack):
        items.append((ps.stem, ps.aln_path, ps))
    return items


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def parse_tomo_size(text: str | None, image_dims: tuple | None) -> tuple[int, int, int] | None:
    if text is None:
        return None
    parts = [int(v) for v in str(text).lower().split("x")]
    if len(parts) == 3:
        return tuple(parts)
    if len(parts) == 1:
        if image_dims is None:
            raise click.UsageError("--tomo-size Z needs the image dims (RawSize) — give XxYxZ")
        return (int(image_dims[0]), int(image_dims[1]), parts[0])
    raise click.UsageError("--tomo-size must be XxYxZ or Z")


def resolve_aretomo_source(aln: Path, p: dict, *, defocus_hand=None, portal=None) -> SeriesMeta:
    """Discovery + CLI precedence for one .aln (local run dir, or a portal run
    whose metadata comes from the API); raises UsageError on conflicts."""
    if portal is not None:
        from cets_nonrigid.meta.portal import discover_portal_series

        disc = discover_portal_series(
            portal.data, aln, cache_dir=portal.cache_dir, voxel_spacing=portal.spec.voxel_spacing,
            stack=portal.stack_path,
        )
    else:
        disc = discover_aretomo_series(
            aln, mdoc_dir=p.get("mdoc_dir"), tilt_stack_dir=p.get("tilt_stack_dir"), no_ctf=bool(p.get("no_ctf")),
        )
    dose, dose_from = p.get("dose_per_tilt"), None
    if dose == "file":
        dose, dose_from = None, "file"
    cli = {
        "pixel_size_a": p.get("pixel_size_a"),
        "tomo_dims_px": parse_tomo_size(p.get("tomo_size"), disc.first("image_dims_px")),
        "voltage_kv": p.get("voltage_kv"),
        "cs_mm": p.get("cs_mm"),
        "amplitude_contrast": p.get("amplitude_contrast"),
        "defocus_hand": defocus_hand,
        "dose_per_tilt": dose,
    }
    try:
        meta = resolve_series(disc, cli, dose_from=dose_from, dose_convention=p.get("dose_convention", "exclusive"))
    except MetaConflictError as e:
        raise click.UsageError(str(e)) from e
    if p.get("no_ctf"):
        meta.ctf_path = None
        meta.ctf_source = "none"
    elif meta.ctf_path is not None:
        meta.ctf_source = "aretomo_ctf_txt"
    if meta.dose_per_tilt is not None and meta.raw_dose is None and meta.acq_order_1b is None:
        meta.warnings.append("dose: no acquisition order found (_TLT.txt / mdoc / portal) — dose not applied")
    return meta


def require_meta(meta: SeriesMeta, *fields: str) -> None:
    missing = [f for f in fields if getattr(meta, f) is None]
    if missing:
        flags = ", ".join(FIELD_FLAGS.get(f, f) for f in missing)
        raise click.UsageError(
            f"{meta.series_name}: could not determine {', '.join(missing)} — pass {flags}"
        )


def echo_meta(meta: SeriesMeta) -> None:
    click.echo(meta.summary_line())
    for w in meta.warnings:
        click.secho(f"  WARNING: {w}", fg="yellow")


def warp_xml_pixel_size(xml_path: Path) -> float | None:
    """``<CTF><Param Name="PixelSize" Value="..."/></CTF>`` of a Warp tilt-series XML."""
    from lxml import etree

    root = etree.fromstring(Path(xml_path).read_bytes())
    ctf = root.find("CTF")
    if ctf is None:
        return None
    for param in ctf.findall("Param"):
        if param.get("Name") == "PixelSize":
            try:
                v = float(param.get("Value"))
            except (TypeError, ValueError):
                return None
            return v if v > 0 else None
    return None


def warp_xml_ctf_params(xml_path: Path) -> dict:
    """``<CTF>`` scalar params (PixelSize, Voltage, Cs, Amplitude) of a Warp XML as floats."""
    from lxml import etree

    root = etree.fromstring(Path(xml_path).read_bytes())
    ctf = root.find("CTF")
    out: dict[str, float] = {}
    if ctf is None:
        return out
    for param in ctf.findall("Param"):
        name = param.get("Name")
        if name in ("PixelSize", "Voltage", "Cs", "Amplitude"):
            try:
                out[name] = float(param.get("Value"))
            except (TypeError, ValueError):
                continue
    return out


class SeriesRunner:
    """Collects per-series outcomes; exits non-zero when any failed."""

    def __init__(self, report, *, fail_fast: bool):
        self.report = report
        self.fail_fast = fail_fast
        self.ok: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def failure(self, name: str, exc: BaseException, *, meta: SeriesMeta | None = None) -> None:
        self.failed.append((name, str(exc)))
        click.secho(f"{name}: FAILED - {exc}", fg="red")
        if self.report is not None:
            self.report.add_series(name, status="failed", error=str(exc),
                                   meta=meta.report() if meta else None,
                                   warnings=meta.warnings if meta else None)
        if self.fail_fast:
            self.finish()
            raise SystemExit(1)

    def success(self, name: str) -> None:
        self.ok.append(name)

    def finish(self) -> None:
        if self.report is not None:
            self.report.write()
            click.echo(f"report: {self.report.path}")
        total = len(self.ok) + len(self.failed)
        if total > 1 or self.failed:
            click.echo(f"{len(self.ok)} converted, {len(self.failed)} failed of {total}")
        if self.failed:
            raise SystemExit(1)


def warp_dims_override(xml: Path, meta, image_px, volume_px):
    """Load-time dims for a Warp XML: from ``--image-px``/``--volume-px`` when
    given; otherwise, when the XML's root dimension attributes are zero (Warp
    re-saves them that way), from the discovered settings/average-header dims.
    ``None`` when the XML carries its own dimensions. Nothing on disk is edited."""
    from lxml import etree

    from cets_nonrigid.io.warp_xml import DimsOverride

    pix = float(meta.pixel_size_a)
    if image_px or volume_px:
        return DimsOverride(
            image_a=tuple(v * pix for v in image_px) if image_px else None,
            volume_a=tuple(v * pix for v in volume_px) if volume_px else None,
        )
    root = etree.fromstring(Path(xml).read_bytes())

    def _zero(attr: str) -> bool:
        raw = (root.get(attr) or "").strip()
        if not raw:
            return True
        try:
            return all(float(v) <= 0 for v in raw.split(","))
        except ValueError:
            return True

    img_zero, vol_zero = _zero("ImageDimensionsAngstrom"), _zero("VolumeDimensionsAngstrom")
    image_a = tuple(v * pix for v in meta.image_dims_px) if img_zero and meta.image_dims_px else None
    volume_a = tuple(v * pix for v in meta.tomo_dims_px) if vol_zero and meta.tomo_dims_px else None
    if image_a is None and volume_a is None:
        return None
    return DimsOverride(image_a=image_a, volume_a=volume_a)
