"""cryoET Data Portal runs as AreTomo3 sources: metadata through the
``cryoet_data_portal`` API client, downloads limited to the ``.aln`` (and,
on request, the mdoc, the tilt stack and point annotations).

The API supplies everything the files next to a local ``.aln`` would:
pixel spacing, image size and section count (``TiltSeries``), acquisition
order and per-image dose (``Frame``), per-section CTF and stage angles
(``PerSectionParameters``, keyed by ``z_index`` = raw section), voltage and
Cs, the tilt axis, the tomogram box (from a ``Tomogram`` at the picked voxel
spacing, or the alignment's volume), and the OME-Zarr URI for
``tomoTiltSeriesURI``. Amplitude contrast and the defocus hand are not in
the portal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import requests

from cets_nonrigid.meta.resolve import Discovered

FILES_HOST = "https://files.cryoetdataportal.cziscience.com/"
S3_BUCKET = "s3://cryoet-data-portal-public/"


def https_to_s3(url: str) -> str:
    return S3_BUCKET + url[len(FILES_HOST):] if url and url.startswith(FILES_HOST) else url


@dataclass
class PortalSection:
    z_index: int
    raw_angle: float
    acquisition_order: int  # 0-based
    exposure_dose: float
    accumulated_dose: float
    frame_name: str
    major_defocus_a: float | None = None
    minor_defocus_a: float | None = None
    astigmatic_angle_deg: float | None = None
    phase_shift_rad: float | None = None  # the portal reports radians


@dataclass
class PortalTomogram:
    id: int
    voxel_spacing: float
    size: tuple
    processing: str
    https_mrc_file: str | None = None


@dataclass
class PortalRunData:
    """Plain data pulled from the API for one run (buildable without a client)."""

    dataset_id: int
    run_id: int
    run_name: str
    https_prefix: str
    tiltseries_id: int
    pixel_spacing: float
    size: tuple  # (x, y, z) of the raw tilt series
    voltage_kv: float | None
    cs_mm: float | None
    tilt_axis_deg: float | None
    https_mrc_file: str | None
    https_omezarr_dir: str | None
    https_angle_list: str | None
    alignment_id: int | None
    alignment_type: str | None
    https_alignment_file: str | None
    alignment_volume_a: tuple | None
    mdoc_url: str | None
    sections: list[PortalSection] = field(default_factory=list)
    tomograms: list[PortalTomogram] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.run_name


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


def portal_client(url: str | None = None):
    from cryoet_data_portal import Client

    return Client(url) if url else Client()


def find_runs(client, *, dataset_id: int | None = None, run_names=(), run_ids=()) -> list:
    from cryoet_data_portal import Run

    runs = []
    for rid in run_ids or ():
        r = Run.get_by_id(client, int(rid))
        if r is None:
            raise ValueError(f"portal run id {rid} not found")
        runs.append(r)
    for name in run_names or ():
        if dataset_id is None:
            raise ValueError("--portal-run NAME needs --portal-dataset")
        hits = Run.find(client, [Run.dataset_id == int(dataset_id), Run.name == name])
        if not hits:
            raise ValueError(f"portal run {name!r} not found in dataset {dataset_id}")
        runs.extend(hits)
    if not runs and dataset_id is not None and not run_names and not run_ids:
        runs = Run.find(client, [Run.dataset_id == int(dataset_id)])
    return runs


def alignment_file_url(alignment_metadata_url: str) -> str | None:
    """The ``.aln`` listed by ``alignment_metadata.json`` (``files`` / ``alignment_path``
    are bucket keys next to the metadata)."""
    try:
        j = requests.get(alignment_metadata_url, timeout=60).json()
    except (requests.RequestException, ValueError):
        return None
    keys = [k for k in (j.get("files") or []) if str(k).lower().endswith(".aln")]
    if not keys and j.get("alignment_path"):
        keys = [j["alignment_path"]]
    if not keys:
        return None
    key = str(keys[0]).lstrip("/")
    return FILES_HOST + key


def fetch_run_data(client, run, *, alignment_id: int | None = None, prefer_local: bool = True) -> PortalRunData:
    from cryoet_data_portal import Alignment, Frame, FrameAcquisitionFile, PerSectionParameters, TiltSeries, Tomogram

    ts_list = TiltSeries.find(client, [TiltSeries.run_id == run.id])
    if not ts_list:
        raise ValueError(f"portal run {run.name} ({run.id}) has no tilt series")
    ts = ts_list[0]
    alignments = Alignment.find(client, [Alignment.run_id == run.id])
    al = None
    if alignment_id is not None:
        al = next((a for a in alignments if a.id == int(alignment_id)), None)
        if al is None:
            raise ValueError(f"alignment {alignment_id} not found in run {run.name} (has {[a.id for a in alignments]})")
    elif alignments:
        ranked = sorted(alignments, key=lambda a: (
            not bool(getattr(a, "is_portal_standard", False)),
            0 if (prefer_local and (a.alignment_type or "").upper() == "LOCAL") else 1,
            a.id,
        ))
        al = ranked[0]
    frames = Frame.find(client, [Frame.run_id == run.id])
    frame_by_id = {f.id: f for f in frames}
    psp = PerSectionParameters.find(client, [PerSectionParameters.run_id == run.id])
    sections = []
    for p in sorted(psp, key=lambda p: p.z_index):
        f = frame_by_id.get(p.frame_id)
        if f is None:
            raise ValueError(f"per-section parameters z={p.z_index} reference a frame not in run {run.name}")
        sections.append(PortalSection(
            z_index=int(p.z_index), raw_angle=float(p.raw_angle), acquisition_order=int(f.acquisition_order),
            exposure_dose=float(f.exposure_dose), accumulated_dose=float(f.accumulated_dose),
            frame_name=str(f.https_frame_path).rsplit("/", 1)[-1],
            major_defocus_a=p.major_defocus, minor_defocus_a=p.minor_defocus,
            astigmatic_angle_deg=p.astigmatic_angle, phase_shift_rad=p.phase_shift,
        ))
    mdocs = FrameAcquisitionFile.find(client, [FrameAcquisitionFile.run_id == run.id])
    tomos = [
        PortalTomogram(id=t.id, voxel_spacing=float(t.voxel_spacing), size=(int(t.size_x), int(t.size_y), int(t.size_z)),
                       processing=str(t.processing), https_mrc_file=t.https_mrc_file)
        for t in Tomogram.find(client, [Tomogram.run_id == run.id])
    ]
    vol = None
    if al is not None and al.volume_z_dimension:
        vol = (float(al.volume_x_dimension), float(al.volume_y_dimension), float(al.volume_z_dimension))
    return PortalRunData(
        dataset_id=int(run.dataset_id), run_id=int(run.id), run_name=str(run.name), https_prefix=str(run.https_prefix),
        tiltseries_id=int(ts.id), pixel_spacing=float(ts.pixel_spacing),
        size=(int(ts.size_x), int(ts.size_y), int(ts.size_z)),
        voltage_kv=float(ts.acceleration_voltage) / 1000.0 if ts.acceleration_voltage else None,
        cs_mm=float(ts.spherical_aberration_constant) if ts.spherical_aberration_constant is not None else None,
        tilt_axis_deg=float(ts.tilt_axis) if ts.tilt_axis is not None else None,
        https_mrc_file=ts.https_mrc_file, https_omezarr_dir=ts.https_omezarr_dir, https_angle_list=ts.https_angle_list,
        alignment_id=al.id if al else None, alignment_type=al.alignment_type if al else None,
        https_alignment_file=(
            (getattr(ts, "https_alignment_file", None) or alignment_file_url(al.https_alignment_metadata))
            if al is not None else None
        ),
        alignment_volume_a=vol, mdoc_url=mdocs[0].https_mdoc_path if mdocs else None,
        sections=sections, tomograms=tomos,
    )


# ---------------------------------------------------------------------------
# downloads (idempotent: same size on disk = done)
# ---------------------------------------------------------------------------


def download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, allow_redirects=True, timeout=60)
    size = int(head.headers.get("content-length", -1)) if head.ok else -1
    if dest.exists() and not overwrite and (size < 0 or dest.stat().st_size == size):
        return dest
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(1024 * 1024))
        tmp.replace(dest)
    return dest


def download_alignment(data: PortalRunData, cache_dir: Path) -> Path:
    if not data.https_alignment_file:
        raise ValueError(f"portal run {data.run_name}: no alignment file to download")
    name = data.https_alignment_file.rsplit("/", 1)[-1]
    return download(data.https_alignment_file, Path(cache_dir) / name)


def download_mdoc(data: PortalRunData, cache_dir: Path, stem: str) -> Path | None:
    if not data.mdoc_url:
        return None
    return download(data.mdoc_url, Path(cache_dir) / f"{stem}.mdoc")


def download_stack(data: PortalRunData, cache_dir: Path, stem: str) -> Path | None:
    if not data.https_mrc_file:
        return None
    return download(data.https_mrc_file, Path(cache_dir) / f"{stem}.mrc")


# ---------------------------------------------------------------------------
# metadata -> Discovered
# ---------------------------------------------------------------------------


def implied_voxel(data: PortalRunData, tomo: PortalTomogram) -> float:
    """Raw-field-implied voxel of a portal tomogram (its header voxel is rounded)."""
    return data.pixel_spacing * data.size[0] / tomo.size[0]


def pick_tomogram(data: PortalRunData, voxel_spacing: float | None = None) -> PortalTomogram | None:
    if not data.tomograms:
        return None
    if voxel_spacing is not None:
        cands = [t for t in data.tomograms if abs(t.voxel_spacing - voxel_spacing) < 1e-3]
        if not cands:
            raise ValueError(f"no portal tomogram at voxel spacing {voxel_spacing} (has {sorted({t.voxel_spacing for t in data.tomograms})})")
    else:
        cands = sorted(data.tomograms, key=lambda t: t.voxel_spacing)
    return cands[0]


def write_ctf_from_sections(data: PortalRunData, path: Path) -> Path | None:
    """AreTomo3-style ``_CTF.txt`` (7 columns, no dfHand) from the per-section
    parameters; None when the portal carries no defoci."""
    from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile, AreTomoCtfRow

    rows = []
    for s in sorted(data.sections, key=lambda s: s.z_index):
        if s.major_defocus_a is None or s.minor_defocus_a is None:
            return None
        rows.append(AreTomoCtfRow(
            micrograph=s.z_index + 1, df_max_a=float(s.major_defocus_a), df_min_a=float(s.minor_defocus_a),
            azimuth_deg=float(s.astigmatic_angle_deg or 0.0), phase_rad=float(s.phase_shift_rad or 0.0),
            score=0.0, res_a=999.99, df_hand=None,
        ))
    if not rows:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    AreTomoCtfFile(rows=rows).to_file(path)
    return path


def discover_portal_series(
    data: PortalRunData,
    aln_path: Path,
    *,
    cache_dir: Path,
    mdoc: Path | None = None,
    voxel_spacing: float | None = None,
    stack: Path | None = None,
) -> Discovered:
    """The AreTomo3 discovery over the downloaded ``.aln`` plus every value the
    API supplies (provenance ``portal``)."""
    from cets_nonrigid.meta.aretomo_run import discover_aretomo_series

    d = discover_aretomo_series(aln_path, mdoc=mdoc, tilt_stack=stack, discover_adjacent=False)
    d.series_name = data.run_name
    src = f"portal run {data.run_id}"
    d.add("pixel_size_a", data.pixel_spacing, "portal", f"{src} tiltseries.pixel_spacing")
    d.add("image_dims_px", (data.size[0], data.size[1]), "portal", f"{src} tiltseries.size")
    d.add("n_raw_sections", data.size[2], "portal", f"{src} tiltseries.size")
    if data.voltage_kv:
        d.add("voltage_kv", data.voltage_kv, "portal", f"{src} tiltseries.acceleration_voltage")
    if data.cs_mm is not None:
        d.add("cs_mm", data.cs_mm, "portal", f"{src} tiltseries.spherical_aberration_constant")
    if data.tilt_axis_deg is not None:
        d.aux["portal_tilt_axis"] = str(data.tilt_axis_deg)
    n_raw = data.size[2]
    if data.sections and len(data.sections) == n_raw and sorted(s.z_index for s in data.sections) == list(range(n_raw)):
        by_z = {s.z_index: s for s in data.sections}
        d.add("acq_order_1b", [by_z[z].acquisition_order + 1 for z in range(n_raw)], "portal", f"{src} frames.acquisition_order", note="portal")
        d.add("dose_per_section", [by_z[z].exposure_dose for z in range(n_raw)], "portal", f"{src} frames.exposure_dose", note="portal")
        d.add("stage_tilt_deg", [by_z[z].raw_angle for z in range(n_raw)], "portal", f"{src} per_section_parameters.raw_angle", note="portal")
        d.add("tilt_image_names", [Path(by_z[z].frame_name).stem for z in range(n_raw)], "portal", f"{src} frames.path")
        ctf_path = write_ctf_from_sections(data, Path(cache_dir) / f"{data.run_name}_CTF.txt")
        if ctf_path is not None:
            d.add("ctf_path", str(ctf_path), "portal", f"{src} per_section_parameters (defocus)")
    tomo = pick_tomogram(data, voxel_spacing)
    if tomo is not None:
        vs = implied_voxel(data, tomo)
        z = int(round(tomo.size[2] * vs / data.pixel_spacing / 2.0) * 2)
        d.add("tomo_dims_px", (data.size[0], data.size[1], z), "portal",
              f"{src} tomogram {tomo.id} ({tomo.voxel_spacing} A: {tomo.size}) x implied voxel {vs:.5f}")
        d.facts["portal_tomogram"] = tomo
        d.facts["portal_voxel_a"] = vs
    elif data.alignment_volume_a:
        z = int(round(data.alignment_volume_a[2] / data.pixel_spacing / 2.0) * 2)
        d.add("tomo_dims_px", (data.size[0], data.size[1], z), "portal", f"{src} alignment.volume_z_dimension/pix")
    if data.https_omezarr_dir:
        d.aux["tilt_series_uri_https"] = data.https_omezarr_dir
        d.aux["tilt_series_uri_s3"] = https_to_s3(data.https_omezarr_dir)
    d.aux["portal_run_id"] = str(data.run_id)
    d.aux["portal_dataset_id"] = str(data.dataset_id)
    return d


def download_annotation_points(client, data: PortalRunData, cache_dir: Path, *, object_name: str | None = None,
                               annotation_id: int | None = None, voxel_spacing: float | None = None) -> tuple:
    """(ndjson path, tomogram voxel spacing, implied voxel A) of one point
    annotation of this run: by ``annotation_id``, or by ``object_name``
    (case-insensitive) preferring ground-truth annotations and then the
    voxel spacing the tomogram box was taken from (``voxel_spacing``, else the
    finest); anything still ambiguous is an error listing the candidates."""
    from cryoet_data_portal import Annotation, AnnotationFile, TomogramVoxelSpacing

    all_anns = list(Annotation.find(client, [Annotation.run_id == data.run_id]))
    if annotation_id is not None:
        anns = [a for a in all_anns if int(a.id) == int(annotation_id)]
        if not anns:
            raise ValueError(f"portal run {data.run_name}: no annotation with id {annotation_id} "
                             f"(has {sorted(int(a.id) for a in all_anns)})")
    else:
        if not object_name:
            raise ValueError("portal picks need an object name or an annotation id")
        anns = [a for a in all_anns if str(a.object_name).lower() == object_name.lower()]
        if not anns:
            names = sorted({str(a.object_name) for a in all_anns})
            raise ValueError(f"portal run {data.run_name}: no annotation {object_name!r} (has {names})")
        gt = [a for a in anns if bool(a.ground_truth_status)]
        if gt:
            anns = gt
    files = []
    for a in anns:
        for f in AnnotationFile.find(client, [AnnotationFile.annotation_shape.annotation_id == a.id]):
            if f.format == "ndjson":
                tvs = TomogramVoxelSpacing.get_by_id(client, f.tomogram_voxel_spacing_id)
                files.append((a, f, float(tvs.voxel_spacing) if tvs is not None else None))
    if not files:
        raise ValueError(f"portal run {data.run_name}: annotation {object_name or annotation_id!r} has no ndjson point file")
    if len({x[1].https_path for x in files}) > 1:
        tomo = pick_tomogram(data, voxel_spacing)
        want = tomo.voxel_spacing if tomo is not None else voxel_spacing
        at_want = [x for x in files if want is not None and x[2] is not None and abs(x[2] - float(want)) < 1e-3]
        if len({x[1].https_path for x in at_want}) == 1:
            files = at_want
        else:
            cands = "; ".join(f"annotation {a.id} ({a.object_name}, ground_truth={bool(a.ground_truth_status)}, "
                              f"voxel {vs})" for a, f, vs in files)
            raise ValueError(f"portal run {data.run_name}: several point annotations match — pass "
                             f"--particles portal:<annotation-id>: {cands}")
    a, f, vs = files[0]
    name = f.https_path.rsplit("/", 1)[-1]
    path = download(f.https_path, Path(cache_dir) / f"{a.id}_{name}")
    tomo = pick_tomogram(data, vs)
    implied = implied_voxel(data, tomo) if tomo is not None else vs
    return path, vs, implied
