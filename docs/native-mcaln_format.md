# `.mcaln` — AreTomo3 2D motion alignment format (v1 — FROZEN 2026-08-28)

Status: **FROZEN v1**. The motion-map semantics were pinned by (a) verbatim
source reading of the full correction chain (`CPatchAlign::DoIt`:
`CGenRealStack` corrects the `frm` buffer in place with the
FmRef-relative global via Fourier phase shift; patches are measured on those
corrected frames; `GCorrectPatchShift` applies ONLY the residual local field
in real space) and (b) the executable kernel contract test
(`tools/mckernel` + `tests/test_m6_kernel_contract.py`): the VERBATIM
`mGCorrect3D` kernel, extracted from the AreTomo3 sources at build time and
run on the GPU with index-encoded frames, samples exactly the source pixels
the Python model predicts — including truncation NN, the r <= 0.5 cutoff,
bad-patch skipping, x2 upsampling, and the out-of-range scramble fill.

Pinned semantics:
- Total map: ``raw(x, f) = x - S_local(x, f) - glob_f`` — corrections
  SUBTRACTED, local field evaluated at the corrected/output coordinate.
- Coordinates are original-resolution pixel INDICES, corner origin, no
  half-pixel offset.
- ``globalShift`` values are the FmRef-relative full shifts
  (``glob[fmRef] = 0``); ``localShift`` values are the post-``MakeRelative``
  residual patch shifts, exactly what `CPatchShifts` holds at correction
  time. `-InMotion` bypasses both `MakeRelative` calls (file shifts final).
- The full `-InMotion` binary round-trip remains the post-freeze integration
  test.

## Syntax (tag-based text)

Magic first line:

```
# AreTomo3 MotionAlign 1.0
```

### `setting` block

```
setting
   raw_frame_count: <int>          # frames in the input movie file
   integrated_frame_count: <int>   # pre-throw integrated frames
   aligned_frame_count: <int>      # integrated frames retained for alignment
   alignment_image_size_px: <Nx> <Ny>
   alignment_pixel_size_A: <float>
   input_to_alignment_scale_xy: <sx> <sy>
   patches: <nx> <ny>
   fmRef: <int>                    # reference frame, ALIGNED-frame index space
   frame_index_base: 0
```

Notes:
- Shifts and patch centers are defined in `alignment_image_size_px`
  coordinates. They are only ever *called* "raw movie pixels" once an
  executable test proves raw-coordinate identity for the MRC, TIFF, and EER
  input paths.
- No unconditional `binning:` tag; `input_to_alignment_scale_xy` carries any
  input-to-alignment scaling (e.g. EER rendering).

### `frameTable` block

One row per integrated frame (pre-throw):

```
frameTable
   <integrated_index> <source_start> <source_count> <included> <aligned_index>
```

- Covers `-Throw`, EER grouping/integration, and preprocessing.
- Excluded rows carry `included=0` and `aligned_index=-1`.
- Requires the AreTomo3 `CFmIntParam` provenance-retention patch: raw-frame
  count, pre-throw integrated count, original source-start/count entries,
  inclusion flags, and the aligned-index mapping are retained immutably
  (`mRemoveFrames()` otherwise compacts them in place).

### `globalShift` block

```
globalShift
   <f> <sx> <sy>
```

### `localShift` blocks (one per patch)

```
localShift
   patchID: <p>
   <f> <cx> <cy> <sx> <sy> <valid>
```

- `valid` is the per-(patch,frame) flag; MotionCor's correction kernel skips
  bad patches, so it must round-trip.
- `f` in both shift blocks is `aligned_index` (as is `fmRef`).

## Validation (both readers)

- magic/version line present and supported;
- all values finite;
- counts consistent with the frame table and `patches`;
- no duplicate `(f)`, `(p, f)` indices;
- `fmRef` and every `f` a valid aligned index.
