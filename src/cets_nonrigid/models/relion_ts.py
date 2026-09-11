"""RELION 5 tomogram projection model and bound particle-set model.

Exact torch port of RELION's tilt-series projection (relion 5.0.1, checkout
210f68c8):

  P_f = s1 * s2 * Rz(zrot) * Ry(ytilt) * Rx(xtilt) * s0
        (tomogram.cpp:17-64; rotations are right-handed CCW, degrees)

with INTEGER-DIVISION centers ``s0 = T(-(w//2, h//2, d//2))`` and
``s2 = T((nx//2, ny//2, 0))`` (tomogram.cpp:44,53), while ``s1`` carries the
per-tilt shifts in tilt-series pixels (``rlnTomo*ShiftAngst / pixel_size``,
tomogram.cpp:48).  Matrices operate on pixel-to-pixel homogeneous coordinates:
input is a bin-1 tomogram voxel coordinate (corner origin), output a
tilt-image pixel coordinate (corner origin).

The public contract of the models is the canonical frame (corner-origin
Angstrom, see ``cets_nonrigid.frames``); all pixel conversions happen inside.

Particle coordinate contract (particle_set.cpp:355-379, 589-619; float centre
tomogram_set.cpp:314):

  q_eff_A = centered_coordinate_A - A_subtomogram @ origin_A   # centre-relative
  p_eff_A = q_eff_A + s * (W/2, H/2, D/2)                      # corner-origin, float centre
  legacy:  p_eff_A = s * legacy_coordinate_px - A_subtomogram @ origin_A

``A_subtomogram`` is a literal port of ``Euler::anglesToMatrix3``
(Euler_angles_relion.h:38-48; NOT the ``anglesToMatrix4`` variant, whose
(2,1) entry carries a sign bug and is not on the particle path).
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch

_F64 = torch.float64


def _rot_x(deg: torch.Tensor) -> torch.Tensor:
    a = torch.deg2rad(deg.to(_F64))
    c, s = torch.cos(a), torch.sin(a)
    m = torch.zeros(*deg.shape, 3, 3, dtype=_F64)
    m[..., 0, 0] = 1
    m[..., 1, 1] = c
    m[..., 1, 2] = -s
    m[..., 2, 1] = s
    m[..., 2, 2] = c
    return m


def _rot_y(deg: torch.Tensor) -> torch.Tensor:
    a = torch.deg2rad(deg.to(_F64))
    c, s = torch.cos(a), torch.sin(a)
    m = torch.zeros(*deg.shape, 3, 3, dtype=_F64)
    m[..., 0, 0] = c
    m[..., 0, 2] = s
    m[..., 1, 1] = 1
    m[..., 2, 0] = -s
    m[..., 2, 2] = c
    return m


def _rot_z(deg: torch.Tensor) -> torch.Tensor:
    a = torch.deg2rad(deg.to(_F64))
    c, s = torch.cos(a), torch.sin(a)
    m = torch.zeros(*deg.shape, 3, 3, dtype=_F64)
    m[..., 0, 0] = c
    m[..., 0, 1] = -s
    m[..., 1, 0] = s
    m[..., 1, 1] = c
    m[..., 2, 2] = 1
    return m


def angles_to_matrix3(rot_deg: torch.Tensor, tilt_deg: torch.Tensor, psi_deg: torch.Tensor) -> torch.Tensor:
    """Literal port of ``Euler::anglesToMatrix3`` (Euler_angles_relion.h:38-48).

    RELION reads star angles in degrees and converts to radians
    (``getAngleInRad``, particle_set.cpp:385-387); this function takes degrees.
    Returns (..., 3, 3) float64.
    """
    phi = torch.deg2rad(torch.as_tensor(rot_deg, dtype=_F64))
    theta = torch.deg2rad(torch.as_tensor(tilt_deg, dtype=_F64))
    chi = torch.deg2rad(torch.as_tensor(psi_deg, dtype=_F64))
    sp, cp = torch.sin(phi), torch.cos(phi)
    st, ct = torch.sin(theta), torch.cos(theta)
    sc, cc = torch.sin(chi), torch.cos(chi)
    m = torch.empty(*phi.shape, 3, 3, dtype=_F64)
    m[..., 0, 0] = cc * ct * cp - sc * sp
    m[..., 0, 1] = cc * ct * sp + sc * cp
    m[..., 0, 2] = -cc * st
    m[..., 1, 0] = -sc * ct * cp - cc * sp
    m[..., 1, 1] = -sc * ct * sp + cc * cp
    m[..., 1, 2] = sc * st
    m[..., 2, 0] = st * cp
    m[..., 2, 1] = st * sp
    m[..., 2, 2] = ct
    return m


def effective_positions_a(
    *,
    pixel_size_a: float,
    tomo_dims_px: tuple[int, int, int],
    centered_coords_a: torch.Tensor | None = None,  # (P, 3) rlnCenteredCoordinate*Angst
    legacy_coords_px: torch.Tensor | None = None,  # (P, 3) rlnCoordinateX/Y/Z (decentered px)
    origins_a: torch.Tensor | None = None,  # (P, 3) rlnOriginX/Y/ZAngst
    subtomo_angles_deg: torch.Tensor | None = None,  # (P, 3) rlnTomoSubtomogramRot/Tilt/Psi
) -> torch.Tensor:
    """Effective particle positions in canonical corner-origin Angstrom.

    Implements RELION's read path exactly: the origin correction
    ``- A_subtomogram @ origin`` applies to BOTH coordinate representations
    (getPosition subtracts it after either branch, particle_set.cpp:355-379);
    centered columns take precedence when both are given (checked first,
    particle_set.cpp:593-605); the corner-origin conversion uses the FLOAT
    tomogram centre (tomogram_set.cpp:314).
    """
    s = float(pixel_size_a)
    if centered_coords_a is not None:
        q = centered_coords_a.to(_F64).clone()
        centre_a = torch.tensor([d / 2.0 for d in tomo_dims_px], dtype=_F64) * s
        p_eff = q + centre_a
    elif legacy_coords_px is not None:
        p_eff = legacy_coords_px.to(_F64) * s
    else:
        raise ValueError("need centered_coords_a or legacy_coords_px")

    if origins_a is not None:
        origins = origins_a.to(_F64)
        if subtomo_angles_deg is not None:
            a = subtomo_angles_deg.to(_F64)
            m = angles_to_matrix3(a[:, 0], a[:, 1], a[:, 2])  # (P, 3, 3)
            rotated = torch.einsum("pij,pj->pi", m, origins)
        else:
            rotated = origins
        p_eff = p_eff - rotated
    return p_eff


def _check_hand(hand) -> int:
    h = int(hand)
    if h not in (-1, 1):
        raise ValueError(f"rlnTomoHand must be +1 or -1, got {hand!r}")
    return h


class RelionTomogramModel:
    """RELION 5 rigid tilt-series projection (implements ``TiltProjectionModel``).

    ``project_volume`` and ``project_volume_global`` are identical for the bare
    model (no local deformation lives here).
    """

    def __init__(
        self,
        *,
        xtilt_deg: torch.Tensor,  # (T,)
        ytilt_deg: torch.Tensor,  # (T,)
        zrot_deg: torch.Tensor,  # (T,)
        xshift_a: torch.Tensor,  # (T,)
        yshift_a: torch.Tensor,  # (T,)
        tomo_dims_px: tuple[int, int, int],
        image_dims_px: tuple[int, int],
        pixel_size_a: float,
        hand: int = 1,
        defocus_slope: float = 1.0,
    ) -> None:
        if pixel_size_a <= 1e-3:
            raise ValueError(f"pixel size {pixel_size_a} is not positive")  # tomogram.cpp:38
        t = torch.as_tensor(ytilt_deg, dtype=_F64).shape[0]
        xt = torch.as_tensor(xtilt_deg, dtype=_F64)
        yt = torch.as_tensor(ytilt_deg, dtype=_F64)
        zr = torch.as_tensor(zrot_deg, dtype=_F64)
        sx = torch.as_tensor(xshift_a, dtype=_F64)
        sy = torch.as_tensor(yshift_a, dtype=_F64)
        for name, v in [("xtilt", xt), ("ytilt", yt), ("zrot", zr), ("xshift", sx), ("yshift", sy)]:
            if v.shape != (t,):
                raise ValueError(f"{name} has shape {tuple(v.shape)}, expected ({t},)")

        self.pixel_size_a = float(pixel_size_a)
        self.tomo_dims_px = tuple(int(d) for d in tomo_dims_px)
        self.image_dims_px = tuple(int(d) for d in image_dims_px)
        self.hand = _check_hand(hand)
        self.defocus_slope = float(defocus_slope)

        # tomogram.cpp:41-62 — integer-division centers, shifts in px.
        r = _rot_z(zr) @ _rot_y(yt) @ _rot_x(xt)  # (T, 3, 3)
        c_int = torch.tensor([d // 2 for d in self.tomo_dims_px], dtype=_F64)
        i_int = torch.tensor([self.image_dims_px[0] // 2, self.image_dims_px[1] // 2, 0.0], dtype=_F64)
        shift_px = torch.stack([sx, sy, torch.zeros(t, dtype=_F64)], dim=-1) / self.pixel_size_a
        trans = shift_px + i_int - torch.einsum("tij,j->ti", r, c_int)

        p = torch.zeros(t, 4, 4, dtype=_F64)
        p[:, :3, :3] = r
        p[:, :3, 3] = trans
        p[:, 3, 3] = 1.0
        self._p = p
        self._angles = (xt, yt, zr, sx, sy)

    @classmethod
    def from_matrices(
        cls,
        matrices: torch.Tensor,  # (T, 4, 4) pixel-to-pixel homogeneous
        *,
        pixel_size_a: float,
        tomo_dims_px: tuple[int, int, int],
        image_dims_px: tuple[int, int],
        hand: int = 1,
        defocus_slope: float = 1.0,
    ) -> RelionTomogramModel:
        """Construct directly from rlnTomoProjX/Y/Z/W-style matrices.

        Malformed/non-rigid matrices are rejected: the trajectory lift relies
        on the linear part being a pure rotation (pseudo-inverse = transpose).
        """
        m = torch.as_tensor(matrices, dtype=_F64)
        if m.ndim != 3 or m.shape[1:] != (4, 4):
            raise ValueError(f"matrices have shape {tuple(m.shape)}, expected (T, 4, 4)")
        if not torch.isfinite(m).all():
            raise ValueError("projection matrices contain non-finite values")
        last = m[:, 3, :]
        expect = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=_F64)
        if not torch.allclose(last, expect.expand_as(last), atol=1e-9):
            raise ValueError("projection matrices are not homogeneous (last row != [0,0,0,1])")
        r = m[:, :3, :3]
        ortho_err = (r @ r.transpose(1, 2) - torch.eye(3, dtype=_F64)).abs().max()
        if ortho_err > 1e-6:
            raise ValueError(f"linear part is not orthonormal (max |R R^T - I| = {ortho_err:.3e})")
        det = torch.linalg.det(r)
        if not torch.all((det - 1.0).abs() < 1e-6):
            raise ValueError(f"linear part is not a proper rotation (det in [{det.min():.6f}, {det.max():.6f}])")

        self = cls.__new__(cls)
        self.pixel_size_a = float(pixel_size_a)
        self.tomo_dims_px = tuple(int(d) for d in tomo_dims_px)
        self.image_dims_px = tuple(int(d) for d in image_dims_px)
        self.hand = _check_hand(hand)
        self.defocus_slope = float(defocus_slope)
        self._p = m.clone()
        self._angles = None
        return self

    def with_ctf_convention(self, *, hand: int | None = None, defocus_slope: float | None = None) -> RelionTomogramModel:
        """Copy with the CTF-depth convention fields replaced (geometry shared)."""
        out = copy.copy(self)
        if hand is not None:
            out.hand = _check_hand(hand)
        if defocus_slope is not None:
            out.defocus_slope = float(defocus_slope)
        return out

    @property
    def n_projections(self) -> int:
        return self._p.shape[0]

    @property
    def projection_matrices(self) -> torch.Tensor:
        """(T, 4, 4) float64, pixel-to-pixel homogeneous."""
        return self._p

    @property
    def rotations(self) -> torch.Tensor:
        """(T, 3, 3) float64 pure rotations (the linear part of P)."""
        return self._p[:, :3, :3]

    @property
    def image_dims_a(self) -> torch.Tensor:
        return torch.tensor(self.image_dims_px, dtype=_F64) * self.pixel_size_a

    @property
    def volume_dims_a(self) -> torch.Tensor:
        return torch.tensor(self.tomo_dims_px, dtype=_F64) * self.pixel_size_a

    def _project_px(self, points_px: torch.Tensor) -> torch.Tensor:
        """(N, 3) tomogram voxel px -> (T, N, 2) tilt-image px."""
        r = self._p[:, :3, :3]
        t = self._p[:, :3, 3]
        out = torch.einsum("tij,nj->tni", r, points_px.to(_F64)) + t[:, None, :]
        return out[..., :2]

    def _finish(self, xy_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.image_dims_a.to(xy_a)
        valid = (
            (xy_a[..., 0] >= 0)
            & (xy_a[..., 0] <= img[0])
            & (xy_a[..., 1] >= 0)
            & (xy_a[..., 1] <= img[1])
            & torch.isfinite(xy_a).all(dim=-1)
        )
        return xy_a, valid

    def project_volume(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xy_px = self._project_px(points_3d.to(_F64) / self.pixel_size_a)
        return self._finish(xy_px * self.pixel_size_a)

    def project_volume_global(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.project_volume(points_3d)

    def ctf_depth(self, points_3d: torch.Tensor, displacement: torch.Tensor | None = None) -> torch.Tensor:
        """Signed defocus contribution (T, N) in Angstrom, RELION convention.

        ``Tomogram::getCtf`` (tomogram.cpp:277-285) adds
        ``dz = handedness * pixelSize * defocusSlope * getDepthOffset(f, pos)``
        with ``getDepthOffset = (P_f pos).z - (P_f centre).z`` in pixels
        (tomogram.cpp:267-274; ``centre`` is the FLOAT tomogram centre,
        tomogram_set.cpp:314). ``pos`` is the STATIC particle coordinate at every
        consumer (subtomo.cpp:780-783, ctf_refinement.cpp:591,
        reconstruct_particle.cpp:374, prediction.cpp:200/389,
        local_particle_refinement.cpp:81) — trajectories never enter, so
        ``displacement`` is ignored and only accepted for protocol symmetry.
        """
        del displacement  # RELION's CTF depth is that of the static coordinate
        pts_px = points_3d.to(_F64) / self.pixel_size_a  # (N, 3) bin-1 voxel px, corner origin
        centre_px = torch.tensor([d / 2.0 for d in self.tomo_dims_px], dtype=_F64)
        r = self._p[:, :3, :3]
        depth_px = torch.einsum("tj,nj->tn", r[:, 2, :], pts_px - centre_px)  # (T, N)
        return self.hand * self.pixel_size_a * self.defocus_slope * depth_px


class RelionParticleSetModel:
    """RELION global model bound to a particle set (implements ``TiltProjectionModel``).

    Protocol semantics (pinned):
      * ``project_volume``       = effective position + trajectory, rigid
        projection, then 2D deformation;
      * ``project_volume_global`` = effective position + rigid projection only
        (no trajectory, no deformation).

    The model is only defined on its bound particle set: ``project_volume``
    must be called with exactly the bound positions (the scattered-point IR
    builder projects the full set once and splits afterwards).
    """

    def __init__(
        self,
        global_model: RelionTomogramModel,
        positions_eff_a: torch.Tensor,  # (P, 3) canonical corner-origin Angstrom
        trajectories_a: torch.Tensor | None = None,  # (T, P, 3) additive Angstrom offsets
        visible: torch.Tensor | None = None,  # (T, P) bool
        deformation: Callable[[torch.Tensor], torch.Tensor] | None = None,  # px -> px, per tilt
    ) -> None:
        self.global_model = global_model
        self.positions_eff_a = positions_eff_a.to(_F64)
        t = global_model.n_projections
        p = self.positions_eff_a.shape[0]
        if trajectories_a is not None:
            trajectories_a = trajectories_a.to(_F64)
            if trajectories_a.shape != (t, p, 3):
                raise ValueError(
                    f"trajectories have shape {tuple(trajectories_a.shape)}, expected ({t}, {p}, 3)"
                )
            if not torch.isfinite(trajectories_a).all():
                raise ValueError("trajectories contain non-finite values")
        self.trajectories_a = trajectories_a
        if visible is not None and visible.shape != (t, p):
            raise ValueError(f"visible has shape {tuple(visible.shape)}, expected ({t}, {p})")
        self.visible = visible
        self.deformation = deformation

    @property
    def n_projections(self) -> int:
        return self.global_model.n_projections

    def _check_bound(self, points_3d: torch.Tensor) -> None:
        if points_3d.shape != self.positions_eff_a.shape or not torch.allclose(
            points_3d.to(_F64), self.positions_eff_a, atol=1e-6
        ):
            raise ValueError(
                "RelionParticleSetModel is only defined on its bound particle set; "
                "project the full set and subset afterwards"
            )

    def project_volume(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_bound(points_3d)
        s = self.global_model.pixel_size_a
        pos = self.positions_eff_a[None, :, :]  # (1, P, 3)
        if self.trajectories_a is not None:
            pos = pos + self.trajectories_a  # (T, P, 3)
        else:
            pos = pos.expand(self.n_projections, -1, -1)
        r = self.global_model.projection_matrices[:, :3, :3]
        t = self.global_model.projection_matrices[:, :3, 3]
        xy_px = torch.einsum("tij,tpj->tpi", r, pos / s) + t[:, None, :]
        xy_px = xy_px[..., :2]
        if self.deformation is not None:
            xy_px = self.deformation(xy_px)
        xy_a, valid = self.global_model._finish(xy_px * s)
        if self.visible is not None:
            valid = valid & self.visible
        return xy_a, valid

    def project_volume_global(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_bound(points_3d)
        xy_a, valid = self.global_model.project_volume(self.positions_eff_a)
        if self.visible is not None:
            valid = valid & self.visible
        return xy_a, valid

    def displace_volume(self, points_3d: torch.Tensor) -> torch.Tensor | None:
        """Per-particle per-frame 3D trajectory (T, P, 3) in Angstrom — the
        additive offset RELION applies before projection (``motion.star``,
        ``trajectory.cpp:172`` ``origin + shifts_Ang[f]``). ``None`` when the
        bound set carries no trajectories (no 3D deformation model)."""
        self._check_bound(points_3d)
        if self.trajectories_a is None:
            return None
        return self.trajectories_a.clone()

    def ctf_depth(self, points_3d: torch.Tensor, displacement: torch.Tensor | None = None) -> torch.Tensor:
        """RELION CTF depth of the STATIC bound positions (trajectories never
        enter RELION's ``getCtf``); see :meth:`RelionTomogramModel.ctf_depth`."""
        self._check_bound(points_3d)
        return self.global_model.ctf_depth(self.positions_eff_a, displacement)
