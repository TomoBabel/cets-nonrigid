"""In-memory sampled observations. Wire descriptors are core CETS models."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from cets_data_model.models import models as m

from cets_nonrigid.context import CetsContext


@dataclass
class SampleBlock:
    points: torch.Tensor
    projected_residual: torch.Tensor
    sample_valid: torch.Tensor
    observation_valid: torch.Tensor
    projection_valid: torch.Tensor
    weights: torch.Tensor
    displacement_3d: torch.Tensor | None = None
    displacement_valid: torch.Tensor | None = None
    ctf_depth: torch.Tensor | None = None
    ctf_depth_valid: torch.Tensor | None = None
    point_ids: list[str] | None = None

    @property
    def count(self):
        return len(self.points)

    def validate(self, context: CetsContext, channels: m.NonRigidChannels):
        n, t, dim = self.count, len(context.row_ids), context.ndim
        required = {
            "points": ((n, dim), torch.float64),
            "projected_residual": ((t, n, 2), torch.float32),
            "sample_valid": ((n,), torch.bool),
            "observation_valid": ((t, n), torch.bool),
            "projection_valid": ((t, n), torch.bool),
            "weights": ((t, n), torch.float32),
        }
        for name, (shape, dtype) in required.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values")
        if (self.weights < 0).any():
            raise ValueError("fit weights must be nonnegative")
        if (self.projection_valid & ~self.observation_valid).any():
            raise ValueError("fitting validity cannot include an unavailable observation")
        if (self.observation_valid & ~self.sample_valid[None]).any():
            raise ValueError("invalid samples cannot have available observations")
        active = torch.as_tensor(context.operators()[2], device=self.points.device)
        if self.observation_valid[~active].any() or self.projected_residual[~active].count_nonzero():
            raise ValueError("rows without global operators require unavailable, zero-filled residuals")
        if self.projected_residual[~self.observation_valid].count_nonzero():
            raise ValueError("unavailable observations must be zero-filled")
        for name, state, shape, mask_name in (
            ("displacement_3d", channels.displacement_3d, (t, n, 3), "displacement_valid"),
            ("ctf_depth", channels.ctf_depth, (t, n), "ctf_depth_valid"),
        ):
            value, mask = getattr(self, name), getattr(self, mask_name)
            if state == "none":
                if value is not None or mask is not None:
                    raise ValueError(f"{name}=none forbids channel arrays")
                continue
            if context.ndim != 3:
                raise ValueError("3D displacement and CTF-depth channels require a tilt-series alignment")
            if mask is None or tuple(mask.shape) != (t, n) or mask.dtype != torch.bool:
                raise ValueError(f"{mask_name} must be a (T,N) boolean mask")
            if (mask & ~self.sample_valid[None]).any():
                raise ValueError(f"{mask_name} cannot include invalid samples")
            if state == "zero_at_samples":
                if value is not None:
                    raise ValueError("zero_at_samples suppresses the displacement array")
            elif value is None or tuple(value.shape) != shape or value.dtype != torch.float32:
                raise ValueError(f"present {name} must have shape {shape} and dtype float32")
            elif not torch.isfinite(value).all() or value[~mask].count_nonzero():
                raise ValueError(f"{name} must be finite and zero-filled where unavailable")
        if self.point_ids is not None:
            if len(self.point_ids) != n or len(set(self.point_ids)) != n or any(not p for p in self.point_ids):
                raise ValueError("point identities must be nonempty, unique, and match the sample count")

    def observations(self, context: CetsContext):
        """Complete finite observations; unavailable positions are NaN in this runtime view."""
        q = context.evaluate_global(self.points) + self.projected_residual.to(torch.float64)
        return torch.where(self.observation_valid[..., None], q, torch.nan)


@dataclass
class DeformationSamples:
    training: SampleBlock
    heldout: SampleBlock
    sampling: m.GridSampling | m.ParticleSampling
    heldout_sampling: m.HeldoutSampling
    channels: m.NonRigidChannels
    context_fingerprint: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def validate(self, context: CetsContext):
        self.training.validate(context, self.channels)
        self.heldout.validate(context, self.channels)
        if self.heldout.count != self.heldout_sampling.count:
            raise ValueError("held-out count disagrees with the descriptor")
        if self.sampling.kind == "grid":
            expected = 1
            for count in self.sampling.grid_shape:
                expected *= count
            if expected != self.training.count:
                raise ValueError("grid dimensions disagree with sample count")
            if self.training.point_ids is not None or self.heldout.point_ids is not None:
                raise ValueError("grid samples do not carry particle identities")
        else:
            train, held = self.training.point_ids, self.heldout.point_ids
            if train is None or held is None or set(train) & set(held):
                raise ValueError("particle training and held-out samples require disjoint identities")
            annotations = {a.id: a for a in context.resolve()[0].annotations or []}
            if self.sampling.annotation_id not in annotations:
                raise ValueError("particle sampling annotation_id does not resolve in the CETS region")
            annotation = annotations[self.sampling.annotation_id]
            if set(train + held) != set(annotation.point_ids or []):
                raise ValueError("particle samples must preserve the complete bound point set")
            from cryoet_alignment.io.cets.annotations import fold_annotation_transform
            import numpy as np

            matrix, shift = fold_annotation_transform(annotation)
            coordinates = np.asarray(annotation.origin3D, dtype=np.float64) @ matrix.T + shift
            lookup = {key: coordinates[i] for i, key in enumerate(annotation.point_ids)}
            for block in (self.training, self.heldout):
                assert block.point_ids is not None  # both lists validated above
                expected = torch.as_tensor(
                    np.asarray([lookup[key] for key in block.point_ids]), dtype=torch.float64
                ).reshape(-1, 3)
                if not torch.allclose(block.points.cpu(), expected, atol=1e-9, rtol=0):
                    raise ValueError("sample positions disagree with the bound CETS point set")


@dataclass
class AlignmentBundle:
    document: m.Dataset
    deformations: dict[tuple[str, str], DeformationSamples] = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    snapshots: dict[tuple[str, str], dict[str, bytes]] = field(default_factory=dict)
    selected_alignment: tuple[str, str] | None = None

    def get_context(self, alignment_id: str | None = None, region_id: str | None = None):
        from cets_nonrigid.context import alignment_contexts

        if alignment_id is None and self.selected_alignment is not None:
            region_id, alignment_id = self.selected_alignment
        candidates = [
            c
            for c in alignment_contexts(self.document)
            if (alignment_id is None or c.alignment_id == alignment_id) and (region_id is None or c.key[0] == region_id)
        ]
        if len(candidates) != 1:
            raise ValueError(f"select one alignment explicitly; found {len(candidates)} matching alignments")
        return candidates[0]

    @property
    def context(self):
        return self.get_context()

    @property
    def samples(self):
        return self.deformations[self.context.key]

    def validate(self):
        from cets_nonrigid.context import alignment_contexts
        from cets_data_model.utils.references import validate_document_references

        validate_document_references(self.document)
        expected = set()
        for context in alignment_contexts(self.document):
            descriptor = context.owner.non_rigid_alignment
            if descriptor is None:
                continue
            context.validate()
            expected.add(context.key)
            if context.key not in self.deformations:
                raise ValueError(f"missing payload for alignment {context.key}")
            data = self.deformations[context.key]
            data.validate(context)
            if (
                data.sampling != descriptor.sampling
                or data.channels != descriptor.channels
                or data.heldout_sampling != descriptor.heldout
            ):
                raise ValueError("payload descriptors disagree with their owning alignment")
            digest = context.digest(point_ids=data.training.point_ids, heldout_point_ids=data.heldout.point_ids)
            if digest != descriptor.context_digest:
                raise ValueError(f"context digest mismatch for alignment {context.key}; create a validated new bundle")
        if set(self.deformations) != expected:
            raise ValueError("bundle contains unbound payloads")
