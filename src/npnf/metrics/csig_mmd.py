"""Censored Signature MMD (CSig-MMD) estimators and censoring parameters.

Per-condition divergence in the signature kernel RKHS that reweights samples
toward a zero-path pivot based on Mahalanobis distance. See Redhead et al.
(https://arxiv.org/abs/2602.10182).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from npnf.metrics.mahalanobis_artifact import (
    SignatureMahalanobisArtifact,
    censoring_threshold,
    load_signature_mahalanobis_artifact,
)
from npnf.metrics.sig_mmd import kernel_grams, mmd_squared
from npnf.metrics.signature import (
    compute_normalized_truncated_sigs,
    cosine_normalized,
    mahalanobis_distance,
    signature_kernel_gram,
    to_kernel_device,
    to_metric_paths,
)


@dataclass(frozen=True)
class CensoringParameters:
    """The CSig-MMD censoring rule before it meets a censoring reference.

    ``resolve`` turns it into a ``Censoring`` (alpha, beta, c²). By default the
    threshold ``c²`` keeps the fraction ``lodged_recall`` of the lodged reference
    draws in the tail, and ``beta = beta_times_c_squared / c²``, so the weight rises
    from 0.12 to 0.88 within about 8 % of ``c²`` for every threshold. ``alpha`` (a
    quantile of the reference distances) and ``beta`` override the rule.
    """

    alpha: float | None = None
    beta: float | None = None
    lodged_recall: float = 0.975
    beta_times_c_squared: float = 25.6

    def __post_init__(self) -> None:
        for name in ("alpha", "lodged_recall"):
            value = getattr(self, name)
            if value is not None and not (0 < value < 1):
                msg = f"{name} must be in (0, 1); got {value}"
                raise ValueError(msg)
        for name in ("beta", "beta_times_c_squared"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                msg = f"{name} must be > 0; got {value}"
                raise ValueError(msg)

    def resolve(self, artifact: SignatureMahalanobisArtifact) -> Censoring:
        alpha, c_squared = censoring_threshold(artifact, self.alpha, self.lodged_recall)
        beta = self.beta_times_c_squared / c_squared if self.beta is None else self.beta
        return Censoring(alpha, beta, c_squared)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--alpha",
            type=float,
            default=None,
            help=(
                "Quantile of the censoring reference Mahalanobis distances for the "
                "CSig-MMD threshold c² (default: the threshold that keeps 97.5 %% of "
                "the lodged reference draws in the tail)."
            ),
        )
        parser.add_argument(
            "--beta",
            type=float,
            default=None,
            help=(
                "Logistic sharpness for CSig-MMD soft censoring weight "
                f"(default: {cls().beta_times_c_squared} / c²)."
            ),
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> CensoringParameters:
        return cls(alpha=args.alpha, beta=args.beta)


def build_condition_index(
    condition_gids: np.ndarray, condition_yss: np.ndarray
) -> dict[tuple[str, str], int]:
    return {
        (str(g), str(y)): i
        for i, (g, y) in enumerate(zip(condition_gids, condition_yss, strict=True))
    }


def censored_mmd_squared(
    paths_x: torch.Tensor,
    paths_y: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    height_scale: float,
    t_norm: torch.Tensor,
) -> float:
    """Unbiased CSig-MMD² between weighted paths and a zero-path pivot.

    Uses the same kernel and self-term convention as ``normalized_sig_mmd``.
    """
    m = paths_x.shape[0]
    n = paths_y.shape[0]

    zero_heights = torch.zeros(1, t_norm.shape[0], dtype=torch.float32)
    zero_path = to_metric_paths(zero_heights, height_scale, t_norm)  # (1, 2G-1, 3)

    paths_x, paths_y, zero_path, w, v = to_kernel_device(
        paths_x, paths_y, zero_path, w, v
    )

    stacked = torch.cat([paths_x, paths_y, zero_path], dim=0)  # (m+n+1, 2G-1, 3)
    gram = signature_kernel_gram(stacked, stacked)
    diag = gram.diagonal()
    gram = cosine_normalized(gram, diag, diag)

    k_xx = gram[:m, :m]
    k_yy = gram[m : m + n, m : m + n]
    k_xy = gram[:m, m : m + n]
    k_x0 = gram[:m, -1]
    k_y0 = gram[m : m + n, -1]
    k_00 = gram[-1, -1]  # = 1 after cosine normalisation

    return _censored_mmd_from_grams(k_xx, k_xy, k_yy, k_x0, k_y0, k_00, w, v)


def _censored_mmd_from_grams(
    k_xx: torch.Tensor,
    k_xy: torch.Tensor,
    k_yy: torch.Tensor,
    k_x0: torch.Tensor,
    k_y0: torch.Tensor,
    k_00: torch.Tensor | float,
    w: torch.Tensor,
    v: torch.Tensor,
) -> float:
    """Unbiased CSig-MMD² from normalised Gram blocks and zero-path kernels."""
    one_minus_w = 1 - w
    one_minus_v = 1 - v

    censored_xx = (
        torch.outer(w, w) * k_xx
        + torch.outer(w * k_x0, one_minus_w)
        + torch.outer(one_minus_w, w * k_x0)
        + torch.outer(one_minus_w, one_minus_w) * k_00
    )
    censored_yy = (
        torch.outer(v, v) * k_yy
        + torch.outer(v * k_y0, one_minus_v)
        + torch.outer(one_minus_v, v * k_y0)
        + torch.outer(one_minus_v, one_minus_v) * k_00
    )
    censored_xy = (
        torch.outer(w, v) * k_xy
        + torch.outer(w * k_x0, one_minus_v)
        + torch.outer(one_minus_w, v * k_y0)
        + torch.outer(one_minus_w, one_minus_v) * k_00
    )

    return mmd_squared(censored_xx, censored_xy, censored_yy)


def sig_and_csig_mmd_squared(
    paths_x: torch.Tensor,
    paths_y: torch.Tensor,
    weight_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    height_scale: float,
    t_norm: torch.Tensor,
    *,
    sigma: float = 1.0,
) -> tuple[float, list[float]]:
    """Sig-MMD² and one CSig-MMD² per weight pair of one block from one Gram set.

    ``paths_x`` are the oracle paths and ``paths_y`` the model paths; each weight pair
    holds the oracle weights ``w`` and the model weights ``v``. The Sig-MMD² equals
    ``normalized_sig_mmd(paths_y, paths_x)`` (model first, as the blocked Sig-MMD
    scorer calls it). Each CSig-MMD² equals ``censored_mmd_squared`` up to float32
    summation order. The three Gram blocks are computed separately, as in
    ``normalized_sig_mmd``, so kernel memory is bounded by the largest block pair
    rather than by the stacked sample. ``sigma`` is the RBF static-kernel parameter.
    """
    zero_heights = torch.zeros(1, t_norm.shape[0], dtype=torch.float32)
    zero_path = to_metric_paths(zero_heights, height_scale, t_norm)

    paths_x, paths_y, zero_path = to_kernel_device(
        paths_x.float(), paths_y.float(), zero_path.float()
    )
    k_yy, k_yx, k_xx = kernel_grams(paths_y, paths_x, sigma=sigma)
    diag_x = k_xx.diagonal()
    diag_y = k_yy.diagonal()
    diag_zero = signature_kernel_gram(zero_path, zero_path, sigma=sigma).diagonal()
    k_x0 = cosine_normalized(
        signature_kernel_gram(paths_x, zero_path, sigma=sigma), diag_x, diag_zero
    ).squeeze(1)
    k_y0 = cosine_normalized(
        signature_kernel_gram(paths_y, zero_path, sigma=sigma), diag_y, diag_zero
    ).squeeze(1)

    k_yy = cosine_normalized(k_yy, diag_y, diag_y)
    k_yx = cosine_normalized(k_yx, diag_y, diag_x)
    k_xx = cosine_normalized(k_xx, diag_x, diag_x)

    sig = mmd_squared(k_yy, k_yx, k_xx)
    csigs = []
    for w, v in weight_pairs:
        w, v = to_kernel_device(w.float(), v.float())
        csigs.append(
            _censored_mmd_from_grams(k_xx, k_yx.T, k_yy, k_x0, k_y0, 1.0, w, v)
        )
    return sig, csigs


@dataclass(frozen=True)
class Censoring:
    """One resolved censoring rule: quantile ``alpha``, sharpness ``beta``, ``c²``."""

    alpha: float
    beta: float
    c_squared: float

    def weights(self, mahalanobis_distances: np.ndarray) -> torch.Tensor:
        distances = np.asarray(mahalanobis_distances)
        return torch.from_numpy(
            1.0 / (1.0 + np.exp(-self.beta * (distances - self.c_squared)))
        ).float()

    @property
    def name(self) -> str:
        return f"alpha={self.alpha:.4f}_beta={self.beta:.4f}"


@dataclass(frozen=True)
class CensoringSweep:
    """Extra censoring rules scored from the same Gram matrices as the main rule.

    Thresholds come from ``lodged_recalls`` (the lodged-recall rule) and ``alphas``
    (quantiles); without either, the main threshold is used. Each threshold is
    combined with every ``beta_times_c_squared`` (beta = value / c²) and every fixed
    ``betas`` value; without either, the main rule's ``beta_times_c_squared``.
    """

    lodged_recalls: tuple[float, ...] = ()
    alphas: tuple[float, ...] = ()
    beta_times_c_squared: tuple[float, ...] = ()
    betas: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.rules(CensoringParameters())  # validates every value

    def __bool__(self) -> bool:
        return any(
            (self.lodged_recalls, self.alphas, self.beta_times_c_squared, self.betas)
        )

    def rules(self, main: CensoringParameters) -> list[CensoringParameters]:
        """Every threshold combined with every beta, as unresolved rules."""
        thresholds = [
            replace(main, alpha=None, lodged_recall=recall)
            for recall in self.lodged_recalls
        ] + [replace(main, alpha=alpha) for alpha in self.alphas] or [main]
        return [
            rule
            for threshold in thresholds
            for rule in [
                replace(threshold, beta=None, beta_times_c_squared=product)
                for product in self.beta_times_c_squared
            ]
            + [replace(threshold, beta=beta) for beta in self.betas]
            or [replace(threshold, beta=None)]
        ]

    def resolve(
        self, artifact: SignatureMahalanobisArtifact, main: CensoringParameters
    ) -> tuple[Censoring, ...]:
        censorings = tuple(rule.resolve(artifact) for rule in self.rules(main))
        names = [censoring.name for censoring in censorings]
        if len(set(names)) != len(names):
            msg = f"Censoring sweep repeats a rule: {names}"
            raise ValueError(msg)
        return censorings

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        for flag, help_text in (
            ("--sweep-lodged-recalls", "lodged recalls of the lodged-recall rule"),
            ("--sweep-alphas", "quantiles of the censoring reference distances"),
            ("--sweep-beta-times-c-squared", "sharpness values beta * c²"),
            ("--sweep-betas", "fixed beta values"),
        ):
            parser.add_argument(
                flag,
                type=float,
                nargs="+",
                default=(),
                help=(
                    f"CSig-MMD sweep: {help_text}. Each extra rule is scored from the "
                    "same Gram matrices and written to <csig-output>/sweep/<rule>."
                ),
            )

    def to_arguments(self) -> list[str]:
        """The command-line arguments that ``from_args`` reads back."""
        arguments: list[str] = []
        for flag, values in (
            ("--sweep-lodged-recalls", self.lodged_recalls),
            ("--sweep-alphas", self.alphas),
            ("--sweep-beta-times-c-squared", self.beta_times_c_squared),
            ("--sweep-betas", self.betas),
        ):
            if values:
                arguments += [flag, *(str(value) for value in values)]
        return arguments

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> CensoringSweep:
        return cls(
            lodged_recalls=tuple(args.sweep_lodged_recalls),
            alphas=tuple(args.sweep_alphas),
            beta_times_c_squared=tuple(args.sweep_beta_times_c_squared),
            betas=tuple(args.sweep_betas),
        )


@dataclass(frozen=True)
class CensoringReference:
    """The CSig-MMD censoring rule of one signature-Mahalanobis artifact.

    Scorers take the oracle trajectories from the dataset, as for Sig-MMD; the
    artifact supplies the oracle Mahalanobis distances, the MRCD fit that places the
    model draws, and the distances that define the threshold ``c_squared``.
    """

    artifact: SignatureMahalanobisArtifact
    censorings: tuple[Censoring, ...]  # the main rule first, then the sweep
    depth: int
    distances: np.ndarray  # (n_artifact_conditions, draw_count)
    condition_index: dict[tuple[str, str], int]

    @property
    def main(self) -> Censoring:
        return self.censorings[0]

    @property
    def draw_count(self) -> int:
        return int(self.distances.shape[1])

    def artifact_row(self, genotype_id: str, yearsite_uid: str) -> int:
        key = (genotype_id, yearsite_uid)
        if key not in self.condition_index:
            msg = (
                f"Signature-Mahalanobis artifact {str(self.artifact.path)!r} does "
                f"not contain condition (genotype_id={genotype_id!r}, "
                f"yearsite_uid={yearsite_uid!r})"
            )
            raise ValueError(msg)
        return self.condition_index[key]

    def score(
        self,
        oracle_paths: torch.Tensor,
        model_paths: torch.Tensor,
        oracle_distances: np.ndarray,
        height_scale: float,
        t_norm: torch.Tensor,
        self_kernel_batch_size: int,
        *,
        label: str,
        sigma: float = 1.0,
    ) -> list[dict[str, float]]:
        """Sig-MMD² and CSig-MMD² of one oracle/model sample pair, per censoring.

        Returns one entry per rule in ``censorings`` (the main rule first).
        ``sig_score`` equals ``normalized_sig_mmd(model_paths, oracle_paths)``.
        ``sigma`` sets the RBF static kernel of the MMD Gram matrices; the censoring
        weights come from the artifact's signatures and do not depend on it.
        """
        model_signatures = compute_normalized_truncated_sigs(
            model_paths.cpu().numpy().copy(),
            depth=self.depth,
            self_kernel_batch_size=self_kernel_batch_size,
        )
        model_distances = mahalanobis_distance(
            model_signatures, self.artifact.mrcd_location, self.artifact.mrcd_precision
        )
        weight_pairs = [
            (censoring.weights(oracle_distances), censoring.weights(model_distances))
            for censoring in self.censorings
        ]
        sig_score, scores = sig_and_csig_mmd_squared(
            oracle_paths, model_paths, weight_pairs, height_scale, t_norm, sigma=sigma
        )
        results = []
        for censoring, (w, v), score in zip(
            self.censorings, weight_pairs, scores, strict=True
        ):
            if not np.isfinite(score):
                msg = (
                    f"Non-finite CSig-MMD score for {label} ({censoring.name}): "
                    f"{score!r}"
                )
                raise ValueError(msg)
            results.append(
                {
                    "mean_w_oracle": float(w.mean()),
                    "mean_w_model": float(v.mean()),
                    "score": score,
                    "sig_score": sig_score,
                }
            )
        return results


def load_censoring_reference(
    path: str | Path,
    censoring: CensoringParameters,
    *,
    sweep: CensoringSweep | None = None,
    prediction_day_axis: torch.Tensor,
    metric_day_axis_norm: torch.Tensor,
    height_scale: float,
) -> CensoringReference:
    """Load an artifact and check that it matches the scorer's metric paths."""
    artifact = load_signature_mahalanobis_artifact(path)
    metadata = artifact.metadata
    for key in ("draw_count", "depth", "height_scale"):
        if key not in metadata:
            msg = f"Signature-Mahalanobis artifact metadata is missing {key!r}"
            raise ValueError(msg)
    draw_count = int(metadata["draw_count"])
    depth = int(metadata["depth"])
    if draw_count <= 0 or depth <= 0:
        msg = "Signature-Mahalanobis artifact draw_count and depth must be > 0"
        raise ValueError(msg)
    if not np.isclose(float(metadata["height_scale"]), height_scale):
        msg = (
            f"Artifact height_scale {metadata['height_scale']} differs from the "
            f"scorer height scale {height_scale}"
        )
        raise ValueError(msg)
    for name, artifact_axis, scorer_axis in (
        ("prediction_day_axis", artifact.prediction_day_axis, prediction_day_axis),
        ("metric_day_axis_norm", artifact.metric_day_axis_norm, metric_day_axis_norm),
    ):
        artifact_tensor = torch.from_numpy(np.asarray(artifact_axis).copy()).float()
        scorer_tensor = scorer_axis.detach().cpu().float()
        if artifact_tensor.shape != scorer_tensor.shape or not torch.allclose(
            artifact_tensor, scorer_tensor
        ):
            msg = f"Artifact {name} differs from the method prediction grid"
            raise ValueError(msg)

    all_distances = np.asarray(artifact.mahalanobis_distances)
    n_conditions = len(artifact.condition_genotype_ids)
    if all_distances.ndim != 1 or all_distances.shape[0] != n_conditions * draw_count:
        msg = (
            "Artifact Mahalanobis distances must be flat with draw_count entries per "
            f"condition: {all_distances.shape} for {n_conditions} conditions and "
            f"draw_count={draw_count}"
        )
        raise ValueError(msg)
    return CensoringReference(
        artifact=artifact,
        censorings=(
            censoring.resolve(artifact),
            *(sweep.resolve(artifact, censoring) if sweep else ()),
        ),
        depth=depth,
        distances=all_distances.reshape(n_conditions, draw_count),
        condition_index=build_condition_index(
            artifact.condition_genotype_ids, artifact.condition_yearsite_uids
        ),
    )
