#!/usr/bin/env python3
"""
Profile-based articulation optimizer for tongue .npy motion files.

Supports multiple phoneme articulation profiles:
  DENTAL        – /TH/, /DH/     → teeth_edge contact (tip + blade)
  ALVEOLAR      – /T/, /D/, /N/  → alveolar_ridge contact (tip + blade)
  ALVEOLAR_FRIC – /S/, /Z/       → near-alveolar with groove (tip + blade + groove)
  VELAR         – /K/, /G/       → soft palate contact (back/dorsum, not tip)

Architecture:
  ArticulationProfile  – per-profile target type, anchor weights, boost factor
  AnatomicalTargets    – teeth_edge ↔ alveolar_ridge ↔ velum auto-sync
  ArticulationRule     – runtime rule assembled from profile + CLI overrides
  optimize_motion()    – frame loop with raised-cosine gating + temporal smoothing

Interactive picking is profile-aware: it prompts for the correct anatomical
landmark and can auto-infer related targets via anatomical proportions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
TONGUE_ANIMATION_DIR = SCRIPT_DIR / "tongue_animation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TONGUE_ANIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(TONGUE_ANIMATION_DIR))

from face_model_io_trimesh import load_face_model_trimesh
from generate_tongue_animation import (
    FaceKitTongueRig,
    TONGUE_SLICE,
    ANCHOR_INDICES,
    BONE_INDICES,
)


# ============================================================
# Constants
# ============================================================
FPS_DEFAULT = 50.0
GUM_START_IDX = 14062
GUM_END_IDX = 17039

# Anchor indices within the 4-point EMA array  (0=T4 back … 3=T1 tip)
BACK_ANCHOR_IDX = 0   # T4 – tongue dorsum / back
DORSUM_ANCHOR_IDX = 1  # T3 – mid-dorsum
BLADE_ANCHOR_IDX = 2   # T2 – blade
TIP_ANCHOR_IDX = 3     # T1 – tip


# ============================================================
# Enums & dataclasses
# ============================================================
class TargetType(str, Enum):
    """Anatomical target that a profile aims for."""
    TEETH_EDGE = "teeth_edge"
    ALVEOLAR_RIDGE = "alveolar_ridge"
    NEAR_ALVEOLAR = "near_alveolar"   # fricatives: small gap preserved
    SOFT_PALATE = "soft_palate"       # velum


class ProfileName(str, Enum):
    DENTAL = "DENTAL"
    ALVEOLAR = "ALVEOLAR"
    ALVEOLAR_FRIC = "ALVEOLAR_FRIC"
    VELAR = "VELAR"


@dataclass
class PhoneInterval:
    start: float
    end: float
    label: str


@dataclass
class ArticulationProfile:
    """Immutable preset that describes *what* to do for a phoneme class."""
    name: ProfileName
    labels: List[str]
    target_type: TargetType
    # Per-anchor strengths (0 = ignore, 1 = full attraction)
    tip_strength: float
    blade_strength: float
    back_strength: float           # used by VELAR
    # Over-articulation boost: >1.0 exaggerates toward target
    boost_factor: float
    # Temporal envelope
    ramp_seconds: float
    max_move: float
    temporal_smoothing: float
    # Fricative-specific: minimum gap distance to preserve
    fricative_gap: float = 0.0
    # Human-readable description for interactive prompts
    description: str = ""


@dataclass
class ArticulationRule:
    """Runtime rule: profile merged with any CLI overrides."""
    name: str
    labels: List[str]
    target_type: TargetType
    ramp_seconds: float
    tip_strength: float
    blade_strength: float
    back_strength: float
    boost_factor: float
    max_move: float
    temporal_smoothing: float
    fricative_gap: float = 0.0


@dataclass
class AnatomicalTargets:
    """
    Resolved geometric targets in (z, y) sagittal coordinates.

    All fields are 1-D arrays of shape (2,) → [z, y].
    teeth_edge is the primary anchor; alveolar_ridge and soft_palate
    can be auto-inferred or manually overridden.
    """
    teeth_edge: np.ndarray
    alveolar_ridge: Optional[np.ndarray] = None
    soft_palate: Optional[np.ndarray] = None

    # ---- auto-sync from teeth_edge ----
    # Anatomical heuristic offsets (in rig units, positive Y = superior,
    # positive Z = anterior).  Calibrated from ICT FaceKit geometry.
    _RIDGE_DY: float = 1.8    # alveolar ridge sits ~1.8 units above teeth edge
    _RIDGE_DZ: float = -1.2   # … and ~1.2 units posterior
    _VELUM_DY: float = 4.5    # soft palate is ~4.5 units above teeth edge
    _VELUM_DZ: float = -6.0   # … and ~6.0 units posterior

    def sync_from_teeth_edge(self) -> None:
        """Infer alveolar_ridge and soft_palate from teeth_edge."""
        te = self.teeth_edge
        if self.alveolar_ridge is None:
            self.alveolar_ridge = np.array(
                [te[0] + self._RIDGE_DZ, te[1] + self._RIDGE_DY], dtype=np.float32
            )
        if self.soft_palate is None:
            self.soft_palate = np.array(
                [te[0] + self._VELUM_DZ, te[1] + self._VELUM_DY], dtype=np.float32
            )

    def target_for(self, target_type: TargetType) -> np.ndarray:
        """Return the (z, y) target point matching a TargetType."""
        self.sync_from_teeth_edge()
        mapping = {
            TargetType.TEETH_EDGE: self.teeth_edge,
            TargetType.ALVEOLAR_RIDGE: self.alveolar_ridge,
            TargetType.NEAR_ALVEOLAR: self.alveolar_ridge,  # gap handled in optimizer
            TargetType.SOFT_PALATE: self.soft_palate,
        }
        tgt = mapping.get(target_type)
        if tgt is None:
            raise ValueError(f"No resolved target for {target_type}")
        return tgt


# ============================================================
# Profile registry
# ============================================================
PROFILE_REGISTRY: Dict[ProfileName, ArticulationProfile] = {
    ProfileName.DENTAL: ArticulationProfile(
        name=ProfileName.DENTAL,
        labels=["TH", "DH"],
        target_type=TargetType.TEETH_EDGE,
        tip_strength=0.95,
        blade_strength=0.55,
        back_strength=0.0,
        boost_factor=1.15,       # slight overshoot – tongue peeks past teeth
        ramp_seconds=0.03,
        max_move=1.10,
        temporal_smoothing=0.30,
        description="Dental fricatives: tongue tip contacts upper teeth edge",
    ),
    ProfileName.ALVEOLAR: ArticulationProfile(
        name=ProfileName.ALVEOLAR,
        labels=["T", "D", "N"],
        target_type=TargetType.ALVEOLAR_RIDGE,
        tip_strength=0.90,
        blade_strength=0.60,
        back_strength=0.0,
        boost_factor=1.10,
        ramp_seconds=0.025,
        max_move=1.20,
        temporal_smoothing=0.28,
        description="Alveolar stops/nasal: tongue tip taps alveolar ridge",
    ),
    ProfileName.ALVEOLAR_FRIC: ArticulationProfile(
        name=ProfileName.ALVEOLAR_FRIC,
        labels=["S", "Z"],
        target_type=TargetType.NEAR_ALVEOLAR,
        tip_strength=0.80,
        blade_strength=0.65,
        back_strength=0.0,
        boost_factor=1.05,
        ramp_seconds=0.03,
        max_move=0.90,
        temporal_smoothing=0.35,
        fricative_gap=0.25,      # preserve ~0.25 rig-unit gap for airflow
        description="Alveolar fricatives: tongue near ridge with central groove",
    ),
    ProfileName.VELAR: ArticulationProfile(
        name=ProfileName.VELAR,
        labels=["K", "G"],
        target_type=TargetType.SOFT_PALATE,
        tip_strength=0.10,       # tip stays mostly neutral
        blade_strength=0.20,
        back_strength=0.90,      # dorsum does the work
        boost_factor=1.12,
        ramp_seconds=0.025,
        max_move=1.30,
        temporal_smoothing=0.25,
        description="Velar stops: tongue dorsum contacts soft palate",
    ),
}


def get_profile(name: str) -> ArticulationProfile:
    """Look up a profile by name (case-insensitive)."""
    key = name.upper().replace("-", "_")
    try:
        return PROFILE_REGISTRY[ProfileName(key)]
    except (ValueError, KeyError):
        valid = ", ".join(p.value for p in ProfileName)
        raise ValueError(f"Unknown profile '{name}'. Valid profiles: {valid}")


def list_profiles() -> str:
    """Return a human-readable summary of all registered profiles."""
    lines = []
    for p in PROFILE_REGISTRY.values():
        lines.append(
            f"  {p.name.value:16s}  labels={','.join(p.labels):10s}  "
            f"target={p.target_type.value:18s}  boost={p.boost_factor:.2f}  "
            f"– {p.description}"
        )
    return "\n".join(lines)


# ============================================================
# Core helpers  (unchanged from original where possible)
# ============================================================
def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def parse_textgrid_intervals(textgrid_path: Path, tier_name: str) -> List[PhoneInterval]:
    intervals: List[PhoneInterval] = []
    in_tier = False
    current: Dict[str, str] = {}

    with textgrid_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("item ["):
                in_tier = False
                continue
            if line.startswith('name = "'):
                tier = line.split("=", 1)[1].strip().strip('"')
                in_tier = tier == tier_name
                continue
            if not in_tier:
                continue
            if line.startswith("intervals ["):
                current = {}
                continue
            if line.startswith("xmin ="):
                current["start"] = line.split("=", 1)[1].strip()
            elif line.startswith("xmax ="):
                current["end"] = line.split("=", 1)[1].strip()
            elif line.startswith("text ="):
                txt = line.split("=", 1)[1].strip().strip('"')
                current["text"] = txt
                if {"start", "end", "text"} <= current.keys():
                    try:
                        s = float(current["start"])
                        e = float(current["end"])
                    except ValueError:
                        s, e = 0.0, 0.0
                    if txt.strip():
                        intervals.append(PhoneInterval(start=s, end=e, label=txt))
    return intervals


def raised_cosine_window(t: float, start: float, end: float, ramp: float) -> float:
    if t < start - ramp or t > end + ramp:
        return 0.0
    if start <= t <= end:
        return 1.0
    if t < start:
        x = (t - (start - ramp)) / max(ramp, 1e-6)
        return float(0.5 - 0.5 * np.cos(np.pi * x))
    x = (t - end) / max(ramp, 1e-6)
    return float(0.5 + 0.5 * np.cos(np.pi * x))


def raw_to_denorm_anchors(
    raw_ema_4x2: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    denorm = np.zeros((len(raw_ema_4x2), 4, 3), dtype=np.float32)
    denorm[:, :, 0] = rig_anchors_4x3[:, 0][None, :]
    denorm[:, :, 1] = rig_anchors_4x3[:, 1][None, :] + raw_ema_4x2[:, :, 1] * std_4x2[:, 1][None, :] * scalar
    denorm[:, :, 2] = rig_anchors_4x3[:, 2][None, :] + raw_ema_4x2[:, :, 0] * std_4x2[:, 0][None, :] * scalar
    return denorm


def denorm_to_raw_anchors(
    denorm_4x3: np.ndarray,
    std_4x2: np.ndarray,
    rig_anchors_4x3: np.ndarray,
    scalar: float,
) -> np.ndarray:
    raw = np.zeros((len(denorm_4x3), 4, 2), dtype=np.float32)
    raw[:, :, 1] = (denorm_4x3[:, :, 1] - rig_anchors_4x3[:, 1][None, :]) / (
        np.maximum(np.abs(std_4x2[:, 1][None, :] * scalar), 1e-6)
    )
    raw[:, :, 0] = (denorm_4x3[:, :, 2] - rig_anchors_4x3[:, 2][None, :]) / (
        np.maximum(np.abs(std_4x2[:, 0][None, :] * scalar), 1e-6)
    )
    return raw


# ============================================================
# Anatomical target estimation
# ============================================================
def estimate_default_teeth_yz(face_model) -> np.ndarray:
    """Estimate teeth-edge position from the gum/teeth vertex region."""
    verts = face_model.neutral_verts
    end_idx = min(GUM_END_IDX, len(verts))
    start_idx = min(GUM_START_IDX, end_idx)
    gum = verts[start_idx:end_idx]
    if len(gum) == 0:
        p = np.mean(verts, axis=0)
        return np.array([p[2], p[1]], dtype=np.float32)

    z_hi = np.percentile(gum[:, 2], 92)
    y_hi = np.percentile(gum[:, 1], 65)
    cand = gum[(gum[:, 2] >= z_hi) & (gum[:, 1] >= y_hi)]
    if len(cand) < 5:
        cand = gum[gum[:, 2] >= z_hi]
    if len(cand) == 0:
        cand = gum
    pt = np.mean(cand, axis=0)
    return np.array([pt[2], pt[1]], dtype=np.float32)


def build_anatomical_targets(
    face_model,
    teeth_yz_override: Optional[np.ndarray] = None,
    alveolar_yz_override: Optional[np.ndarray] = None,
    velum_yz_override: Optional[np.ndarray] = None,
) -> AnatomicalTargets:
    """Build an AnatomicalTargets, auto-syncing missing fields."""
    teeth = teeth_yz_override if teeth_yz_override is not None else estimate_default_teeth_yz(face_model)
    targets = AnatomicalTargets(
        teeth_edge=teeth,
        alveolar_ridge=alveolar_yz_override,
        soft_palate=velum_yz_override,
    )
    targets.sync_from_teeth_edge()
    return targets


# ============================================================
# Interactive picking  (profile-aware)
# ============================================================
def parse_yz(s: Optional[str]) -> Optional[np.ndarray]:
    if not s:
        return None
    try:
        z_s, y_s = s.split(",")
        return np.array([float(z_s), float(y_s)], dtype=np.float32)
    except Exception:
        raise ValueError(f"Could not parse yz pair: {s!r}. Expected format: 'z,y'")


def _ginput_one(ax, title: str) -> np.ndarray:
    ax.set_title(title, fontsize=10)
    plt.draw()
    pts = plt.ginput(1, timeout=-1)
    if not pts:
        raise RuntimeError("No point selected.")
    z, y = pts[0]
    ax.scatter([z], [y], c="yellow", s=70, marker="x")
    plt.draw()
    return np.array([z, y], dtype=np.float32)


_TARGET_PICK_PROMPTS: Dict[TargetType, str] = {
    TargetType.TEETH_EDGE: "Click TEETH EDGE (upper incisor tip)",
    TargetType.ALVEOLAR_RIDGE: "Click ALVEOLAR RIDGE (gum behind upper teeth)",
    TargetType.NEAR_ALVEOLAR: "Click ALVEOLAR RIDGE (tongue approaches but keeps gap)",
    TargetType.SOFT_PALATE: "Click SOFT PALATE / VELUM (rear roof of mouth)",
}


def pick_targets_interactive(
    face_model,
    denorm_anchors: np.ndarray,
    active_frame_idx: np.ndarray,
    profile: ArticulationProfile,
    existing_targets: AnatomicalTargets,
) -> AnatomicalTargets:
    """
    Profile-aware interactive target picking.

    Always asks for teeth_edge first (primary anchor), then shows the
    auto-inferred secondary targets and lets the user accept or override
    the one that matches the active profile.
    """
    verts = face_model.neutral_verts
    midline = np.abs(verts[:, 0]) < 0.6
    yz_face = verts[midline][:, [2, 1]]

    # Choose the track to highlight based on which anchor the profile moves most
    if profile.back_strength > profile.tip_strength:
        track_idx = BACK_ANCHOR_IDX
        track_label = "back/dorsum in target windows"
    else:
        track_idx = TIP_ANCHOR_IDX
        track_label = "tip in target windows"

    track_yz = denorm_anchors[:, track_idx][:, [2, 1]]
    active_track = track_yz[active_frame_idx] if len(active_frame_idx) else track_yz

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(yz_face[:, 0], yz_face[:, 1], s=1, c="#b0b0b0", alpha=0.35, label="face midline")
    if len(active_track) > 0:
        ax.scatter(active_track[:, 0], active_track[:, 1], s=7, c="#d62728", alpha=0.35, label=track_label)
    ax.set_xlabel("Z (Anterior →)")
    ax.set_ylabel("Y (Superior →)")
    ax.legend(loc="best")

    # --- Step 1: pick teeth_edge (always) ---
    teeth_yz = _ginput_one(ax, "Step 1/2: Click TEETH EDGE (upper incisor tip)")

    # --- auto-sync to compute all derived targets ---
    targets = AnatomicalTargets(teeth_edge=teeth_yz)
    targets.sync_from_teeth_edge()

    # Show inferred points
    ax.scatter([targets.alveolar_ridge[0]], [targets.alveolar_ridge[1]],
               c="cyan", s=90, marker="^", label="auto: alveolar ridge")
    ax.scatter([targets.soft_palate[0]], [targets.soft_palate[1]],
               c="magenta", s=90, marker="s", label="auto: soft palate")
    ax.legend(loc="best")
    plt.draw()

    # --- Step 2: let user accept or override the profile-specific target ---
    prompt = _TARGET_PICK_PROMPTS.get(
        profile.target_type,
        f"Click target for {profile.target_type.value}",
    )
    override_yz = _ginput_one(ax, f"Step 2/2: {prompt}  (or click near auto-marker to accept)")

    # Store the override in the correct slot
    if profile.target_type in (TargetType.TEETH_EDGE,):
        targets.teeth_edge = override_yz
    elif profile.target_type in (TargetType.ALVEOLAR_RIDGE, TargetType.NEAR_ALVEOLAR):
        targets.alveolar_ridge = override_yz
    elif profile.target_type == TargetType.SOFT_PALATE:
        targets.soft_palate = override_yz

    plt.close(fig)
    return targets


def pick_interval_targets_interactive(
    denorm_anchors: np.ndarray,
    intervals: List[PhoneInterval],
    fps: float,
    primary_anchor_idx: int = TIP_ANCHOR_IDX,
) -> Dict[int, np.ndarray]:
    targets: Dict[int, np.ndarray] = {}
    if not intervals:
        return targets

    fig, ax = plt.subplots(figsize=(9, 6))
    track = denorm_anchors[:, primary_anchor_idx][:, [2, 1]]

    for idx, ph in enumerate(intervals):
        s = max(0, int(np.floor(ph.start * fps)))
        e = min(len(track), int(np.ceil(ph.end * fps)) + 1)
        seg = track[s:e]
        if len(seg) == 0:
            continue

        ax.clear()
        ax.scatter(track[:, 0], track[:, 1], s=2, c="#cccccc", alpha=0.25)
        ax.scatter(seg[:, 0], seg[:, 1], s=10, c="#d62728", alpha=0.9)
        ax.set_xlabel("Z")
        ax.set_ylabel("Y")
        ax.set_title(f"Interval {idx+1}/{len(intervals)}: /{ph.label}/ "
                      f"{ph.start:.3f}-{ph.end:.3f}s | Click target")
        plt.draw()

        pt = plt.ginput(1, timeout=-1)
        if pt:
            z, y = pt[0]
            targets[idx] = np.array([z, y], dtype=np.float32)
            ax.scatter([z], [y], s=70, c="yellow", marker="x")
            plt.draw()

    plt.close(fig)
    return targets


# ============================================================
# Frame-level optimization  (profile-aware)
# ============================================================
def apply_rule_to_frame(
    anchors: np.ndarray,
    target_yz: np.ndarray,
    weight: float,
    rule: ArticulationRule,
    prev_corrected: Optional[np.ndarray],
) -> np.ndarray:
    """
    Apply articulation rule to a single frame's 4 anchors.

    Supports:
    - per-anchor strengths (tip, blade, back)
    - boost_factor for over-articulation
    - fricative_gap to preserve minimum distance to target
    - temporal smoothing against previous corrected frame
    """
    if weight <= 1e-6:
        return anchors.copy()

    corrected = anchors.copy()

    def _move_anchor(anchor_idx: int, strength: float, tgt_yz: np.ndarray,
                     apply_gap: bool = False):
        alpha = float(np.clip(strength * weight, 0.0, 1.0))
        if alpha <= 1e-6:
            return
        src = corrected[anchor_idx].copy()
        tgt = np.array([src[0], tgt_yz[1], tgt_yz[0]], dtype=np.float32)

        # Boost: scale the displacement beyond natural to exaggerate articulation
        direction = tgt - src
        boosted_tgt = src + direction * rule.boost_factor

        moved = (1.0 - alpha) * src + alpha * boosted_tgt

        # Fricative gap: pull back if we got too close
        if apply_gap and rule.fricative_gap > 0:
            gap_vec = moved - tgt
            gap_dist = float(np.linalg.norm(gap_vec))
            if gap_dist < rule.fricative_gap and gap_dist > 1e-6:
                moved = tgt + gap_vec * (rule.fricative_gap / gap_dist)

        delta = moved - src
        dist = float(np.linalg.norm(delta))
        if dist > rule.max_move:
            moved = src + delta * (rule.max_move / max(dist, 1e-6))
        corrected[anchor_idx] = moved

    # --- Tip ---
    if rule.tip_strength > 0:
        is_fricative = rule.target_type == TargetType.NEAR_ALVEOLAR
        _move_anchor(TIP_ANCHOR_IDX, rule.tip_strength, target_yz,
                     apply_gap=is_fricative)

    # --- Blade ---
    if rule.blade_strength > 0:
        # Blade target is interpolated between current blade pos and the
        # primary target (same heuristic as original code)
        blade_current_yz = corrected[BLADE_ANCHOR_IDX][[2, 1]]
        blade_tgt = 0.5 * blade_current_yz + 0.5 * target_yz
        _move_anchor(BLADE_ANCHOR_IDX, rule.blade_strength, blade_tgt)

    # --- Back / Dorsum  (VELAR profile) ---
    if rule.back_strength > 0:
        _move_anchor(BACK_ANCHOR_IDX, rule.back_strength, target_yz)
        # Also gently lift dorsum (T3) as sympathetic co-articulation
        dorsum_tgt_yz = 0.6 * corrected[DORSUM_ANCHOR_IDX][[2, 1]] + 0.4 * target_yz
        _move_anchor(DORSUM_ANCHOR_IDX, rule.back_strength * 0.6, dorsum_tgt_yz)

    # --- Temporal smoothing ---
    if prev_corrected is not None:
        beta = float(np.clip(rule.temporal_smoothing, 0.0, 1.0))
        corrected = beta * prev_corrected + (1.0 - beta) * corrected

    return corrected


def build_active_intervals(all_intervals: List[PhoneInterval], labels: List[str]) -> List[PhoneInterval]:
    labels_norm = {normalize_phone(x) for x in labels}
    return [ph for ph in all_intervals if normalize_phone(ph.label) in labels_norm]


def best_interval_weight(t: float, intervals: List[PhoneInterval], ramp: float) -> Tuple[float, int]:
    best_w = 0.0
    best_i = -1
    for i, ph in enumerate(intervals):
        w = raised_cosine_window(t, ph.start, ph.end, ramp)
        if w > best_w:
            best_w = w
            best_i = i
    return best_w, best_i


def optimize_motion(
    denorm_anchors: np.ndarray,
    fps: float,
    intervals: List[PhoneInterval],
    rule: ArticulationRule,
    global_target_yz: np.ndarray,
    interval_targets: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    Run the optimization loop over all frames.

    For each frame, compute the raised-cosine weight from active intervals,
    then apply the articulation rule with the appropriate target.
    """
    out = denorm_anchors.copy()
    prev_corr = None

    for i in range(len(out)):
        t = i / fps
        w, interval_idx = best_interval_weight(t, intervals, rule.ramp_seconds)
        if w <= 0.0:
            prev_corr = out[i].copy()
            continue

        target = global_target_yz
        if interval_targets and interval_idx in interval_targets:
            target = interval_targets[interval_idx]

        out[i] = apply_rule_to_frame(out[i], target, w, rule, prev_corr)
        prev_corr = out[i].copy()

    return out


# ============================================================
# Profile → Rule assembly
# ============================================================
def profile_to_rule(profile: ArticulationProfile, **overrides) -> ArticulationRule:
    """
    Convert an ArticulationProfile to a runtime ArticulationRule,
    applying any CLI overrides on top.
    """
    params = dict(
        name=profile.name.value,
        labels=list(profile.labels),
        target_type=profile.target_type,
        ramp_seconds=profile.ramp_seconds,
        tip_strength=profile.tip_strength,
        blade_strength=profile.blade_strength,
        back_strength=profile.back_strength,
        boost_factor=profile.boost_factor,
        max_move=profile.max_move,
        temporal_smoothing=profile.temporal_smoothing,
        fricative_gap=profile.fricative_gap,
    )
    # Apply CLI overrides (only non-None values)
    for k, v in overrides.items():
        if v is not None and k in params:
            params[k] = v
    return ArticulationRule(**params)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile-based articulation optimizer for tongue .npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available profiles:\n{list_profiles()}",
    )

    # --- data paths ---
    parser.add_argument("--dataset-id", default="1_wayne_0_75_75")
    parser.add_argument("--npy-path", default=None,
                        help="Input .npy path (default: tongue_scripts/outputs/<dataset>.npy)")
    parser.add_argument("--out-path", default=None,
                        help="Output .npy path (default: tongue_scripts/outputs/<dataset>_optimized.npy)")
    parser.add_argument("--textgrid-path", default=None)
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--std-path", default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy"))
    parser.add_argument("--scalar", type=float, default=0.20)
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)

    # --- profile selection ---
    parser.add_argument("--profile", default="DENTAL",
                        choices=[p.value for p in ProfileName],
                        help="Articulation profile to apply (default: DENTAL)")
    parser.add_argument("--labels", default=None,
                        help="Override comma-separated phone labels (default: from profile)")

    # --- per-parameter overrides  (None = use profile default) ---
    parser.add_argument("--ramp-seconds", type=float, default=None)
    parser.add_argument("--tip-strength", type=float, default=None)
    parser.add_argument("--blade-strength", type=float, default=None)
    parser.add_argument("--back-strength", type=float, default=None)
    parser.add_argument("--boost-factor", type=float, default=None,
                        help="Over-articulation factor (>1.0 exaggerates)")
    parser.add_argument("--max-move", type=float, default=None)
    parser.add_argument("--temporal-smoothing", type=float, default=None)
    parser.add_argument("--fricative-gap", type=float, default=None)

    # --- interactive picking ---
    parser.add_argument("--interactive-pick", action="store_true", default=True)
    parser.add_argument("--no-interactive-pick", dest="interactive_pick", action="store_false")
    parser.add_argument("--per-interval-picks", action="store_true", default=False)

    # --- manual target overrides ---
    parser.add_argument("--teeth-yz", default=None, help="Manual teeth edge as 'z,y'")
    parser.add_argument("--alveolar-yz", default=None, help="Manual alveolar ridge as 'z,y'")
    parser.add_argument("--velum-yz", default=None, help="Manual soft palate as 'z,y'")
    parser.add_argument("--tip-target-yz", default=None,
                        help="Legacy: direct tip target override as 'z,y'")

    # --- multi-profile mode ---
    parser.add_argument("--multi-profile", nargs="*", default=None,
                        help="Run multiple profiles sequentially, e.g. --multi-profile DENTAL ALVEOLAR VELAR")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def run_single_profile(
    profile: ArticulationProfile,
    denorm: np.ndarray,
    all_intervals: List[PhoneInterval],
    fps: float,
    face_model,
    anatomical_targets: AnatomicalTargets,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, dict]:
    """Run optimization for a single profile. Returns (optimized_denorm, stats_dict)."""

    # Resolve labels
    labels = profile.labels
    if args.labels is not None and args.multi_profile is None:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    active_intervals = build_active_intervals(all_intervals, labels)
    if not active_intervals:
        print(f"  ⚠  No intervals for profile {profile.name.value} labels={labels}, skipping.")
        return denorm, {"active_intervals": 0, "active_frames": 0, "skipped": True}

    # Build runtime rule with any CLI overrides
    override_keys = [
        "ramp_seconds", "tip_strength", "blade_strength", "back_strength",
        "boost_factor", "max_move", "temporal_smoothing", "fricative_gap",
    ]
    overrides = {k: getattr(args, k.replace("-", "_"), None) for k in override_keys}
    overrides["labels"] = labels
    rule = profile_to_rule(profile, **overrides)

    # Compute active frame indices
    active_frame_idx = []
    for ph in active_intervals:
        s = max(0, int(np.floor(ph.start * fps)))
        e = min(len(denorm), int(np.ceil(ph.end * fps)) + 1)
        if e > s:
            active_frame_idx.extend(range(s, e))
    active_frame_idx = np.array(sorted(set(active_frame_idx)), dtype=int)

    # Interactive picking (profile-aware)
    if args.interactive_pick:
        anatomical_targets = pick_targets_interactive(
            face_model, denorm, active_frame_idx, profile, anatomical_targets
        )

    # Resolve the geometric target for this profile
    global_target_yz = anatomical_targets.target_for(profile.target_type)

    # Legacy --tip-target-yz override
    tip_override = parse_yz(args.tip_target_yz)
    if tip_override is not None:
        global_target_yz = tip_override

    # Per-interval picks
    interval_targets = None
    if args.interactive_pick and args.per_interval_picks:
        primary_idx = BACK_ANCHOR_IDX if profile.back_strength > profile.tip_strength else TIP_ANCHOR_IDX
        interval_targets = pick_interval_targets_interactive(
            denorm, active_intervals, fps, primary_anchor_idx=primary_idx
        )

    # Measure before
    measure_anchor = BACK_ANCHOR_IDX if profile.back_strength > profile.tip_strength else TIP_ANCHOR_IDX
    before_dist = np.linalg.norm(
        denorm[active_frame_idx, measure_anchor][:, [2, 1]] - global_target_yz[None, :], axis=1
    )

    # Optimize
    denorm_opt = optimize_motion(
        denorm, fps, active_intervals, rule,
        global_target_yz=global_target_yz,
        interval_targets=interval_targets,
    )

    after_dist = np.linalg.norm(
        denorm_opt[active_frame_idx, measure_anchor][:, [2, 1]] - global_target_yz[None, :], axis=1
    )

    stats = {
        "profile": profile.name.value,
        "labels": labels,
        "target_type": profile.target_type.value,
        "active_intervals": len(active_intervals),
        "active_frames": int(len(active_frame_idx)),
        "boost_factor": rule.boost_factor,
        "mean_dist_before": float(np.mean(before_dist)) if len(before_dist) else None,
        "mean_dist_after": float(np.mean(after_dist)) if len(after_dist) else None,
        "rule": {
            "name": rule.name,
            "ramp_seconds": rule.ramp_seconds,
            "tip_strength": rule.tip_strength,
            "blade_strength": rule.blade_strength,
            "back_strength": rule.back_strength,
            "boost_factor": rule.boost_factor,
            "max_move": rule.max_move,
            "temporal_smoothing": rule.temporal_smoothing,
            "fricative_gap": rule.fricative_gap,
        },
        "targets": {
            "teeth_edge": anatomical_targets.teeth_edge.tolist(),
            "alveolar_ridge": anatomical_targets.alveolar_ridge.tolist()
                if anatomical_targets.alveolar_ridge is not None else None,
            "soft_palate": anatomical_targets.soft_palate.tolist()
                if anatomical_targets.soft_palate is not None else None,
            "global_target_yz": global_target_yz.tolist(),
            "interval_targets": {str(k): v.tolist() for k, v in (interval_targets or {}).items()},
        },
    }

    if len(before_dist):
        print(f"  {profile.name.value}: anchor→target mean dist: "
              f"{np.mean(before_dist):.4f} → {np.mean(after_dist):.4f}  "
              f"({len(active_intervals)} intervals, {len(active_frame_idx)} frames)")

    return denorm_opt, stats


def main() -> None:
    args = parse_args()

    # --- Resolve paths ---
    npy_path = Path(args.npy_path) if args.npy_path else (
        SCRIPT_DIR / "outputs" / f"{args.dataset_id}.npy")
    out_path = Path(args.out_path) if args.out_path else (
        SCRIPT_DIR / "outputs" / f"{args.dataset_id}_optimized.npy")
    textgrid_path = Path(args.textgrid_path) if args.textgrid_path else (
        PROJECT_ROOT / "data" / "beat_cache" / "beat_english_v0.2.1"
        / "beat_english_v0.2.1" / "1" / f"{args.dataset_id}.TextGrid"
    )

    if not npy_path.exists():
        raise FileNotFoundError(f"Missing input npy: {npy_path}")
    if not textgrid_path.exists():
        raise FileNotFoundError(f"Missing TextGrid: {textgrid_path}")

    # --- Load data ---
    raw_motion = np.load(npy_path)
    if raw_motion.ndim != 2 or raw_motion.shape[1] < 8:
        raise ValueError(f"Expected (N, >=8) motion array, got shape={raw_motion.shape}")

    face_model = load_face_model_trimesh(str(Path(args.face_model_dir)))
    tongue_rig = FaceKitTongueRig(
        face_model.neutral_verts,
        face_model.faces,
        TONGUE_SLICE,
        ANCHOR_INDICES,
        BONE_INDICES,
        {
            "rotation_deg": 5,
            "thickness": 1.2,
            "shift_y": 0,
            "shift_z": 0,
            "std_scalar": args.scalar,
        },
    )

    std_raw = np.load(args.std_path)
    std_4x2 = std_raw.flatten()[:8].reshape(4, 2).astype(np.float32)
    raw_ema = raw_motion[:, :8].reshape(-1, 4, 2).astype(np.float32)
    denorm = raw_to_denorm_anchors(raw_ema, std_4x2, tongue_rig.anchors.astype(np.float32), args.scalar)

    all_intervals = parse_textgrid_intervals(textgrid_path, args.phone_tier)

    # --- Build anatomical targets from manual overrides or auto-estimation ---
    anatomical_targets = build_anatomical_targets(
        face_model,
        teeth_yz_override=parse_yz(args.teeth_yz),
        alveolar_yz_override=parse_yz(args.alveolar_yz),
        velum_yz_override=parse_yz(args.velum_yz),
    )

    # --- Determine which profiles to run ---
    if args.multi_profile:
        profile_names = [ProfileName(p.upper()) for p in args.multi_profile]
    else:
        profile_names = [ProfileName(args.profile.upper())]

    profiles = [PROFILE_REGISTRY[pn] for pn in profile_names]
    print(f"Running {len(profiles)} profile(s): {[p.name.value for p in profiles]}")

    # --- Sequential profile application ---
    all_stats: List[dict] = []
    denorm_current = denorm

    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"Profile: {profile.name.value}  –  {profile.description}")
        print(f"  labels = {profile.labels},  target = {profile.target_type.value}")
        print(f"  boost = {profile.boost_factor},  gap = {profile.fricative_gap}")
        print(f"{'='*60}")

        denorm_current, stats = run_single_profile(
            profile, denorm_current, all_intervals,
            args.fps, face_model, anatomical_targets, args,
        )
        all_stats.append(stats)

    # --- Write output ---
    raw_opt = denorm_to_raw_anchors(denorm_current, std_4x2,
                                     tongue_rig.anchors.astype(np.float32), args.scalar)
    out_motion = raw_motion.copy().astype(np.float32)
    out_motion[:, :8] = raw_opt.reshape(-1, 8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out_motion)

    # --- Sidecar report ---
    sidecar = out_path.with_suffix(".json")
    report = {
        "input_npy": str(npy_path),
        "output_npy": str(out_path),
        "textgrid": str(textgrid_path),
        "profiles_applied": [s.get("profile") for s in all_stats],
        "profile_stats": all_stats,
    }
    sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved optimized npy : {out_path}")
    print(f"Saved report        : {sidecar}")


if __name__ == "__main__":
    main()
