#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib>=3.8",
#     "scipy>=1.11",
#     "trimesh",
# ]
# ///
"""
Interactive tongue ground-truth editor.

Displays a sagittal cross-section of the face model at each frame,
with 4 draggable tongue anchor points (T4-back → T1-tip).
The user positions anchors per-phoneme to define MRI-grounded targets,
which are saved to JSON for downstream comparison/alignment.

Controls
--------
← / →       prev / next keyframe within current phone
↑ / ↓       prev / next instance of current phone label
PageUp/Dn   prev / next phone class
1-7          jump to phone class by number
Tab/Shift+Tab  next/prev phone (any class, sequential)
G            jump to time (enter seconds in terminal)
F            find phone by label (enter in terminal)
S            save current keyframe GT
E            export all GT to JSON
R            reset GT anchors to current EMA values
Q            export and quit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from face_model_io_trimesh import load_face_model_trimesh
    from test import process_beat_data, load_ema_motion, FaceKitTongueRig, TONGUE_CONFIG
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from tongue_scripts.face_model_io_trimesh import load_face_model_trimesh
    from tongue_scripts.test import (
        process_beat_data,
        load_ema_motion,
        FaceKitTongueRig,
        TONGUE_CONFIG,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TONGUE_SLICE = slice(16611, 17039)
ANCHOR_INDICES = [16661, 16696, 16755, 16758]
BONE_INDICES = [16661, 16757]
ANCHOR_NAMES = ["T4 (Back)", "T3 (Dorsum)", "T2 (Blade)", "T1 (Tip)"]
FPS = 50

PHONE_CLASSES = OrderedDict(
    [
        ("Alveolar", ["T", "D", "N", "L", "S", "Z"]),
        ("Velar", ["K", "G", "NG"]),
        ("Palatal", ["CH", "JH", "SH"]),
        ("Dental", ["TH", "DH"]),
        ("Liquid", ["R", "W", "Y"]),
        ("Vowel-Open", ["AA", "AE", "AH"]),
        ("Vowel-Close", ["IY", "UW", "EH"]),
    ]
)

DRAG_RADIUS = 0.8  # plot-unit threshold for grabbing an anchor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class PhoneInterval:
    idx: int
    start: float
    end: float
    label: str
    normalized: str


@dataclass
class WordInterval:
    start: float
    end: float
    text: str


@dataclass
class GTKeyframe:
    frame_idx: int
    time: float
    anchors_yz: List[List[float]]


@dataclass
class GTTarget:
    phone_label: str
    phone_normalized: str
    interval_idx: int
    time_start: float
    time_end: float
    keyframes: List[GTKeyframe] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TextGrid helpers
# ---------------------------------------------------------------------------
def normalize_phone(label: str) -> str:
    return re.sub(r"[0-9]", "", label.upper())


def _parse_textgrid_tier(textgrid_path: Path, tier_name: str):
    """Yield (idx, start, end, text) tuples for a TextGrid tier."""
    in_tier = False
    current: dict = {}
    idx = 0
    with textgrid_path.open("r", encoding="utf-8") as f:
        for raw in f:
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
                current["s"] = line.split("=", 1)[1].strip()
            elif line.startswith("xmax ="):
                current["e"] = line.split("=", 1)[1].strip()
            elif line.startswith("text ="):
                txt = line.split("=", 1)[1].strip()
                if txt.startswith('"') and txt.endswith('"'):
                    txt = txt[1:-1]
                current["t"] = txt
                if {"s", "e", "t"} <= current.keys():
                    try:
                        s, e = float(current["s"]), float(current["e"])
                    except ValueError:
                        s, e = 0.0, 0.0
                    yield idx, s, e, current["t"]
                    idx += 1


def parse_phones(path: Path) -> List[PhoneInterval]:
    out = []
    for idx, s, e, txt in _parse_textgrid_tier(path, "phones"):
        if txt.strip():
            out.append(
                PhoneInterval(idx=idx, start=s, end=e, label=txt, normalized=normalize_phone(txt))
            )
    return out


def parse_words(path: Path) -> List[WordInterval]:
    out = []
    for _, s, e, txt in _parse_textgrid_tier(path, "words"):
        if txt.strip():
            out.append(WordInterval(start=s, end=e, text=txt))
    return out


def get_keyframe_indices(phone: PhoneInterval, fps: float, n_kf: int = 4) -> List[int]:
    s = int(phone.start * fps)
    e = int(phone.end * fps)
    n = max(1, e - s)
    if n <= n_kf:
        return list(range(s, max(s + 1, e)))
    step = (n - 1) / (n_kf - 1)
    return [s + int(round(i * step)) for i in range(n_kf)]


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------
class TongueGTEditor:
    def __init__(
        self,
        dataset_id: str,
        beat_root: str,
        npy_dir: str,
        std_path: str,
        face_model_dir: str,
        output_dir: str,
    ):
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gt_path = self.output_dir / f"{dataset_id}_tongue_gt.json"

        beat_root_p = Path(beat_root)
        self.json_path = beat_root_p / f"{dataset_id}.json"
        self.textgrid_path = beat_root_p / f"{dataset_id}.TextGrid"
        self.npy_path = Path(npy_dir) / f"{dataset_id}.npy"

        # ------ load data ------
        print("Loading face model …")
        self.face_model = load_face_model_trimesh(face_model_dir)

        print("Loading BEAT blendshapes …")
        try:
            self.face_seq = process_beat_data(str(self.json_path), self.face_model, target_fps=FPS)
        except Exception as exc:
            print(f"  Warning: BEAT JSON load failed ({exc}); using neutral face")
            self.face_seq = np.zeros((3000, len(self.face_model.expression_names)), dtype=np.float32)

        print("Setting up tongue rig …")
        self.tongue_rig = FaceKitTongueRig(
            self.face_model.neutral_verts,
            self.face_model.faces,
            TONGUE_SLICE,
            ANCHOR_INDICES,
            BONE_INDICES,
            TONGUE_CONFIG,
        )

        print("Loading EMA motion …")
        self.ema_seq = load_ema_motion(
            str(self.npy_path), std_path, self.tongue_rig.anchors, TONGUE_CONFIG["std_scalar"]
        )

        print("Parsing TextGrid …")
        self.all_phones = parse_phones(self.textgrid_path)
        self.all_words = parse_words(self.textgrid_path)

        # phone lookup: normalized → [PhoneInterval, …]
        self.phone_lookup: Dict[str, List[PhoneInterval]] = {}
        for p in self.all_phones:
            self.phone_lookup.setdefault(p.normalized, []).append(p)

        # navigable (class_name, phone_label) pairs — only those present in data
        self.nav_classes: List[Tuple[str, str]] = []
        for cls, labels in PHONE_CLASSES.items():
            for lbl in labels:
                if lbl in self.phone_lookup:
                    self.nav_classes.append((cls, lbl))
        if not self.nav_classes:
            raise RuntimeError("No matching phone classes found in TextGrid")

        self.max_frame = min(len(self.face_seq), len(self.ema_seq)) - 1

        # navigation state
        self.nav_idx = 0
        self.instance_idx = 0
        self.keyframe_idx = 0

        # GT anchors for current editing (4, 3) in rig space
        self.gt_anchors_3d = self.ema_seq[0].copy()

        # drag state
        self.dragging: Optional[int] = None

        # saved GT
        self.gt_targets: List[GTTarget] = []
        self._load_gt()

        # precompute face-only midline mask (exclude tongue+gum region for clean display)
        self._face_midline = np.abs(self.face_model.neutral_verts[:, 0]) < 0.5
        self._face_midline[TONGUE_SLICE] = False

        self._setup_figure()
        self._navigate_to_current()

    # ----- helpers -----

    def _label(self) -> Optional[str]:
        return self.nav_classes[self.nav_idx][1] if self.nav_classes else None

    def _instances(self) -> List[PhoneInterval]:
        lbl = self._label()
        return self.phone_lookup.get(lbl, []) if lbl else []

    def _phone(self) -> Optional[PhoneInterval]:
        insts = self._instances()
        return insts[self.instance_idx] if insts and self.instance_idx < len(insts) else None

    def _keyframes(self) -> List[int]:
        ph = self._phone()
        return get_keyframe_indices(ph, FPS) if ph else []

    def _frame(self) -> int:
        kfs = self._keyframes()
        if kfs and self.keyframe_idx < len(kfs):
            return min(kfs[self.keyframe_idx], self.max_frame)
        return 0

    def _word_at(self, t: float) -> str:
        for w in self.all_words:
            if w.start <= t <= w.end:
                return w.text
        return ""

    # ----- figure setup -----

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.canvas.manager.set_window_title(f"Tongue GT Editor — {self.dataset_id}")

        # main sagittal axes
        self.ax = self.fig.add_axes([0.04, 0.22, 0.58, 0.74])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("Z (Anterior →)")
        self.ax.set_ylabel("Y (Superior →)")
        self.ax.set_xlim(-2, 14)
        self.ax.set_ylim(-12, 2)

        # info text axes
        self.ax_info = self.fig.add_axes([0.65, 0.22, 0.33, 0.74])
        self.ax_info.axis("off")

        # phone timeline bar (between main view and slider)
        self.ax_timeline = self.fig.add_axes([0.04, 0.12, 0.58, 0.07])

        # frame slider
        self.ax_slider = self.fig.add_axes([0.04, 0.03, 0.58, 0.04])
        self.slider = Slider(self.ax_slider, "Frame", 0, self.max_frame, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        # buttons
        self.ax_btn_save = self.fig.add_axes([0.68, 0.03, 0.08, 0.04])
        self.btn_save = Button(self.ax_btn_save, "Save (S)")
        self.btn_save.on_clicked(lambda _: self._save_keyframe())

        self.ax_btn_export = self.fig.add_axes([0.78, 0.03, 0.08, 0.04])
        self.btn_export = Button(self.ax_btn_export, "Export (E)")
        self.btn_export.on_clicked(lambda _: self._export_gt())

        self.ax_btn_reset = self.fig.add_axes([0.88, 0.03, 0.08, 0.04])
        self.btn_reset = Button(self.ax_btn_reset, "Reset (R)")
        self.btn_reset.on_clicked(lambda _: self._reset_gt())

        # artists — face scatter
        self.face_scat = self.ax.scatter([], [], s=1, c="#aaaaaa", alpha=0.4, zorder=1)
        # gum scatter (14062..tongue_start)
        self._gum_midline = np.abs(self.face_model.neutral_verts[:, 0]) < 0.5
        self._gum_midline[:14062] = False
        self._gum_midline[TONGUE_SLICE] = False
        self.gum_scat = self.ax.scatter([], [], s=1, c="#cc9999", alpha=0.35, zorder=1)
        # tongue scatter (from EMA rig)
        self.tongue_scat = self.ax.scatter([], [], s=3, c="#88bbff", alpha=0.5, zorder=2)
        # EMA spline & anchors
        (self.ema_line,) = self.ax.plot([], [], "b-", lw=1.5, alpha=0.5, zorder=3, label="EMA spline")
        self.ema_anch = self.ax.scatter(
            [], [], s=80, c="blue", marker="x", linewidths=1.5, zorder=4, label="EMA anchors"
        )
        # GT spline & anchors
        (self.gt_line,) = self.ax.plot([], [], "-", color="#00cc44", lw=2.5, zorder=5, label="GT spline")
        self.gt_anch = self.ax.scatter(
            [], [], s=180, c="red", marker="o", edgecolors="darkred", linewidths=2, zorder=6, label="GT (drag)"
        )
        # anchor labels (T4..T1)
        self.anchor_texts = [
            self.ax.text(0, 0, ANCHOR_NAMES[i], fontsize=7, color="darkred", zorder=7)
            for i in range(4)
        ]

        self.ax.legend(loc="upper left", fontsize=8)

        # connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ----- drawing -----

    def _navigate_to_current(self):
        f = self._frame()
        self.slider.set_val(f)
        self._init_gt_from_ema(f)
        self._restore_saved_gt()
        self._draw(f)

    def _init_gt_from_ema(self, frame: int):
        frame = max(0, min(frame, self.max_frame))
        self.gt_anchors_3d = self.ema_seq[frame].copy()

    def _restore_saved_gt(self):
        """If GT was saved for the current phone/keyframe, reload it."""
        ph = self._phone()
        if ph is None:
            return
        kfs = self._keyframes()
        if not kfs or self.keyframe_idx >= len(kfs):
            return
        f = kfs[self.keyframe_idx]
        for tgt in self.gt_targets:
            if tgt.interval_idx == ph.idx:
                for kf in tgt.keyframes:
                    if kf.frame_idx == f:
                        for i, (y, z) in enumerate(kf.anchors_yz):
                            self.gt_anchors_3d[i, 1] = y
                            self.gt_anchors_3d[i, 2] = z
                        return

    def _draw(self, frame: int | None = None):
        if frame is None:
            frame = self._frame()
        frame = max(0, min(frame, self.max_frame))

        # --- face mesh ---
        if frame < len(self.face_seq):
            w = {n: v for n, v in zip(self.face_model.expression_names, self.face_seq[frame])}
        else:
            w = {}
        verts = self.face_model.deform(w).copy()

        face_pts = verts[self._face_midline][:, [2, 1]]
        self.face_scat.set_offsets(face_pts)
        gum_pts = verts[self._gum_midline][:, [2, 1]]
        self.gum_scat.set_offsets(gum_pts)

        # --- tongue from EMA ---
        if frame < len(self.ema_seq):
            ema_anc = self.ema_seq[frame]
            t_verts, _, t_sp = self.tongue_rig.deform(ema_anc)
            t_mid = np.abs(t_verts[:, 0]) < 1.0
            self.tongue_scat.set_offsets(t_verts[t_mid][:, [2, 1]])
            u = np.linspace(0, 1, 100)
            sp_pts = t_sp(u)[:, [2, 1]]
            self.ema_line.set_data(sp_pts[:, 0], sp_pts[:, 1])
            self.ema_anch.set_offsets(ema_anc[:, [2, 1]])
        else:
            self.tongue_scat.set_offsets(np.empty((0, 2)))
            self.ema_line.set_data([], [])
            self.ema_anch.set_offsets(np.empty((0, 2)))

        # --- GT visual ---
        self._update_gt_visual()

        # --- info panel ---
        self._update_info(frame)

        # --- timeline bar ---
        self._draw_timeline(frame)

        self.fig.canvas.draw_idle()

    def _update_gt_visual(self):
        zy = self.gt_anchors_3d[:, [2, 1]]
        self.gt_anch.set_offsets(zy)
        for i in range(4):
            self.anchor_texts[i].set_position((zy[i, 0] + 0.15, zy[i, 1] + 0.15))
        try:
            sp = make_interp_spline(np.linspace(0, 1, 4), self.gt_anchors_3d, k=3)
            pts = sp(np.linspace(0, 1, 100))[:, [2, 1]]
            self.gt_line.set_data(pts[:, 0], pts[:, 1])
        except Exception:
            self.gt_line.set_data([], [])

    def _update_info(self, frame: int):
        self.ax_info.clear()
        self.ax_info.axis("off")

        t = frame / FPS
        ph = self._phone()
        cls = self.nav_classes[self.nav_idx][0] if self.nav_classes else "—"
        lbl = self._label() or "—"
        insts = self._instances()
        kfs = self._keyframes()
        word = self._word_at(t)
        n_saved = sum(len(tg.keyframes) for tg in self.gt_targets)

        # which keyframes of this phone instance already have GT?
        saved_kf_frames = set()
        if ph:
            for tg in self.gt_targets:
                if tg.interval_idx == ph.idx:
                    for kf in tg.keyframes:
                        saved_kf_frames.add(kf.frame_idx)

        kf_status = []
        for ki, kf_frame in enumerate(kfs):
            marker = "●" if kf_frame in saved_kf_frames else "○"
            cur = " ◄" if ki == self.keyframe_idx else ""
            kf_status.append(f"  [{ki+1}] f{kf_frame} {marker}{cur}")

        lines = [
            f"Frame: {frame}   Time: {t:.2f}s",
            f'Word: "{word}"',
            "",
            f"Phone Class: {cls}",
            f"Phone: /{lbl}/   ({ph.label if ph else '—'})",
            f"Instance: {self.instance_idx + 1}/{len(insts)}",
            f"  time: {ph.start:.2f}–{ph.end:.2f}s" if ph else "",
            f"Keyframes:  (● saved  ○ unsaved)",
            *kf_status,
            "",
            "GT Anchors (Y, Z):",
        ]
        for i in range(4):
            lines.append(f"  {ANCHOR_NAMES[i]}: ({self.gt_anchors_3d[i,1]:.2f}, {self.gt_anchors_3d[i,2]:.2f})")
        lines += [
            "",
            f"Total saved keyframes: {n_saved}",
            "",
            "── Navigation ──",
            "← →   keyframe  ↑ ↓ instance",
            "PgUp/Dn  phone class",
            "Tab / Shift+Tab  next/prev phone",
        ]
        # class shortcuts
        class_names = list(PHONE_CLASSES.keys())
        for ci, cn in enumerate(class_names):
            marker = "▸" if ci < len(class_names) and self.nav_classes and self.nav_classes[self.nav_idx][0] == cn else " "
            lines.append(f" {marker}{ci+1}  {cn}")
        lines += [
            "G  go to time   F  find phone",
            "S  save   E  export",
            "R  reset  Q  quit",
        ]

        self.ax_info.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=self.ax_info.transAxes,
            fontsize=9,
            va="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

    def _draw_timeline(self, frame: int):
        """Draw a compact phone timeline bar with colour-coded phone classes."""
        self.ax_timeline.clear()
        total_t = (self.max_frame + 1) / FPS

        # colour map for phone classes
        class_colours = {
            "Alveolar": "#e6194b",
            "Velar": "#3cb44b",
            "Palatal": "#4363d8",
            "Dental": "#f58231",
            "Liquid": "#911eb4",
            "Vowel-Open": "#42d4f4",
            "Vowel-Close": "#f032e6",
        }

        # gather saved interval indices
        saved_idxs = {tg.interval_idx for tg in self.gt_targets if tg.keyframes}

        # determine visible window (centre on current time, show ±2s for zoom)
        cur_t = frame / FPS
        win_half = 2.0  # Zoom in: show only ±2s
        t_lo = max(0, cur_t - win_half)
        t_hi = min(total_t, cur_t + win_half)
        # if near edges expand the other side
        if t_lo == 0:
            t_hi = min(total_t, 2 * win_half)
        if t_hi == total_t:
            t_lo = max(0, total_t - 2 * win_half)

        # draw phone rectangles
        for ph in self.all_phones:
            if ph.end < t_lo or ph.start > t_hi:
                continue
            # determine colour
            colour = "#cccccc"
            for cls, labels in PHONE_CLASSES.items():
                if ph.normalized in labels:
                    colour = class_colours.get(cls, "#cccccc")
                    break
            alpha = 0.85 if ph.idx in saved_idxs else 0.35
            x0 = max(ph.start, t_lo)
            x1 = min(ph.end, t_hi)
            # Increase thickness: set height to 2.0 (was 0.8)
            self.ax_timeline.barh(0, x1 - x0, left=x0, height=2.0, color=colour, alpha=alpha, edgecolor="none")
            # label if wide enough
            w = x1 - x0
            if w > 0.04:
                self.ax_timeline.text(
                    (x0 + x1) / 2, 0, ph.normalized, ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold", clip_on=True,
                )

        # highlight current phone
        ph = self._phone()
        if ph:
            # Increase thickness: set height to 2.0 (was 0.8)
            self.ax_timeline.barh(0, ph.end - ph.start, left=ph.start, height=2.0,
                                  fill=False, edgecolor="red", linewidth=2)

        # playhead
        self.ax_timeline.axvline(cur_t, color="black", lw=1.5, zorder=10)

        self.ax_timeline.set_xlim(t_lo, t_hi)
        # Adjust y-limits to fit thicker bars
        self.ax_timeline.set_ylim(-1.2, 1.2)
        self.ax_timeline.set_yticks([])
        self.ax_timeline.set_xlabel(f"Time (s)  —  window {t_lo:.1f}–{t_hi:.1f}s  of  {total_t:.1f}s", fontsize=8)
        self.ax_timeline.tick_params(axis="x", labelsize=7)

    # ----- events -----

    def _on_slider(self, val):
        self._draw(int(val))

    def _on_press(self, event):
        if event.button != 1:
            return
        if event.inaxes == self.ax_timeline:
            # Click-to-jump on timeline: go to the phone at that time
            if event.xdata is not None:
                self._jump_to_time(event.xdata)
            return
        if event.inaxes != self.ax:
            return
        zy = self.gt_anchors_3d[:, [2, 1]]
        click = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(zy - click, axis=1)
        if dists.min() < DRAG_RADIUS:
            self.dragging = int(dists.argmin())

    def _on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax:
            return
        self.gt_anchors_3d[self.dragging, 2] = event.xdata  # Z
        self.gt_anchors_3d[self.dragging, 1] = event.ydata  # Y
        self._update_gt_visual()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self.dragging is not None:
            self.dragging = None
            self._draw()

    def _on_key(self, event):
        if event.key == "right":
            kfs = self._keyframes()
            if kfs and self.keyframe_idx < len(kfs) - 1:
                self._auto_save()
                self.keyframe_idx += 1
                self._navigate_to_current()
        elif event.key == "left":
            if self.keyframe_idx > 0:
                self._auto_save()
                self.keyframe_idx -= 1
                self._navigate_to_current()
        elif event.key == "up":
            insts = self._instances()
            if insts and self.instance_idx < len(insts) - 1:
                self._auto_save()
                self.instance_idx += 1
                self.keyframe_idx = 0
                self._navigate_to_current()
        elif event.key == "down":
            if self.instance_idx > 0:
                self._auto_save()
                self.instance_idx -= 1
                self.keyframe_idx = 0
                self._navigate_to_current()
        elif event.key == "pageup":
            if self.nav_idx < len(self.nav_classes) - 1:
                self._auto_save()
                self.nav_idx += 1
                self.instance_idx = 0
                self.keyframe_idx = 0
                self._navigate_to_current()
        elif event.key == "pagedown":
            if self.nav_idx > 0:
                self._auto_save()
                self.nav_idx -= 1
                self.instance_idx = 0
                self.keyframe_idx = 0
                self._navigate_to_current()
        # ----- Number keys 1-7: jump directly to phone class -----
        elif event.key in ("1", "2", "3", "4", "5", "6", "7"):
            target_cls_idx = int(event.key) - 1
            class_names = list(PHONE_CLASSES.keys())
            if target_cls_idx < len(class_names):
                target_cls = class_names[target_cls_idx]
                # find first nav entry for this class
                for ni, (cn, _) in enumerate(self.nav_classes):
                    if cn == target_cls:
                        self._auto_save()
                        self.nav_idx = ni
                        self.instance_idx = 0
                        self.keyframe_idx = 0
                        self._navigate_to_current()
                        break
        # ----- Tab / Shift+Tab: sequential phone navigation -----
        elif event.key == "tab":
            self._jump_sequential_phone(forward=True)
        elif event.key == "shift+tab":
            self._jump_sequential_phone(forward=False)
        # ----- G: go to time -----
        elif event.key == "g":
            self._goto_time()
        # ----- F: find phone by label -----
        elif event.key == "f":
            self._find_phone()
        elif event.key == "s":
            self._save_keyframe()
        elif event.key == "e":
            self._export_gt()
        elif event.key == "r":
            self._reset_gt()
        elif event.key == "q":
            self._export_gt()
            plt.close(self.fig)

    def _jump_sequential_phone(self, forward: bool = True):
        """Jump to the next/prev phone interval sequentially in time."""
        ph = self._phone()
        if ph is None:
            return
        # find this phone's position in all_phones by interval_idx
        current_pos = None
        for i, p in enumerate(self.all_phones):
            if p.idx == ph.idx:
                current_pos = i
                break
        if current_pos is None:
            return

        step = 1 if forward else -1
        pos = current_pos + step
        while 0 <= pos < len(self.all_phones):
            candidate = self.all_phones[pos]
            # find this phone in nav_classes
            for ni, (cn, lbl) in enumerate(self.nav_classes):
                if lbl == candidate.normalized:
                    # find instance index
                    instances = self.phone_lookup.get(lbl, [])
                    for ii, inst in enumerate(instances):
                        if inst.idx == candidate.idx:
                            self._auto_save()
                            self.nav_idx = ni
                            self.instance_idx = ii
                            self.keyframe_idx = 0
                            self._navigate_to_current()
                            return
            pos += step
        print("  (no more navigable phones in that direction)")

    def _goto_time(self):
        """Jump to a specific time by prompting in the terminal."""
        try:
            total_t = (self.max_frame + 1) / FPS
            s = input(f"  Go to time (0–{total_t:.1f}s): ").strip()
            t = float(s)
            target_frame = int(t * FPS)
            target_frame = max(0, min(target_frame, self.max_frame))
            self._jump_to_time(t, fallback_frame=target_frame)
        except (ValueError, EOFError):
            print("  (cancelled)")

    def _jump_to_time(self, t: float, fallback_frame: Optional[int] = None):
        """Jump to the phone interval that contains time t."""
        if fallback_frame is None:
            fallback_frame = int(t * FPS)
        fallback_frame = max(0, min(fallback_frame, self.max_frame))
        # find nearest phone at that time
        for ph in self.all_phones:
            if ph.start <= t <= ph.end:
                for ni, (cn, lbl) in enumerate(self.nav_classes):
                    if lbl == ph.normalized:
                        instances = self.phone_lookup.get(lbl, [])
                        for ii, inst in enumerate(instances):
                            if inst.idx == ph.idx:
                                self._auto_save()
                                self.nav_idx = ni
                                self.instance_idx = ii
                                self.keyframe_idx = 0
                                self._navigate_to_current()
                                return
                # phone exists but not in our nav classes — just move slider
                break
        # fallback: just move to that frame
        self.slider.set_val(fallback_frame)
        self._init_gt_from_ema(fallback_frame)
        self._draw(fallback_frame)
        print(f"  Jumped to frame {fallback_frame} ({t:.2f}s)")

    def _find_phone(self):
        """Find a phone by label string."""
        try:
            labels_present = sorted(set(p.normalized for p in self.all_phones if p.normalized))
            print(f"  Available phones: {', '.join(labels_present)}")
            s = input("  Enter phone label (e.g. T, K, IY): ").strip().upper()
            s = re.sub(r"[0-9]", "", s)
            if not s:
                print("  (cancelled)")
                return
            for ni, (cn, lbl) in enumerate(self.nav_classes):
                if lbl == s:
                    self._auto_save()
                    self.nav_idx = ni
                    self.instance_idx = 0
                    self.keyframe_idx = 0
                    self._navigate_to_current()
                    return
            print(f"  Phone /{s}/ not found in navigation classes")
        except (ValueError, EOFError):
            print("  (cancelled)")

    # ----- GT save / load -----

    def _auto_save(self):
        """Silently persist current anchors when navigating away."""
        self._save_keyframe(silent=True)

    def _save_keyframe(self, silent: bool = False):
        ph = self._phone()
        if ph is None:
            return
        kfs = self._keyframes()
        if not kfs or self.keyframe_idx >= len(kfs):
            return
        f = kfs[self.keyframe_idx]
        yz = [[float(self.gt_anchors_3d[i, 1]), float(self.gt_anchors_3d[i, 2])] for i in range(4)]
        kf = GTKeyframe(frame_idx=f, time=f / FPS, anchors_yz=yz)

        # find or create target
        tgt = None
        for t in self.gt_targets:
            if t.interval_idx == ph.idx:
                tgt = t
                break
        if tgt is None:
            tgt = GTTarget(
                phone_label=ph.label,
                phone_normalized=ph.normalized,
                interval_idx=ph.idx,
                time_start=ph.start,
                time_end=ph.end,
            )
            self.gt_targets.append(tgt)

        # update or append
        for i, ek in enumerate(tgt.keyframes):
            if ek.frame_idx == f:
                tgt.keyframes[i] = kf
                if not silent:
                    print(f"  Updated /{ph.normalized}/ frame {f}")
                self._draw()
                return
        tgt.keyframes.append(kf)
        tgt.keyframes.sort(key=lambda k: k.frame_idx)
        if not silent:
            print(f"  Saved /{ph.normalized}/ frame {f}")
        self._draw()

    def _reset_gt(self):
        self._init_gt_from_ema(self._frame())
        self._draw()

    def _export_gt(self):
        data = {
            "dataset_id": self.dataset_id,
            "analysis_fps": FPS,
            "anchor_names": ANCHOR_NAMES,
            "tongue_config": TONGUE_CONFIG,
            "targets": [
                {
                    "phone_label": t.phone_label,
                    "phone_normalized": t.phone_normalized,
                    "interval_idx": t.interval_idx,
                    "time_start": t.time_start,
                    "time_end": t.time_end,
                    "keyframes": [
                        {"frame_idx": kf.frame_idx, "time": kf.time, "anchors_yz": kf.anchors_yz}
                        for kf in t.keyframes
                    ],
                }
                for t in self.gt_targets
            ],
        }
        self.gt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        n = sum(len(t.keyframes) for t in self.gt_targets)
        print(f"Exported {n} GT keyframes → {self.gt_path}")

    def _load_gt(self):
        if not self.gt_path.exists():
            return
        try:
            raw = json.loads(self.gt_path.read_text(encoding="utf-8"))
            for td in raw.get("targets", []):
                tgt = GTTarget(
                    phone_label=td["phone_label"],
                    phone_normalized=td["phone_normalized"],
                    interval_idx=td["interval_idx"],
                    time_start=td["time_start"],
                    time_end=td["time_end"],
                )
                for kfd in td.get("keyframes", []):
                    tgt.keyframes.append(
                        GTKeyframe(
                            frame_idx=kfd["frame_idx"],
                            time=kfd["time"],
                            anchors_yz=kfd["anchors_yz"],
                        )
                    )
                self.gt_targets.append(tgt)
            n = sum(len(t.keyframes) for t in self.gt_targets)
            print(f"Loaded {n} GT keyframes from {self.gt_path}")
        except Exception as exc:
            print(f"Warning: could not load GT file: {exc}")

    # ----- entry -----

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Interactive tongue ground-truth editor")
    parser.add_argument("--dataset-id", default="1_wayne_0_112_112")
    parser.add_argument(
        "--beat-root",
        default=str(
            PROJECT_ROOT
            / "ADFA_EVALUATION"
            / "data"
            / "beat_cache_speaker1"
            / "beat_english_v0.2.1"
            / "beat_english_v0.2.1"
            / "1"
        ),
    )
    parser.add_argument("--tongue-npy-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument(
        "--std-path", default=str(SCRIPT_DIR / "normalising_vectors" / "JW13_4points_std.npy")
    )
    parser.add_argument("--face-model-dir", default=str(PROJECT_ROOT / "FaceXModel"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "jaw_tongue_sync"))
    args = parser.parse_args()

    editor = TongueGTEditor(
        dataset_id=args.dataset_id,
        beat_root=args.beat_root,
        npy_dir=args.tongue_npy_dir,
        std_path=args.std_path,
        face_model_dir=args.face_model_dir,
        output_dir=args.output_dir,
    )
    editor.run()


if __name__ == "__main__":
    main()
