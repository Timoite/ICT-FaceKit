"""Phone-level confusion metrics for TextGrid-backed VSR evaluation.

The reference stream is expected to come from BEAT TextGrid ``phones`` tiers.
Hypothesis streams are generated from recognized text with the same
``g2p_en``/CMUDict ARPAbet convention used by :mod:`evaluation_script.ver`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Sequence

from evaluation_script.ver import VISUAL_MAP, VOWEL_PHONEMES, _g2p


DELETE_TOKEN = "<DEL>"
INSERT_TOKEN = "<INS>"
SPN_TOKEN = "SPN"

KNOWN_PHONES: set[str] = set(VISUAL_MAP) | set(VOWEL_PHONEMES)

FOCUSED_TONGUE_PHONES: tuple[str, ...] = (
    "TH",
    "DH",
    "T",
    "D",
    "N",
    "L",
    "S",
    "Z",
    "K",
    "G",
    "NG",
    "R",
    "CH",
    "JH",
    "SH",
    "ZH",
    "Y",
)

PHONE_ANALYSIS_CATEGORIES: dict[str, str] = {
    # Tongue-sensitive classes.
    "TH": "TONGUE_DENTAL",
    "DH": "TONGUE_DENTAL",
    "T": "TONGUE_ALVEOLAR",
    "D": "TONGUE_ALVEOLAR",
    "N": "TONGUE_ALVEOLAR",
    "S": "TONGUE_ALVEOLAR_FRIC",
    "Z": "TONGUE_ALVEOLAR_FRIC",
    "L": "TONGUE_LATERAL",
    "R": "TONGUE_RHOTIC",
    "ER": "TONGUE_RHOTIC",
    "K": "TONGUE_VELAR",
    "G": "TONGUE_VELAR",
    "NG": "TONGUE_VELAR",
    "SH": "TONGUE_POSTALVEOLAR",
    "ZH": "TONGUE_POSTALVEOLAR",
    "CH": "TONGUE_POSTALVEOLAR",
    "JH": "TONGUE_POSTALVEOLAR",
    "Y": "TONGUE_GLIDE_Y",
    # Lip/face-visible but not primary tongue targets for this experiment.
    "P": "LIP_BILABIAL",
    "B": "LIP_BILABIAL",
    "M": "LIP_BILABIAL",
    "F": "LIP_LABIODENTAL",
    "V": "LIP_LABIODENTAL",
    "W": "LIP_GLIDE_W",
    "HH": "GLOTTAL",
    # Vowels.
    "UW": "VOWEL_ROUNDED",
    "UH": "VOWEL_ROUNDED",
    "OW": "VOWEL_ROUNDED",
    "AO": "VOWEL_ROUNDED",
    "OY": "VOWEL_ROUNDED",
    "IY": "VOWEL_SPREAD",
    "IH": "VOWEL_SPREAD",
    "EY": "VOWEL_SPREAD",
    "EH": "VOWEL_SPREAD",
    "AA": "VOWEL_OPEN",
    "AE": "VOWEL_OPEN",
    "AH": "VOWEL_OPEN",
    "AX": "VOWEL_OPEN",
    "AW": "VOWEL_OPEN_ROUND",
    "AY": "VOWEL_OPEN_SPREAD",
}

TONGUE_SENSITIVE_PHONES: set[str] = {
    phone
    for phone, category in PHONE_ANALYSIS_CATEGORIES.items()
    if category.startswith("TONGUE_")
}


@dataclass(frozen=True)
class TextGridInterval:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AlignmentStep:
    op: str
    ref: str | None
    hyp: str | None
    ref_index: int | None
    hyp_index: int | None


@dataclass
class PhoneConfusionStats:
    ref_count: int = 0
    hyp_count: int = 0
    correct: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    per_ref_total: Counter[str] = field(default_factory=Counter)
    per_ref_correct: Counter[str] = field(default_factory=Counter)

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def phone_error_rate(self) -> float:
        if self.ref_count == 0:
            return 0.0 if self.total_errors == 0 else float("inf")
        return self.total_errors / float(self.ref_count)

    @property
    def phone_accuracy(self) -> float:
        if self.ref_count == 0:
            return 0.0
        return self.correct / float(self.ref_count)

    def merge(self, other: "PhoneConfusionStats") -> None:
        self.ref_count += other.ref_count
        self.hyp_count += other.hyp_count
        self.correct += other.correct
        self.substitutions += other.substitutions
        self.deletions += other.deletions
        self.insertions += other.insertions
        self.confusion.update(other.confusion)
        self.per_ref_total.update(other.per_ref_total)
        self.per_ref_correct.update(other.per_ref_correct)

    def recall_for(self, phone: str) -> float:
        total = self.per_ref_total.get(phone, 0)
        if total == 0:
            return 0.0
        return self.per_ref_correct.get(phone, 0) / float(total)

    def tongue_sensitive_recall(self) -> float:
        total = sum(self.per_ref_total.get(phone, 0) for phone in TONGUE_SENSITIVE_PHONES)
        correct = sum(self.per_ref_correct.get(phone, 0) for phone in TONGUE_SENSITIVE_PHONES)
        if total == 0:
            return 0.0
        return correct / float(total)


@dataclass(frozen=True)
class PhoneConfusionResult:
    ref_phones: list[str]
    hyp_phones: list[str]
    alignment: list[AlignmentStep]
    stats: PhoneConfusionStats


def normalize_phone_label(label: str) -> str:
    """Normalize TextGrid/g2p labels to stress-free uppercase ARPAbet."""
    return re.sub(r"[0-9]", "", label.strip().upper())


def parse_textgrid_intervals(textgrid_path: str | Path, tier_name: str) -> list[TextGridInterval]:
    """Parse a simple Praat TextGrid IntervalTier by name."""
    path = Path(textgrid_path)
    intervals: list[TextGridInterval] = []
    in_tier = False
    current: dict[str, str] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip().replace("\r", "")
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
                continue
            if line.startswith("xmax ="):
                current["end"] = line.split("=", 1)[1].strip()
                continue
            if line.startswith("text ="):
                text_value = line.split("=", 1)[1].strip()
                if text_value.startswith('"') and text_value.endswith('"'):
                    text_value = text_value[1:-1]
                current["text"] = text_value
                if {"start", "end", "text"} <= current.keys():
                    try:
                        start = float(current["start"])
                        end = float(current["end"])
                    except ValueError:
                        start, end = 0.0, 0.0
                    intervals.append(TextGridInterval(start=start, end=end, text=current["text"]))
    return intervals


def parse_textgrid_phones(
    textgrid_path: str | Path,
    tier_name: str = "phones",
    include_spn: bool = False,
) -> list[TextGridInterval]:
    """Return non-empty, normalized phone intervals from a TextGrid phone tier."""
    out: list[TextGridInterval] = []
    for interval in parse_textgrid_intervals(textgrid_path, tier_name):
        phone = normalize_phone_label(interval.text)
        if not phone:
            continue
        if phone == SPN_TOKEN and not include_spn:
            continue
        out.append(TextGridInterval(start=interval.start, end=interval.end, text=phone))
    return out


def text_to_phone_stream(text: str) -> list[str]:
    """Convert transcript text to normalized ARPAbet phones."""
    phones: list[str] = []
    for token in _g2p(text):
        if not token or token.isspace() or not token[0].isalpha():
            continue
        phone = normalize_phone_label(token)
        if phone in KNOWN_PHONES:
            phones.append(phone)
    return phones


def align_phones(ref_phones: Sequence[str], hyp_phones: Sequence[str]) -> list[AlignmentStep]:
    """Levenshtein-align two phone streams and return operation steps."""
    n = len(ref_phones)
    m = len(hyp_phones)
    costs = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = costs[i - 1][j - 1] + (0 if ref_phones[i - 1] == hyp_phones[j - 1] else 1)
            delete = costs[i - 1][j] + 1
            insert = costs[i][j - 1] + 1
            costs[i][j] = min(diag, delete, insert)

    steps_reversed: list[AlignmentStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag_cost = costs[i - 1][j - 1] + (
                0 if ref_phones[i - 1] == hyp_phones[j - 1] else 1
            )
            if costs[i][j] == diag_cost:
                op = "match" if ref_phones[i - 1] == hyp_phones[j - 1] else "sub"
                steps_reversed.append(
                    AlignmentStep(
                        op=op,
                        ref=ref_phones[i - 1],
                        hyp=hyp_phones[j - 1],
                        ref_index=i - 1,
                        hyp_index=j - 1,
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            steps_reversed.append(
                AlignmentStep(
                    op="del",
                    ref=ref_phones[i - 1],
                    hyp=None,
                    ref_index=i - 1,
                    hyp_index=None,
                )
            )
            i -= 1
            continue
        if j > 0:
            steps_reversed.append(
                AlignmentStep(
                    op="ins",
                    ref=None,
                    hyp=hyp_phones[j - 1],
                    ref_index=None,
                    hyp_index=j - 1,
                )
            )
            j -= 1

    return list(reversed(steps_reversed))


def stats_from_alignment(
    alignment: Iterable[AlignmentStep],
    ref_count: int,
    hyp_count: int,
) -> PhoneConfusionStats:
    stats = PhoneConfusionStats(ref_count=ref_count, hyp_count=hyp_count)

    for step in alignment:
        if step.ref is not None:
            stats.per_ref_total[step.ref] += 1
        if step.op == "match":
            if step.ref is None or step.hyp is None:
                continue
            stats.correct += 1
            stats.per_ref_correct[step.ref] += 1
            stats.confusion[(step.ref, step.hyp)] += 1
        elif step.op == "sub":
            if step.ref is None or step.hyp is None:
                continue
            stats.substitutions += 1
            stats.confusion[(step.ref, step.hyp)] += 1
        elif step.op == "del":
            if step.ref is None:
                continue
            stats.deletions += 1
            stats.confusion[(step.ref, DELETE_TOKEN)] += 1
        elif step.op == "ins":
            if step.hyp is None:
                continue
            stats.insertions += 1
            stats.confusion[(INSERT_TOKEN, step.hyp)] += 1
    return stats


def evaluate_phone_confusion(
    ref_phones: Sequence[str],
    hypothesis_text: str,
) -> PhoneConfusionResult:
    hyp_phones = text_to_phone_stream(hypothesis_text)
    ref_list = list(ref_phones)
    alignment = align_phones(ref_list, hyp_phones)
    stats = stats_from_alignment(alignment, ref_count=len(ref_list), hyp_count=len(hyp_phones))
    return PhoneConfusionResult(
        ref_phones=ref_list,
        hyp_phones=hyp_phones,
        alignment=alignment,
        stats=stats,
    )


def visual_category_for_phone(phone: str) -> str:
    return VISUAL_MAP.get(phone, "V_UNKNOWN")


def analysis_category_for_phone(phone: str) -> str:
    return PHONE_ANALYSIS_CATEGORIES.get(phone, "UNKNOWN")


def assert_known_phone_category_coverage() -> None:
    missing = sorted(KNOWN_PHONES - set(PHONE_ANALYSIS_CATEGORIES))
    if missing:
        raise AssertionError(f"Missing analysis categories for phones: {', '.join(missing)}")
