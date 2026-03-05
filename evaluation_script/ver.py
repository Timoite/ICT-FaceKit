"""
Viseme Error Rate (VER) evaluation.

Focuses only on VER using an expanded ARPAbet -> viseme mapping.
"""

from __future__ import annotations

import re

import g2p_en
import nltk
from jiwer import wer


# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
def _ensure_nltk(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


_ensure_nltk("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
_ensure_nltk("corpora/cmudict", "cmudict")

_g2p = g2p_en.G2p()


# ===================================================================
# 1. EXPANDED VISEME MAP  (all 39 ARPAbet phonemes from g2p_en / CMUDict)
# ===================================================================
# Design rationale
# ~~~~~~~~~~~~~~~~
# • Tongue-sensitive categories (DENTAL, ALVEOLAR, LATERAL, RHOTIC,
#   POSTALVEOLAR) are kept as *separate* visemes so that active-tongue
#   articulations produce measurably different viseme strings.
# • Vowels are split into ROUNDED / SPREAD / OPEN to capture lip-shape
#   differences visible on the rendered face mesh.
# • Glides W and Y get their own categories because they combine strong
#   lip rounding (W) or spreading (Y) with distinctive tongue movement.
# • Diphthongs AW (open→round) and AY (open→spread) each get a unique
#   viseme to capture the visible trajectory.

VISUAL_MAP: dict[str, str] = {
    # ---- Bilabial (lip closure) ----
    "P":  "V_BILABIAL",
    "B":  "V_BILABIAL",
    "M":  "V_BILABIAL",

    # ---- Labiodental (lower lip → upper teeth) ----
    "F":  "V_LABIODENTAL",
    "V":  "V_LABIODENTAL",

    # ---- Dental fricative (tongue visible between teeth) ----
    "TH": "V_DENTAL",
    "DH": "V_DENTAL",

    # ---- Alveolar plosive / nasal (tongue tip at alveolar ridge) ----
    "T":  "V_ALVEOLAR",
    "D":  "V_ALVEOLAR",
    "N":  "V_ALVEOLAR",

    # ---- Alveolar fricative (narrow groove, teeth close together) ----
    "S":  "V_ALVEOLAR_FRIC",
    "Z":  "V_ALVEOLAR_FRIC",

    # ---- Post-alveolar / palatal (lips slightly protruded, tongue blade back) ----
    "SH": "V_POSTALVEOLAR",
    "ZH": "V_POSTALVEOLAR",
    "CH": "V_POSTALVEOLAR",
    "JH": "V_POSTALVEOLAR",

    # ---- Velar (back of tongue → soft palate) ----
    "K":  "V_VELAR",
    "G":  "V_VELAR",
    "NG": "V_VELAR",

    # ---- Glottal (open mouth, no visible articulator) ----
    "HH": "V_GLOTTAL",

    # ---- Lateral (tongue tip up, sides lowered – visually distinct) ----
    "L":  "V_LATERAL",

    # ---- Rhotic (tongue curled back, slight lip rounding) ----
    "R":  "V_RHOTIC",

    # ---- Glide W (strong lip rounding + velar tongue position) ----
    "W":  "V_GLIDE_W",

    # ---- Glide Y (spread lips + high-front tongue / palatal) ----
    "Y":  "V_GLIDE_Y",

    # ==== Vowels ====

    # ---- Rounded vowels (lips round) ----
    "UW": "V_ROUNDED",
    "UH": "V_ROUNDED",
    "OW": "V_ROUNDED",
    "AO": "V_ROUNDED",
    "OY": "V_ROUNDED",

    # ---- Spread vowels (lips retracted / smile-like) ----
    "IY": "V_SPREAD",
    "IH": "V_SPREAD",
    "EY": "V_SPREAD",
    "EH": "V_SPREAD",

    # ---- Open / relaxed vowels (jaw drops, wide mouth) ----
    "AA": "V_OPEN",
    "AE": "V_OPEN",
    "AH": "V_OPEN",
    "AX": "V_OPEN",           # reduced schwa (rare in CMUDict, g2p_en may emit)

    # ---- Diphthongs with visible trajectory ----
    "AW": "V_OPEN_ROUND",     # open → rounded  (AA → UH)
    "AY": "V_OPEN_SPREAD",    # open → spread    (AA → IH)

    # ---- R-colored vowel (distinct lip + tongue shape) ----
    "ER": "V_RHOTIC",
}

VOWEL_PHONEMES: set[str] = {
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH",
    "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}

# Total unique viseme classes: 17
#   V_BILABIAL, V_LABIODENTAL, V_DENTAL, V_ALVEOLAR, V_ALVEOLAR_FRIC,
#   V_POSTALVEOLAR, V_VELAR, V_GLOTTAL, V_LATERAL, V_RHOTIC,
#   V_GLIDE_W, V_GLIDE_Y, V_ROUNDED, V_SPREAD, V_OPEN,
#   V_OPEN_ROUND, V_OPEN_SPREAD


def text_to_viseme_stream(text: str, vowel_mode: str = "grouped") -> str:
    """
    Convert free-form text -> ARPAbet phonemes -> token stream.

    Parameters
    ----------
    vowel_mode:
        - "grouped": map vowels to viseme categories in VISUAL_MAP
        - "exact": keep each vowel as its own token, e.g., VOW_AA, VOW_IH
    """
    if vowel_mode not in {"grouped", "exact"}:
        raise ValueError(f"Unsupported vowel_mode: {vowel_mode}")

    phonemes = _g2p(text)
    visemes: list[str] = []
    for p in phonemes:
        if not p or p.isspace() or not p[0].isalpha():
            continue
        # Strip stress digit: "AA1" → "AA"
        if p[-1].isdigit():
            p = p[:-1]

        if vowel_mode == "exact" and p in VOWEL_PHONEMES:
            visemes.append(f"VOW_{p}")
        else:
            visemes.append(VISUAL_MAP.get(p, "V_OPEN"))
    return " ".join(visemes)


def calculate_ver(ground_truth: str, hypothesis: str, vowel_mode: str = "grouped"):
    """
    Compute Viseme Error Rate.

    Returns
    -------
    (error_rate, ref_viseme_str, hyp_viseme_str)
    """
    ref = text_to_viseme_stream(ground_truth, vowel_mode=vowel_mode)
    hyp = text_to_viseme_stream(hypothesis, vowel_mode=vowel_mode)
    error_rate = wer(ref, hyp)
    return error_rate, ref, hyp


# ===================================================================
# CLI demo
# ===================================================================
if __name__ == "__main__":
    gt = (
        "the most angry event in my childhood is that my dad planned to take me "
        "to disneyland to have a fun time with him however on the day before he "
        "told me that because of overtime at work he can't go with me he promised "
        "me many many times they will take me for my birthday celebration however "
        "it didn't come true i was pretty upset and what makes me angry about it "
        "as this is not the only one time there's something happens"
    )
    hyp = (
        "because as you've seen in fucking show it's a big game and it's doubled "
        "in its size so it's diamonds and those are on the tables in the room "
        "because i know it's time to work i know it's time to work i've used to "
        "be very friendly times to see the tables of my friends in the room and "
        "those are technically damaged i was very upset i wouldn't be in reality "
        "of that and see not the only one time it seems to happen"
    )

    score, ref_visemes, hyp_visemes = calculate_ver(gt, hyp)

    print("=" * 72)
    print("  VER EVALUATION")
    print("=" * 72)
    print(f"\n  Viseme Error Rate (VER) : {score:.4f}")
    print(f"  Viseme Accuracy         : {(1.0 - score) * 100.0:.2f}%")
    print(f"\n  Ref visemes (first 90)  : {ref_visemes[:90]} ...")
    print(f"  Hyp visemes (first 90)  : {hyp_visemes[:90]} ...")
    print("=" * 72)