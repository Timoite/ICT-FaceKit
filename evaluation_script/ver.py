import g2p_en
import nltk
from jiwer import wer

# 1. Initialize the Grapheme-to-Phoneme converter
def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)

ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
ensure_nltk_resource("corpora/cmudict", "cmudict")

g2p = g2p_en.G2p()

# 2. Define a Phoneme -> Viseme Map
# This uses a standard "Visual Grouping" (similar to Oculus/MPEG-4)
# We group sounds that LOOK the same into the same "Viseme ID"
VISUAL_MAP = {
    # Bilabials (Lips touching) - P, B, M look identical
    'P': 'V_BILABIAL', 'B': 'V_BILABIAL', 'M': 'V_BILABIAL',
    
    # Labiodentals (Top teeth on bottom lip) - F, V
    'F': 'V_LABIODENTAL', 'V': 'V_LABIODENTAL',
    
    # Dental/Alveolar (Tongue behind teeth) - T, D, S, Z, TH, DH
    'T': 'V_DENTAL', 'D': 'V_DENTAL', 'S': 'V_DENTAL', 'Z': 'V_DENTAL',
    'TH': 'V_DENTAL', 'DH': 'V_DENTAL', 'N': 'V_DENTAL', 'L': 'V_DENTAL',
    
    # Rounding (Lips forward) - W, R, UW, OW
    'W': 'V_ROUND', 'R': 'V_ROUND', 'UW': 'V_ROUND', 'OW': 'V_ROUND', 'UH': 'V_ROUND',
    
    # Wide/Stretch (Lips pulled back) - IY, IH, AE, EH, AY
    'IY': 'V_WIDE', 'IH': 'V_WIDE', 'AE': 'V_WIDE', 'EH': 'V_WIDE', 'AY': 'V_WIDE',
    
    # Open (Jaw drop) - AA, AO, AH
    'AA': 'V_OPEN', 'AO': 'V_OPEN', 'AH': 'V_OPEN',
    
    # Silence/Neutral
    ' ': 'V_SIL', 'SIL': 'V_SIL'
}

def text_to_viseme_stream(text):
    # Convert text to phonemes
    phonemes = g2p(text)
    
    # Remove stress markers (numbers) from phonemes (e.g., "AA1" -> "AA")
    clean_phonemes = [p[:-1] if p[-1].isdigit() else p for p in phonemes]
    
    # Map to Visemes (Default to 'V_OPEN' if unknown to avoid crashes)
    visemes = [VISUAL_MAP.get(p, 'V_OPEN') for p in clean_phonemes if p in VISUAL_MAP]
    
    # Join into a "sentence" of visemes for JiWER to process
    return " ".join(visemes)

def calculate_ver(ground_truth, hypothesis):
    # Convert both texts to "Viseme Sentences"
    ref_visemes = text_to_viseme_stream(ground_truth)
    hyp_visemes = text_to_viseme_stream(hypothesis)
    
    # Calculate Error Rate on the Viseme strings
    error_rate = wer(ref_visemes, hyp_visemes)
    
    return error_rate, ref_visemes, hyp_visemes

# --- YOUR DATA ---
gt = "the most angry event in my childhood is that my dad planned to take me to disneyland"
inf = "because as you've seen in fucking show it's a big game and it's doubled in its size so it's diamonds"

score, ref_v, hyp_v = calculate_ver(gt, inf)

print(f"Viseme Error Rate: {score:.2f}")
print(f"\nGround Truth Visemes: {ref_v[:50]}...")
print(f"Inference Visemes:    {hyp_v[:50]}...")