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
# Visual grouping table agreed for this project:
# - Bilabial Plosive: P, M, B
# - Labiodentals: F, V
# - Dental Fricative: TH, DH
# - Alveolar Plosive: T, D, N
# - Alveolar Fricatives: S, Z
# - Post-Alveolar Fricative: SH, ZH
# - Velar Plosive: K, G
VISUAL_MAP = {
    # Bilabial Plosive
    'P': 'V_BILABIAL_PLOSIVE', 'M': 'V_BILABIAL_PLOSIVE', 'B': 'V_BILABIAL_PLOSIVE',
    # Labiodentals
    'F': 'V_LABIODENTAL', 'V': 'V_LABIODENTAL',
    # Dental Fricative
    'TH': 'V_DENTAL_FRICATIVE', 'DH': 'V_DENTAL_FRICATIVE',
    # Alveolar Plosive
    'T': 'V_ALVEOLAR_PLOSIVE', 'D': 'V_ALVEOLAR_PLOSIVE', 'N': 'V_ALVEOLAR_PLOSIVE',
    # Alveolar Fricatives
    'S': 'V_ALVEOLAR_FRICATIVE', 'Z': 'V_ALVEOLAR_FRICATIVE',
    # Post-Alveolar Fricative
    'SH': 'V_POST_ALVEOLAR_FRICATIVE', 'ZH': 'V_POST_ALVEOLAR_FRICATIVE',
    # Velar Plosive
    'K': 'V_VELAR_PLOSIVE', 'G': 'V_VELAR_PLOSIVE',
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

# Data
gt = "the most angry event in my childhood is that my dad planned to take me to disneyland to have a fun time with him however on the day before he told me that because of overtime at work he can't go with me he promised me many many times they will take me for my birthday celebration however it didn't come true i was pretty upset and what makes me angry about it as this is not the only one time there's something happens"
inf = "because as you've seen in fucking show it's a big game and it's doubled in its size so it's diamonds and those are on the tables in the room because i know it's time to work i know it's time to work i've used to be very friendly times to see the tables of my friends in the room and those are technically damaged i was very upset i wouldn't be in reality of that and see not the only one time it seems to happen"

score, ref_v, hyp_v = calculate_ver(gt, inf)

print(f"Viseme Error Rate: {score:.2f}")
print(f"\nGround Truth Visemes: {ref_v[:50]}...")
print(f"Inference Visemes:    {hyp_v[:50]}...")