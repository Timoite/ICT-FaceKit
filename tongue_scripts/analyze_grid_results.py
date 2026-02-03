#!/usr/bin/env python3
"""
Analyze grid search results and identify best tongue configurations.
"""
import json
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.absolute()
TEST_OUTPUT_DIR = SCRIPT_DIR / "tongue_param_tests"

# Ground truth (first 15 seconds from TextGrid)
GROUND_TRUTH = "the most angry event in my childhood is that my dad planned to take me to disneyland to have a fun time with him however on the day before he told me that because of overtime at work he can't"

def simple_word_error_rate(hyp, ref):
    """
    Calculate simple word error rate.
    """
    hyp_words = hyp.lower().split()
    ref_words = ref.lower().split()
    
    # Count matches
    matches = 0
    for h_word in hyp_words:
        if h_word in ref_words:
            matches += 1
    
    # Calculate coverage
    if len(hyp_words) == 0:
        return 1.0
    
    coverage = matches / len(ref_words) if len(ref_words) > 0 else 0
    return 1.0 - coverage

def analyze_results():
    """
    Analyze all test results.
    """
    print("="*60)
    print("TONGUE PARAMETER GRID SEARCH ANALYSIS")
    print("="*60)
    
    # Load results
    results_file = TEST_OUTPUT_DIR / "all_results.json"
    if not results_file.exists():
        print(f"No results found at {results_file}")
        print("Run the grid search first!")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    successful = [r for r in results if "transcript" in r]
    failed = [r for r in results if "error" in r]
    
    print(f"\nTotal configurations tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    # Calculate WER for each
    print("\nCalculating Word Error Rates...")
    for result in successful:
        wer = simple_word_error_rate(result['transcript'], GROUND_TRUTH)
        result['wer'] = wer
        
        # Check for key phrases
        transcript_lower = result['transcript'].lower()
        result['has_disneyland'] = 'disneyland' in transcript_lower
        result['has_childhood'] = 'childhood' in transcript_lower
        result['has_angry'] = 'angry' in transcript_lower
    
    # Sort by WER
    successful_sorted = sorted(successful, key=lambda x: x['wer'])
    
    print("\n" + "="*60)
    print("TOP 10 BEST CONFIGURATIONS (by WER)")
    print("="*60)
    for i, result in enumerate(successful_sorted[:10], 1):
        print(f"\n{i}. {result['config_name']}")
        print(f"   WER: {result['wer']:.2%}")
        print(f"   Rotation: {result['rotation']}°")
        print(f"   Thickness: {result['thickness']}")
        print(f"   Std Scalar: {result['std_scalar']}")
        print(f"   Transcript: {result['transcript'][:100]}...")
        print(f"   Keywords: disneyland={result['has_disneyland']}, childhood={result['has_childhood']}, angry={result['has_angry']}")
    
    # Find configs with keywords
    print("\n" + "="*60)
    print("CONFIGURATIONS WITH KEYWORDS")
    print("="*60)
    
    has_disneyland = [r for r in successful if r['has_disneyland']]
    has_childhood = [r for r in successful if r['has_childhood']]
    has_angry = [r for r in successful if r['has_angry']]
    
    print(f"\nContains 'disneyland': {len(has_disneyland)} configs")
    for r in has_disneyland[:5]:
        print(f"  - {r['config_name']}: {r['transcript'][:80]}...")
    
    print(f"\nContains 'childhood': {len(has_childhood)} configs")
    for r in has_childhood[:5]:
        print(f"  - {r['config_name']}: {r['transcript'][:80]}...")
    
    print(f"\nContains 'angry': {len(has_angry)} configs")
    for r in has_angry[:5]:
        print(f"  - {r['config_name']}: {r['transcript'][:80]}...")
    
    # Parameter analysis
    print("\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    # Group by rotation
    rotation_groups = defaultdict(list)
    for r in successful:
        rotation_groups[r['rotation']].append(r['wer'])
    
    print("\nAverage WER by Rotation:")
    for rot in sorted(rotation_groups.keys()):
        avg_wer = sum(rotation_groups[rot]) / len(rotation_groups[rot])
        print(f"  {rot:2d}°: {avg_wer:.2%} (n={len(rotation_groups[rot])})")
    
    # Group by thickness
    thickness_groups = defaultdict(list)
    for r in successful:
        thickness_groups[r['thickness']].append(r['wer'])
    
    print("\nAverage WER by Thickness:")
    for thick in sorted(thickness_groups.keys()):
        avg_wer = sum(thickness_groups[thick]) / len(thickness_groups[thick])
        print(f"  {thick:.1f}: {avg_wer:.2%} (n={len(thickness_groups[thick])})")
    
    # Group by std_scalar
    scalar_groups = defaultdict(list)
    for r in successful:
        scalar_groups[r['std_scalar']].append(r['wer'])
    
    print("\nAverage WER by Std Scalar:")
    for scalar in sorted(scalar_groups.keys()):
        avg_wer = sum(scalar_groups[scalar]) / len(scalar_groups[scalar])
        print(f"  {scalar:.2f}: {avg_wer:.2%} (n={len(scalar_groups[scalar])})")
    
    # Save summary
    summary = {
        "ground_truth": GROUND_TRUTH,
        "total_configs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "top_10": successful_sorted[:10],
        "best_wer": successful_sorted[0] if successful_sorted else None,
        "parameter_analysis": {
            "by_rotation": {rot: sum(wers)/len(wers) for rot, wers in rotation_groups.items()},
            "by_thickness": {thick: sum(wers)/len(wers) for thick, wers in thickness_groups.items()},
            "by_std_scalar": {scalar: sum(wers)/len(wers) for scalar, wers in scalar_groups.items()}
        }
    }
    
    summary_file = TEST_OUTPUT_DIR / "analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"\nFull results: {results_file}")
    print(f"All videos: {TEST_OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_results()
