from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evaluation_script.phone_confusion import (
    DELETE_TOKEN,
    INSERT_TOKEN,
    KNOWN_PHONES,
    align_phones,
    assert_known_phone_category_coverage,
    evaluate_phone_confusion,
    parse_textgrid_phones,
    stats_from_alignment,
    text_to_phone_stream,
)
from tongue_scripts.evaluate_phoneme_confusion import discover_pairs


TEXTGRID_FIXTURE = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 1
            intervals [1]:
                xmin = 0
                xmax = 1
                text = "thin tea"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 7
            intervals [1]:
                xmin = 0.0
                xmax = 0.1
                text = ""
            intervals [2]:
                xmin = 0.1
                xmax = 0.2
                text = "TH"
            intervals [3]:
                xmin = 0.2
                xmax = 0.3
                text = "IH1"
            intervals [4]:
                xmin = 0.3
                xmax = 0.4
                text = "N"
            intervals [5]:
                xmin = 0.4
                xmax = 0.5
                text = "spn"
            intervals [6]:
                xmin = 0.5
                xmax = 0.6
                text = "T"
            intervals [7]:
                xmin = 0.6
                xmax = 0.7
                text = "IY2"
"""


class PhoneConfusionTests(unittest.TestCase):
    def test_parse_textgrid_phones_strips_stress_and_excludes_spn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.TextGrid"
            path.write_text(TEXTGRID_FIXTURE, encoding="utf-8")

            phones = [interval.text for interval in parse_textgrid_phones(path)]

        self.assertEqual(phones, ["TH", "IH", "N", "T", "IY"])

    def test_text_to_phone_stream_uses_normalized_arpabet(self) -> None:
        phones = text_to_phone_stream("thin tea")

        self.assertIn("TH", phones)
        self.assertIn("T", phones)
        self.assertTrue(all(phone in KNOWN_PHONES for phone in phones))
        self.assertTrue(all(not phone[-1:].isdigit() for phone in phones))

    def test_alignment_counts_match_sub_delete_insert(self) -> None:
        alignment = align_phones(["T", "D", "K"], ["T", "G", "S", "K"])
        stats = stats_from_alignment(alignment, ref_count=3, hyp_count=4)

        self.assertEqual(stats.correct, 2)
        self.assertEqual(stats.substitutions, 1)
        self.assertEqual(stats.insertions, 1)
        self.assertEqual(stats.deletions, 0)
        substituted_from_d = sum(
            count
            for (ref, hyp), count in stats.confusion.items()
            if ref == "D" and hyp != DELETE_TOKEN
        )
        insertions = sum(
            count
            for (ref, _hyp), count in stats.confusion.items()
            if ref == INSERT_TOKEN
        )
        self.assertEqual(substituted_from_d, 1)
        self.assertEqual(insertions, 1)

    def test_alignment_counts_deletion(self) -> None:
        alignment = align_phones(["T", "D", "K"], ["T", "K"])
        stats = stats_from_alignment(alignment, ref_count=3, hyp_count=2)

        self.assertEqual(stats.correct, 2)
        self.assertEqual(stats.deletions, 1)
        self.assertEqual(stats.confusion[("D", DELETE_TOKEN)], 1)

    def test_evaluate_phone_confusion_recall(self) -> None:
        result = evaluate_phone_confusion(["TH", "IH", "N"], "thin")

        self.assertEqual(result.stats.ref_count, 3)
        self.assertGreaterEqual(result.stats.correct, 2)
        self.assertGreater(result.stats.tongue_sensitive_recall(), 0.0)

    def test_every_known_phone_has_analysis_category(self) -> None:
        assert_known_phone_category_coverage()

    def test_discover_pairs_matches_textgrid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_root = root / "videos"
            tg_root = root / "textgrids"
            video_root.mkdir()
            (tg_root / "1").mkdir(parents=True)
            (video_root / "1_demo_0_1_1_with_tongue_with_audio.mp4").touch()
            (video_root / "1_demo_0_1_1_passive_tongue_with_audio.mp4").touch()
            (tg_root / "1" / "1_demo_0_1_1.TextGrid").write_text(TEXTGRID_FIXTURE, encoding="utf-8")

            pairs = discover_pairs([video_root], [tg_root], dataset_filter=None)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].dataset_id, "1_demo_0_1_1")
        self.assertIsNotNone(pairs[0].textgrid_path)


if __name__ == "__main__":
    unittest.main()
