from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSR_DIR = PROJECT_ROOT / "ADFA_EVALUATION" / "Visual_Speech_Recognition_for_Multiple_Languages"
if str(VSR_DIR) not in sys.path:
    sys.path.insert(0, str(VSR_DIR))

import infer_pipeline


def test_load_decode_settings_from_ini(tmp_path: Path) -> None:
    config_path = tmp_path / "decode.ini"
    config_path.write_text(
        "\n".join(
            [
                "[decode]",
                "beam_size=80",
                "penalty=0.5",
                "ctc_weight=0.3",
                "lm_weight=0.0",
            ]
        ),
        encoding="utf-8",
    )

    settings = infer_pipeline.load_decode_settings(str(config_path))

    assert settings.beam_size == 80
    assert settings.penalty == 0.5
    assert settings.ctc_weight == 0.3
    assert settings.lm_weight == 0.0


def test_load_decode_settings_allows_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "decode.ini"
    config_path.write_text(
        "\n".join(
            [
                "[decode]",
                "beam_size=80",
                "penalty=0.5",
                "ctc_weight=0.3",
                "lm_weight=0.0",
            ]
        ),
        encoding="utf-8",
    )

    settings = infer_pipeline.load_decode_settings(
        str(config_path),
        beam_size=20,
        ctc_weight=0.5,
    )

    assert settings.beam_size == 20
    assert settings.penalty == 0.5
    assert settings.ctc_weight == 0.5
    assert settings.lm_weight == 0.0
