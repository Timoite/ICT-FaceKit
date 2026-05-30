## Run: vocaset_sentence01_passive_vs_std0p27_z0p00_th1p40_ctc0p6_fresh | 2026-05-27 19:35:23

### Settings
- dataset id: `vocaset_FaceTalk_170725_00137_TA_sentence01`
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `full`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/mifs/ht467/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`
- ground truth source: `/home/mifs/ht467/ICT-FaceKit/tests/vocaset_outputs/ground_truth_FaceTalk_170725_00137_TA_sentence01.txt`
- report mode: `write`

### Experiment Metadata
- hypothesis: active tongue should outperform passive tongue on same-speaker Wayne data
- video: `/home/mifs/ht467/ICT-FaceKit/tests/vocaset_outputs/passive/vocaset_FaceTalk_170725_00137_TA_sentence01_passive_tongue_with_audio.mp4`
- video: `/home/mifs/ht467/ICT-FaceKit/tests/vocaset_outputs/grid_search/videos/vocaset_FaceTalk_170725_00137_TA_sentence01_std0p27_z0p00_th1p40_active_tongue_with_audio.mp4`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| vocaset_FaceTalk_170725_00137_TA_sentence01_passive_tongue_with_audio.mp4 | 0.6176 | 1.0000 | 1.0000 | 0.8088 | 38.24% | 0.00% | 9 |
| vocaset_FaceTalk_170725_00137_TA_sentence01_std0p27_z0p00_th1p40_active_tongue_with_audio.mp4 | 0.6176 | 1.0000 | 1.0000 | 0.8088 | 38.24% | 0.00% | 11 |

- Best (by composite): **vocaset_FaceTalk_170725_00137_TA_sentence01_passive_tongue_with_audio.mp4** (VER=0.6176, WER_norm=1.0000, Composite=0.8088)
- Worst (by composite): **vocaset_FaceTalk_170725_00137_TA_sentence01_std0p27_z0p00_th1p40_active_tongue_with_audio.mp4** (VER=0.6176, WER_norm=1.0000, Composite=0.8088)
- VER gap (worst - best): **0.0000**
- WER gap (worst - best): **0.0000**
- Composite gap (worst - best): **0.0000**

### Ground Truth
she had your dark suit in greasy wash water all year.

### Hypotheses
#### vocaset_FaceTalk_170725_00137_TA_sentence01_passive_tongue_with_audio.mp4
- VER: 0.6176
- WER(norm): 1.0000
- WER(raw): 1.0000
- Composite Index: 0.8088
- Viseme Accuracy: 38.24%
- Word Accuracy(norm): 0.00%
- HYP: I JUST STARTED TO INCREASE THE POSH PARK R

#### vocaset_FaceTalk_170725_00137_TA_sentence01_std0p27_z0p00_th1p40_active_tongue_with_audio.mp4
- VER: 0.6176
- WER(norm): 1.0000
- WER(raw): 1.0000
- Composite Index: 0.8088
- Viseme Accuracy: 38.24%
- Word Accuracy(norm): 0.00%
- HYP: SO HOW DO YOU START TO INCREASE THE BRUSH PRODUCT ALREADY

---
