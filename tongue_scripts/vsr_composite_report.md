## Run: active_vs_passive_current_params_v2 | 2026-03-05 11:10:46

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER | Composite (0.5*VER + 0.5*WER) | Viseme Accuracy | Word Accuracy | HYP words |
|---|---:|---:|---:|---:|---:|---:|
| 1_wayne_0_75_75_with_tongue_with_audio.mp4 | 0.6325 | 0.9048 | 0.7686 | 36.75% | 9.52% | 74 |
| 1_wayne_0_75_75_passive_tongue_with_audio.mp4 | 0.6749 | 0.9048 | 0.7898 | 32.51% | 9.52% | 73 |

- Best (by composite): **1_wayne_0_75_75_with_tongue_with_audio.mp4** (VER=0.6325, WER=0.9048, Composite=0.7686)
- Worst (by composite): **1_wayne_0_75_75_passive_tongue_with_audio.mp4** (VER=0.6749, WER=0.9048, Composite=0.7898)
- VER gap (worst - best): **0.0424**
- WER gap (worst - best): **0.0000**
- Composite gap (worst - best): **0.0212**

### Ground Truth
the most angry event in my childhood is that my dad planned to take me to disneyland to have a fun time with him however on the day before he told me that because of overtime at work he can't go with me he promised me many many times they will take me for my birthday celebration however it didn't come true i was pretty upset and what makes me angry about it as this is not the only one time there's something happens

### Hypotheses
#### 1_wayne_0_75_75_with_tongue_with_audio.mp4
- VER: 0.6325
- WER: 0.9048
- Composite Index: 0.7686
- Viseme Accuracy: 36.75%
- Word Accuracy: 9.52%
- HYP: BECAUSE IT IS DIFFICULT FOR YOU TO FIND ANY PLACE IN THE PAST THAT IS IN THE PAST AND THOSE ARE THE DIAMONDS THOSE ARE THE ONLY DIAMONDS IN THE PRESENT TENSE OF THE WORLD THEY ARE DIAMONDS AND IF YOU WANT TO FIND THEM MANY TIMES IT WILL TAKE YOU TO THE PRESENT DAY AND THOSE ARE IN THE PRESENT DAY OR THE PRESENT DAY AND YOU WILL FIND THAT SOMETHING THAT HAPPENS

#### 1_wayne_0_75_75_passive_tongue_with_audio.mp4
- VER: 0.6749
- WER: 0.9048
- Composite Index: 0.7898
- Viseme Accuracy: 32.51%
- Word Accuracy: 9.52%
- HYP: IT'S BEEN TEN YEARS SINCE THEN BUT IT'S SHOWN TO BE SAFE AND STABLE IN EACH AND SANCTIONED ASSIGNMENT I'M NOT ON A TABLET OR SO THE COMPUTER KNOWS WHAT TYPE OF WORK I'LL USE IT FOR I'VE USED IT FOR A FEW TIMES SINCE I'VE USED IT SINCE THE EXAMS I'VE USED IT EVERY DAY I WOULD PLEASE BE OUT OF THAT AND CONSIDER DOING THE ONLY WORK ASSIGNMENT SEEMS TO HAPPEN

---
## Run: active_vs_passive_case_punct_normalized_wer | 2026-03-05 11:15:34

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_wayne_0_75_75_with_tongue_with_audio.mp4 | 0.6325 | 0.9080 | 0.9048 | 0.7703 | 36.75% | 9.20% | 74 |
| 1_wayne_0_75_75_passive_tongue_with_audio.mp4 | 0.6749 | 0.9310 | 0.9048 | 0.8030 | 32.51% | 6.90% | 73 |

- Best (by composite): **1_wayne_0_75_75_with_tongue_with_audio.mp4** (VER=0.6325, WER_norm=0.9080, Composite=0.7703)
- Worst (by composite): **1_wayne_0_75_75_passive_tongue_with_audio.mp4** (VER=0.6749, WER_norm=0.9310, Composite=0.8030)
- VER gap (worst - best): **0.0424**
- WER gap (worst - best): **0.0230**
- Composite gap (worst - best): **0.0327**

### Ground Truth
the most angry event in my childhood is that my dad planned to take me to disneyland to have a fun time with him however on the day before he told me that because of overtime at work he can't go with me he promised me many many times they will take me for my birthday celebration however it didn't come true i was pretty upset and what makes me angry about it as this is not the only one time there's something happens

### Hypotheses
#### 1_wayne_0_75_75_with_tongue_with_audio.mp4
- VER: 0.6325
- WER(norm): 0.9080
- WER(raw): 0.9048
- Composite Index: 0.7703
- Viseme Accuracy: 36.75%
- Word Accuracy(norm): 9.20%
- HYP: BECAUSE IT IS DIFFICULT FOR YOU TO FIND ANY PLACE IN THE PAST THAT IS IN THE PAST AND THOSE ARE THE DIAMONDS THOSE ARE THE ONLY DIAMONDS IN THE PRESENT TENSE OF THE WORLD THEY ARE DIAMONDS AND IF YOU WANT TO FIND THEM MANY TIMES IT WILL TAKE YOU TO THE PRESENT DAY AND THOSE ARE IN THE PRESENT DAY OR THE PRESENT DAY AND YOU WILL FIND THAT SOMETHING THAT HAPPENS

#### 1_wayne_0_75_75_passive_tongue_with_audio.mp4
- VER: 0.6749
- WER(norm): 0.9310
- WER(raw): 0.9048
- Composite Index: 0.8030
- Viseme Accuracy: 32.51%
- Word Accuracy(norm): 6.90%
- HYP: IT'S BEEN TEN YEARS SINCE THEN BUT IT'S SHOWN TO BE SAFE AND STABLE IN EACH AND SANCTIONED ASSIGNMENT I'M NOT ON A TABLET OR SO THE COMPUTER KNOWS WHAT TYPE OF WORK I'LL USE IT FOR I'VE USED IT FOR A FEW TIMES SINCE I'VE USED IT SINCE THE EXAMS I'VE USED IT EVERY DAY I WOULD PLEASE BE OUT OF THAT AND CONSIDER DOING THE ONLY WORK ASSIGNMENT SEEMS TO HAPPEN

---
## Run: active_vs_passive_vowel_exact_mode | 2026-03-05 11:31:14

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `exact`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1_wayne_0_75_75_with_tongue_with_audio.mp4 | 0.6890 | 0.9080 | 0.9048 | 0.7985 | 31.10% | 9.20% | 74 |
| 1_wayne_0_75_75_passive_tongue_with_audio.mp4 | 0.7138 | 0.9310 | 0.9048 | 0.8224 | 28.62% | 6.90% | 73 |

- Best (by composite): **1_wayne_0_75_75_with_tongue_with_audio.mp4** (VER=0.6890, WER_norm=0.9080, Composite=0.7985)
- Worst (by composite): **1_wayne_0_75_75_passive_tongue_with_audio.mp4** (VER=0.7138, WER_norm=0.9310, Composite=0.8224)
- VER gap (worst - best): **0.0247**
- WER gap (worst - best): **0.0230**
- Composite gap (worst - best): **0.0239**

### Ground Truth
the most angry event in my childhood is that my dad planned to take me to disneyland to have a fun time with him however on the day before he told me that because of overtime at work he can't go with me he promised me many many times they will take me for my birthday celebration however it didn't come true i was pretty upset and what makes me angry about it as this is not the only one time there's something happens

### Hypotheses
#### 1_wayne_0_75_75_with_tongue_with_audio.mp4
- VER: 0.6890
- WER(norm): 0.9080
- WER(raw): 0.9048
- Composite Index: 0.7985
- Viseme Accuracy: 31.10%
- Word Accuracy(norm): 9.20%
- HYP: BECAUSE IT IS DIFFICULT FOR YOU TO FIND ANY PLACE IN THE PAST THAT IS IN THE PAST AND THOSE ARE THE DIAMONDS THOSE ARE THE ONLY DIAMONDS IN THE PRESENT TENSE OF THE WORLD THEY ARE DIAMONDS AND IF YOU WANT TO FIND THEM MANY TIMES IT WILL TAKE YOU TO THE PRESENT DAY AND THOSE ARE IN THE PRESENT DAY OR THE PRESENT DAY AND YOU WILL FIND THAT SOMETHING THAT HAPPENS

#### 1_wayne_0_75_75_passive_tongue_with_audio.mp4
- VER: 0.7138
- WER(norm): 0.9310
- WER(raw): 0.9048
- Composite Index: 0.8224
- Viseme Accuracy: 28.62%
- Word Accuracy(norm): 6.90%
- HYP: IT'S BEEN TEN YEARS SINCE THEN BUT IT'S SHOWN TO BE SAFE AND STABLE IN EACH AND SANCTIONED ASSIGNMENT I'M NOT ON A TABLET OR SO THE COMPUTER KNOWS WHAT TYPE OF WORK I'LL USE IT FOR I'VE USED IT FOR A FEW TIMES SINCE I'VE USED IT SINCE THE EXAMS I'VE USED IT EVERY DAY I WOULD PLEASE BE OUT OF THAT AND CONSIDER DOING THE ONLY WORK ASSIGNMENT SEEMS TO HAPPEN

---
## Run: multi_speaker_2_2_scott_0_11_11_48.0s_grouped | 2026-03-05 12:07:14

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2_scott_0_11_11_with_tongue_with_audio.mp4 | 0.7592 | 0.8750 | 0.9130 | 0.8171 | 24.08% | 12.50% | 70 |
| 2_scott_0_11_11_passive_tongue_with_audio.mp4 | 0.7397 | 0.9444 | 0.9493 | 0.8421 | 26.03% | 5.56% | 51 |

- Best (by composite): **2_scott_0_11_11_with_tongue_with_audio.mp4** (VER=0.7592, WER_norm=0.8750, Composite=0.8171)
- Worst (by composite): **2_scott_0_11_11_passive_tongue_with_audio.mp4** (VER=0.7397, WER_norm=0.9444, Composite=0.8421)
- VER gap (worst - best): **-0.0195**
- WER gap (worst - best): **0.0694**
- Composite gap (worst - best): **0.0250**

### Ground Truth
i've got a room in my house or season i got a room in my house in my study room i have a lot of books in my study room such as fashion magazines inspirational books etc also there is a computer in my study in my study room usually i will check my computer to learn some study related information such as math problems that just can't do i also am able to use it just for fun one example i'm able to catch up on my on the latest netflix episodes of my favorite show and sometimes as i'm able to use it to check out some clothes and look inside the latest fashion magazines however most of them most of the time i just can't add it to my cart so i don't buy them

### Hypotheses
#### 2_scott_0_11_11_with_tongue_with_audio.mp4
- VER: 0.7592
- WER(norm): 0.8750
- WER(raw): 0.9130
- Composite Index: 0.8171
- Viseme Accuracy: 24.08%
- Word Accuracy(norm): 12.50%
- HYP: I DECIDED TO GIVE IT A TRY I CAN'T TELL YOU THE SECRET I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M GOING TO TELL YOU THE TRUTH I'M

#### 2_scott_0_11_11_passive_tongue_with_audio.mp4
- VER: 0.7397
- WER(norm): 0.9444
- WER(raw): 0.9493
- Composite Index: 0.8421
- Viseme Accuracy: 26.03%
- Word Accuracy(norm): 5.56%
- HYP: AS I SHOWED YOU IN MY COURSES TODAY I'M GOING TO TEACH YOU TODAY'S VIDEOS AS WELL AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AND VIDEOS AS WELL AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS AS VIDEOS

---
## Run: multi_speaker_4_4_lawrence_0_23_23_50.0s_grouped | 2026-03-05 12:11:45

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4_lawrence_0_23_23_with_tongue_with_audio.mp4 | 0.7530 | 0.9420 | 0.9489 | 0.8475 | 24.70% | 5.80% | 53 |
| 4_lawrence_0_23_23_passive_tongue_with_audio.mp4 | 0.7264 | 0.9710 | 0.9708 | 0.8487 | 27.36% | 2.90% | 72 |

- Best (by composite): **4_lawrence_0_23_23_with_tongue_with_audio.mp4** (VER=0.7530, WER_norm=0.9420, Composite=0.8475)
- Worst (by composite): **4_lawrence_0_23_23_passive_tongue_with_audio.mp4** (VER=0.7264, WER_norm=0.9710, Composite=0.8487)
- VER gap (worst - best): **-0.0266**
- WER gap (worst - best): **0.0290**
- Composite gap (worst - best): **0.0012**

### Ground Truth
can remember the first time i tucker yucky or iraq dumplings as we would call them in america i was pretty new to japan at the time and i was walking through a major shopping area near number station in osaka there were a lot of takoyaki stands and a ton of signs showing how good they were so i figured why not give them a shot now if you are not familiar with the food you would probably do exactly what i did the second i got them i put one in my mouth and i eli's i had made a mistake as it felt like a hot coal was in my mouth and i was a man with a burning mouth without a drink in my hand so i did what anyone would do and

### Hypotheses
#### 4_lawrence_0_23_23_with_tongue_with_audio.mp4
- VER: 0.7530
- WER(norm): 0.9420
- WER(raw): 0.9489
- Composite Index: 0.8475
- Viseme Accuracy: 24.70%
- Word Accuracy(norm): 5.80%
- HYP: AND LIKEWISE THERE ARE MANY DIFFERENT WAYS TO DO THIS YOU CAN DO THIS IF YOU WANT BUT LIKEWISE THERE'S OTHER WAYS TO DO IT LIKEWISE THERE'S OTHER WAYS TO DO IT LIKEWISE THERE'S OTHER WAYS TO DO IT LIKEWISE THERE'S OTHER WAYS TO DO IT LIKEWISE THERE'S OTHER WAYS TO DO THIS

#### 4_lawrence_0_23_23_passive_tongue_with_audio.mp4
- VER: 0.7264
- WER(norm): 0.9710
- WER(raw): 0.9708
- Composite Index: 0.8487
- Viseme Accuracy: 27.36%
- Word Accuracy(norm): 2.90%
- HYP: AND YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT WHEN YOU WANT TO USE IT

---
## Run: multi_speaker_16_16_jorge_0_1_1_52.0s_grouped | 2026-03-05 12:17:00

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16_jorge_0_1_1_passive_tongue_with_audio.mp4 | 0.7418 | 0.9082 | 0.9096 | 0.8250 | 25.82% | 9.18% | 86 |
| 16_jorge_0_1_1_with_tongue_with_audio.mp4 | 0.8056 | 0.9082 | 0.9149 | 0.8569 | 19.44% | 9.18% | 50 |

- Best (by composite): **16_jorge_0_1_1_passive_tongue_with_audio.mp4** (VER=0.7418, WER_norm=0.9082, Composite=0.8250)
- Worst (by composite): **16_jorge_0_1_1_with_tongue_with_audio.mp4** (VER=0.8056, WER_norm=0.9082, Composite=0.8569)
- VER gap (worst - best): **0.0638**
- WER gap (worst - best): **0.0000**
- Composite gap (worst - best): **0.0319**

### Ground Truth
the first thing i like to do on weekends is relaxing and i'll go shopping if i'm not that tired since i started my job i think it's very important to get a good sleep during the weekend because when you have to work monday through friday the whole week you're very tired so getting a good rest is as important as completing an excellent job in my spare time if i feel okay i will go for a walk or hike in nature sometimes i try to organize something for my friends volunteer at the buddhist temple on the weekend or i can just walk around enjoying the sunshine i right to live a healthy lifestyle considering how much time i spend sitting at work i always try to move as much as i can while i'm not working and some other days when i'm when i'm free i like to listen to music or watch the commentary movies on my laptop but sometimes i'll just sleep i especially liked watching japanese anime i think watching anime is helpful for me to learn and express japanese but

### Hypotheses
#### 16_jorge_0_1_1_passive_tongue_with_audio.mp4
- VER: 0.7418
- WER(norm): 0.9082
- WER(raw): 0.9096
- Composite Index: 0.8250
- Viseme Accuracy: 25.82%
- Word Accuracy(norm): 9.18%
- HYP: THE NEXT THING I WANT TO SHOW YOU IS THE SIMPLEST WAY TO DO IT THE SIMPLEST WAY TO DO IT IS TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO DO IT IS

#### 16_jorge_0_1_1_with_tongue_with_audio.mp4
- VER: 0.8056
- WER(norm): 0.9082
- WER(raw): 0.9149
- Composite Index: 0.8569
- Viseme Accuracy: 19.44%
- Word Accuracy(norm): 9.18%
- HYP: IN GENERAL I CURATE I CURATE I CURATE MY NEPHEW AND I CURATE I CURATE MY NEPHEW AND I CURATE I CURATE MY NEPHEW AND I CURATE I CURATE MY NEPHEW AND I CURATE I CURATE MY NEPHEW AND I CURATE I CURATE MY NEPHEW AND I CURATE MY NEPHEW

---
## Run: multi_speaker_5_5_stewart_0_10_10_53.0s_grouped | 2026-03-05 12:21:56

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5_stewart_0_10_10_passive_tongue_with_audio.mp4 | 0.6900 | 0.9080 | 0.9064 | 0.7990 | 31.00% | 9.20% | 107 |
| 5_stewart_0_10_10_with_tongue_with_audio.mp4 | 0.7419 | 0.9368 | 0.9357 | 0.8394 | 25.81% | 6.32% | 38 |

- Best (by composite): **5_stewart_0_10_10_passive_tongue_with_audio.mp4** (VER=0.6900, WER_norm=0.9080, Composite=0.7990)
- Worst (by composite): **5_stewart_0_10_10_with_tongue_with_audio.mp4** (VER=0.7419, WER_norm=0.9368, Composite=0.8394)
- VER gap (worst - best): **0.0520**
- WER gap (worst - best): **0.0287**
- Composite gap (worst - best): **0.0404**

### Ground Truth
i would prefer to choose a major that is easy to find a good job in the future like finance or marketing for example there's no one that can deny the most common reason for attending university is to get prepared for a good job in the future so whether the major will lead us to a good job on not is the most important reason why we choose our major and if we find a good job with a decent payment we can use the money that we have learned from it to by our own interest for example i like painting a lot however i choose painting of my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance-related job after graduation after university i can get the high salary and in my free time i can use my salary to hire a professional teacher to teach me how to draw

### Hypotheses
#### 5_stewart_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.6900
- WER(norm): 0.9080
- WER(raw): 0.9064
- Composite Index: 0.7990
- Viseme Accuracy: 31.00%
- Word Accuracy(norm): 9.20%
- HYP: OUR APPROACH TO DISSERTATION IS TO DISSERT THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW

#### 5_stewart_0_10_10_with_tongue_with_audio.mp4
- VER: 0.7419
- WER(norm): 0.9368
- WER(raw): 0.9357
- Composite Index: 0.8394
- Viseme Accuracy: 25.81%
- Word Accuracy(norm): 6.32%
- HYP: UNIVERSITY OF THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS

---
## Run: multi_speaker_7_7_sophie_0_10_10_54.0s_grouped | 2026-03-05 12:26:37

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7_sophie_0_10_10_passive_tongue_with_audio.mp4 | 0.7276 | 0.9222 | 0.9209 | 0.8249 | 27.24% | 7.78% | 69 |
| 7_sophie_0_10_10_with_tongue_with_audio.mp4 | 0.7724 | 0.9556 | 0.9605 | 0.8640 | 22.76% | 4.44% | 55 |

- Best (by composite): **7_sophie_0_10_10_passive_tongue_with_audio.mp4** (VER=0.7276, WER_norm=0.9222, Composite=0.8249)
- Worst (by composite): **7_sophie_0_10_10_with_tongue_with_audio.mp4** (VER=0.7724, WER_norm=0.9556, Composite=0.8640)
- VER gap (worst - best): **0.0448**
- WER gap (worst - best): **0.0333**
- Composite gap (worst - best): **0.0391**

### Ground Truth
i would prefer to choose a major that is easy for me to find a good job in the future like finance or marketing for example there's no there's no one that can deny that the most common reason for attending universities to get prepared for a good job in the future so whether the major would lead us to a good job or not is the most important reason why we choose our major if we find a good job with a decent payment we can choose the money that we have and from it from from it to satisfy our own interest for example i like painting a lot so however i choose to painting in my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance related job after graduation after university can get high salary and in my free time i can use my salary to hire professional teacher to teach me how to draw

### Hypotheses
#### 7_sophie_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.7276
- WER(norm): 0.9222
- WER(raw): 0.9209
- Composite Index: 0.8249
- Viseme Accuracy: 27.24%
- Word Accuracy(norm): 7.78%
- HYP: AS YOU CAN SEE I HAVE THREE DIFFERENT WAYS OF DOING IT ONE IS USING DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING DIFFERENT WAYS

#### 7_sophie_0_10_10_with_tongue_with_audio.mp4
- VER: 0.7724
- WER(norm): 0.9556
- WER(raw): 0.9605
- Composite Index: 0.8640
- Viseme Accuracy: 22.76%
- Word Accuracy(norm): 4.44%
- HYP: I DON'T KNOW IF YOU CAN SEE IT ON THE SCREEN BUT IF YOU CAN SEE IT ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON THE SCREEN IT'S ON SCREEN

---
## Run: multi_speaker_2_2_scott_0_11_11_48.0s_grouped | 2026-03-05 12:38:41

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2_scott_0_11_11_with_tongue_with_audio.mp4 | 0.7137 | 0.8750 | 0.8913 | 0.7943 | 28.63% | 12.50% | 54 |
| 2_scott_0_11_11_passive_tongue_with_audio.mp4 | 0.7115 | 0.9236 | 0.9275 | 0.8176 | 28.85% | 7.64% | 113 |

- Best (by composite): **2_scott_0_11_11_with_tongue_with_audio.mp4** (VER=0.7137, WER_norm=0.8750, Composite=0.7943)
- Worst (by composite): **2_scott_0_11_11_passive_tongue_with_audio.mp4** (VER=0.7115, WER_norm=0.9236, Composite=0.8176)
- VER gap (worst - best): **-0.0022**
- WER gap (worst - best): **0.0486**
- Composite gap (worst - best): **0.0232**

### Ground Truth
i've got a room in my house or season i got a room in my house in my study room i have a lot of books in my study room such as fashion magazines inspirational books etc also there is a computer in my study in my study room usually i will check my computer to learn some study related information such as math problems that just can't do i also am able to use it just for fun one example i'm able to catch up on my on the latest netflix episodes of my favorite show and sometimes as i'm able to use it to check out some clothes and look inside the latest fashion magazines however most of them most of the time i just can't add it to my cart so i don't buy them

### Hypotheses
#### 2_scott_0_11_11_with_tongue_with_audio.mp4
- VER: 0.7137
- WER(norm): 0.8750
- WER(raw): 0.8913
- Composite Index: 0.7943
- Viseme Accuracy: 28.63%
- Word Accuracy(norm): 12.50%
- HYP: I DECIDED TO USE MY VIDEOS I DECIDED TO USE MY VIDEOS AND VIDEOS BECAUSE I WANTED TO VIDEOS AND VIDEOS BECAUSE I WANTED TO VIDEOS AND VIDEOS BECAUSE I WANTED TO VIDEOS AND VIDEOS BECAUSE I WANTED TO SHOWCASE MY VIDEOS AND MY VIDEOS AND MY VIDEOS SO I DECIDED TO USE THEM

#### 2_scott_0_11_11_passive_tongue_with_audio.mp4
- VER: 0.7115
- WER(norm): 0.9236
- WER(raw): 0.9275
- Composite Index: 0.8176
- Viseme Accuracy: 28.85%
- Word Accuracy(norm): 7.64%
- HYP: I DECIDED TO USE MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D MY D'S D MY D MY D MY D MY D MY D MY D MY D'S D MY D MY D MY D MY D MY D MY D'S D MY D MY D MY D MY D MY D

---
## Run: multi_speaker_4_4_lawrence_0_23_23_50.0s_grouped | 2026-03-05 12:42:47

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4_lawrence_0_23_23_passive_tongue_with_audio.mp4 | 0.7094 | 0.9565 | 0.9562 | 0.8330 | 29.06% | 4.35% | 42 |
| 4_lawrence_0_23_23_with_tongue_with_audio.mp4 | 0.7215 | 0.9638 | 0.9635 | 0.8427 | 27.85% | 3.62% | 48 |

- Best (by composite): **4_lawrence_0_23_23_passive_tongue_with_audio.mp4** (VER=0.7094, WER_norm=0.9565, Composite=0.8330)
- Worst (by composite): **4_lawrence_0_23_23_with_tongue_with_audio.mp4** (VER=0.7215, WER_norm=0.9638, Composite=0.8427)
- VER gap (worst - best): **0.0121**
- WER gap (worst - best): **0.0072**
- Composite gap (worst - best): **0.0097**

### Ground Truth
can remember the first time i tucker yucky or iraq dumplings as we would call them in america i was pretty new to japan at the time and i was walking through a major shopping area near number station in osaka there were a lot of takoyaki stands and a ton of signs showing how good they were so i figured why not give them a shot now if you are not familiar with the food you would probably do exactly what i did the second i got them i put one in my mouth and i eli's i had made a mistake as it felt like a hot coal was in my mouth and i was a man with a burning mouth without a drink in my hand so i did what anyone would do and

### Hypotheses
#### 4_lawrence_0_23_23_passive_tongue_with_audio.mp4
- VER: 0.7094
- WER(norm): 0.9565
- WER(raw): 0.9562
- Composite Index: 0.8330
- Viseme Accuracy: 29.06%
- Word Accuracy(norm): 4.35%
- HYP: AS YOU CAN SEE THERE ARE A LOT OF DIFFERENT SIZES AND DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES DIFFERENT SIZES

#### 4_lawrence_0_23_23_with_tongue_with_audio.mp4
- VER: 0.7215
- WER(norm): 0.9638
- WER(raw): 0.9635
- Composite Index: 0.8427
- Viseme Accuracy: 27.85%
- Word Accuracy(norm): 3.62%
- HYP: AND LIKE EVERY ONE OF YOU HAS DIFFERENT MEANINGS IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT MEANING IT GIVES YOU DIFFERENT

---
## Run: multi_speaker_16_16_jorge_0_1_1_52.0s_grouped | 2026-03-05 12:47:38

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16_jorge_0_1_1_passive_tongue_with_audio.mp4 | 0.7418 | 0.9082 | 0.9096 | 0.8250 | 25.82% | 9.18% | 86 |
| 16_jorge_0_1_1_with_tongue_with_audio.mp4 | 0.7605 | 0.9031 | 0.9043 | 0.8318 | 23.95% | 9.69% | 82 |

- Best (by composite): **16_jorge_0_1_1_passive_tongue_with_audio.mp4** (VER=0.7418, WER_norm=0.9082, Composite=0.8250)
- Worst (by composite): **16_jorge_0_1_1_with_tongue_with_audio.mp4** (VER=0.7605, WER_norm=0.9031, Composite=0.8318)
- VER gap (worst - best): **0.0187**
- WER gap (worst - best): **-0.0051**
- Composite gap (worst - best): **0.0068**

### Ground Truth
the first thing i like to do on weekends is relaxing and i'll go shopping if i'm not that tired since i started my job i think it's very important to get a good sleep during the weekend because when you have to work monday through friday the whole week you're very tired so getting a good rest is as important as completing an excellent job in my spare time if i feel okay i will go for a walk or hike in nature sometimes i try to organize something for my friends volunteer at the buddhist temple on the weekend or i can just walk around enjoying the sunshine i right to live a healthy lifestyle considering how much time i spend sitting at work i always try to move as much as i can while i'm not working and some other days when i'm when i'm free i like to listen to music or watch the commentary movies on my laptop but sometimes i'll just sleep i especially liked watching japanese anime i think watching anime is helpful for me to learn and express japanese but

### Hypotheses
#### 16_jorge_0_1_1_passive_tongue_with_audio.mp4
- VER: 0.7418
- WER(norm): 0.9082
- WER(raw): 0.9096
- Composite Index: 0.8250
- Viseme Accuracy: 25.82%
- Word Accuracy(norm): 9.18%
- HYP: THE NEXT THING I WANT TO SHOW YOU IS THE SIMPLEST WAY TO DO IT THE SIMPLEST WAY TO DO IT IS TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO USE THE SIMPLEST WAY TO DO IT IS

#### 16_jorge_0_1_1_with_tongue_with_audio.mp4
- VER: 0.7605
- WER(norm): 0.9031
- WER(raw): 0.9043
- Composite Index: 0.8318
- Viseme Accuracy: 23.95%
- Word Accuracy(norm): 9.69%
- HYP: IN GENERAL I WOULD LIKE TO INVITE YOU TO JOIN ME IN THIS SESSION I WOULD LIKE TO INVITE YOU TO JOIN ME IN THIS SESSION BECAUSE I WANT YOU TO JOIN ME IN THIS SESSION I WOULD LIKE TO INVITE YOU TO JOIN ME IN THIS SESSION AND THIS SESSION I WOULD LIKE TO INVITE YOU TO JOIN ME IN THIS SESSION AND THIS SESSION I WOULD LIKE TO INVITE YOU TO JOIN ME IN FOR THIS SESSION AND THIS SESSION

---
## Run: multi_speaker_5_5_stewart_0_10_10_53.0s_grouped | 2026-03-05 12:52:49

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5_stewart_0_10_10_passive_tongue_with_audio.mp4 | 0.6900 | 0.9080 | 0.9064 | 0.7990 | 31.00% | 9.20% | 107 |
| 5_stewart_0_10_10_with_tongue_with_audio.mp4 | 0.7419 | 0.9368 | 0.9357 | 0.8394 | 25.81% | 6.32% | 38 |

- Best (by composite): **5_stewart_0_10_10_passive_tongue_with_audio.mp4** (VER=0.6900, WER_norm=0.9080, Composite=0.7990)
- Worst (by composite): **5_stewart_0_10_10_with_tongue_with_audio.mp4** (VER=0.7419, WER_norm=0.9368, Composite=0.8394)
- VER gap (worst - best): **0.0520**
- WER gap (worst - best): **0.0287**
- Composite gap (worst - best): **0.0404**

### Ground Truth
i would prefer to choose a major that is easy to find a good job in the future like finance or marketing for example there's no one that can deny the most common reason for attending university is to get prepared for a good job in the future so whether the major will lead us to a good job on not is the most important reason why we choose our major and if we find a good job with a decent payment we can use the money that we have learned from it to by our own interest for example i like painting a lot however i choose painting of my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance-related job after graduation after university i can get the high salary and in my free time i can use my salary to hire a professional teacher to teach me how to draw

### Hypotheses
#### 5_stewart_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.6900
- WER(norm): 0.9080
- WER(raw): 0.9064
- Composite Index: 0.7990
- Viseme Accuracy: 31.00%
- Word Accuracy(norm): 9.20%
- HYP: OUR APPROACH TO DISSERTATION IS TO DISSERT THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW THE NEED TO KNOW

#### 5_stewart_0_10_10_with_tongue_with_audio.mp4
- VER: 0.7419
- WER(norm): 0.9368
- WER(raw): 0.9357
- Composite Index: 0.8394
- Viseme Accuracy: 25.81%
- Word Accuracy(norm): 6.32%
- HYP: UNIVERSITY OF THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS THE UNIVERSITY OF ILLINOIS

---
## Run: multi_speaker_7_7_sophie_0_10_10_54.0s_grouped | 2026-03-05 12:57:47

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7_sophie_0_10_10_with_tongue_with_audio.mp4 | 0.6948 | 0.9500 | 0.9605 | 0.8224 | 30.52% | 5.00% | 70 |
| 7_sophie_0_10_10_passive_tongue_with_audio.mp4 | 0.7276 | 0.9222 | 0.9209 | 0.8249 | 27.24% | 7.78% | 69 |

- Best (by composite): **7_sophie_0_10_10_with_tongue_with_audio.mp4** (VER=0.6948, WER_norm=0.9500, Composite=0.8224)
- Worst (by composite): **7_sophie_0_10_10_passive_tongue_with_audio.mp4** (VER=0.7276, WER_norm=0.9222, Composite=0.8249)
- VER gap (worst - best): **0.0328**
- WER gap (worst - best): **-0.0278**
- Composite gap (worst - best): **0.0025**

### Ground Truth
i would prefer to choose a major that is easy for me to find a good job in the future like finance or marketing for example there's no there's no one that can deny that the most common reason for attending universities to get prepared for a good job in the future so whether the major would lead us to a good job or not is the most important reason why we choose our major if we find a good job with a decent payment we can choose the money that we have and from it from from it to satisfy our own interest for example i like painting a lot so however i choose to painting in my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance related job after graduation after university can get high salary and in my free time i can use my salary to hire professional teacher to teach me how to draw

### Hypotheses
#### 7_sophie_0_10_10_with_tongue_with_audio.mp4
- VER: 0.6948
- WER(norm): 0.9500
- WER(raw): 0.9605
- Composite Index: 0.8224
- Viseme Accuracy: 30.52%
- Word Accuracy(norm): 5.00%
- HYP: I DON'T KNOW IF YOU CAN SEE IT ON THE SCREEN BUT IF YOU CAN SEE IT ON THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN IT'S DIFFERENT FROM THE SCREEN

#### 7_sophie_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.7276
- WER(norm): 0.9222
- WER(raw): 0.9209
- Composite Index: 0.8249
- Viseme Accuracy: 27.24%
- Word Accuracy(norm): 7.78%
- HYP: AS YOU CAN SEE I HAVE THREE DIFFERENT WAYS OF DOING IT ONE IS USING DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING IT I CAN USE DIFFERENT WAYS OF DOING DIFFERENT WAYS

---
## Run: segmented_multi_speaker_2_2_scott_0_11_11 | 2026-03-05 13:11:18

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `segmented`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2_scott_0_11_11_passive_tongue_with_audio.mp4 | 0.6768 | 0.9028 | 0.9130 | 0.7898 | 32.32% | 9.72% | 121 |
| 2_scott_0_11_11_with_tongue_with_audio.mp4 | 0.6985 | 0.9236 | 0.9203 | 0.8110 | 30.15% | 7.64% | 136 |

- Best (by composite): **2_scott_0_11_11_passive_tongue_with_audio.mp4** (VER=0.6768, WER_norm=0.9028, Composite=0.7898)
- Worst (by composite): **2_scott_0_11_11_with_tongue_with_audio.mp4** (VER=0.6985, WER_norm=0.9236, Composite=0.8110)
- VER gap (worst - best): **0.0217**
- WER gap (worst - best): **0.0208**
- Composite gap (worst - best): **0.0213**

### Ground Truth
i've got a room in my house or season i got a room in my house in my study room i have a lot of books in my study room such as fashion magazines inspirational books etc also there is a computer in my study in my study room usually i will check my computer to learn some study related information such as math problems that just can't do i also am able to use it just for fun one example i'm able to catch up on my on the latest netflix episodes of my favorite show and sometimes as i'm able to use it to check out some clothes and look inside the latest fashion magazines however most of them most of the time i just can't add it to my cart so i don't buy them

### Hypotheses
#### 2_scott_0_11_11_passive_tongue_with_audio.mp4
- VER: 0.6768
- WER(norm): 0.9028
- WER(raw): 0.9130
- Composite Index: 0.7898
- Viseme Accuracy: 32.32%
- Word Accuracy(norm): 9.72%
- HYP: I DON'T KNOW IF IT'S POSSIBLE I DON'T KNOW IF IT'S POSSIBLE I'M DELIGHTED FOR THE UNIVERSITY OF EAST ASIA AND THIS IS FOR THE UNIVERSITY I'M GOING TO SHOW YOU THE NEWEST COMPUTER I'M GOING TO SHOW YOU HI I'M JOSHUA FROM NEW ZEALAND AND I'LL TELL YOU THE INTRODUCTION AND THE SHOW HOW TO TRAVEL TEACHING AND HI WELCOME TO FUNKY FIVE T V MY NAME IS DON FUNK AND MY HUSBAND BRETT FUNK AND I HAVE ANOTHER EPISODE OF T V SHOW I SOMETIMES GET BIBLES TO CHECK OUT SOME OF THE NEWS I DO NOT UNDERSTAND THE EDUCATION PROCESS I WANTED IT'S THE FIRST TIME THAT I'VE BEEN TEACHING ENGLISH AND RESEARCH AS WELL AS YOU TUBE

#### 2_scott_0_11_11_with_tongue_with_audio.mp4
- VER: 0.6985
- WER(norm): 0.9236
- WER(raw): 0.9203
- Composite Index: 0.8110
- Viseme Accuracy: 30.15%
- Word Accuracy(norm): 7.64%
- HYP: I STAND AT THE UNIVERSITY OF CHICAGO I STAND AT THE UNIVERSITY OF SAN FRANCISCO AND THE CARDINAL VOLUNTARY STATE OF EXISTENCE THE AGE OF EIGHTEEN IS VOLUNTARY I ALSO DON'T KNOW WHAT'S GOING ON I'M GOING TO DO THAT YOU KNOW I'M NOT DRIVING BECAUSE YOU KNOW THERE IS A CAR THERE WAS AN OLD CAR DONATION AND YOU KNOW I DON'T DRIVE ALL THESE THINGS AND HI HOW ARE YOU I'M JILL DUFFY HERE IN THE STUDIO AND TODAY I'M GOING TO TALK ABOUT ANOTHER EPISODE OF THE CHANNEL AND SOMETIMES I'LL BE ABLE TO USE THE TRIANGLES OF THE SCHOOLS I CAN USE TO HELP YOU GUIDE YOUR EDUCATION PROCESS AND ALSO IT WAS THE FIRST TIME I'M GOING TO JUST KIND OF ENJOY IT AND ENJOY IT SO I CHOSE TO FAST

---
## Run: segmented_multi_speaker_4_4_lawrence_0_23_23 | 2026-03-05 13:13:17

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `segmented`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4_lawrence_0_23_23_passive_tongue_with_audio.mp4 | 0.7094 | 0.9420 | 0.9416 | 0.8257 | 29.06% | 5.80% | 114 |
| 4_lawrence_0_23_23_with_tongue_with_audio.mp4 | 0.7022 | 0.9710 | 0.9270 | 0.8366 | 29.78% | 2.90% | 115 |

- Best (by composite): **4_lawrence_0_23_23_passive_tongue_with_audio.mp4** (VER=0.7094, WER_norm=0.9420, Composite=0.8257)
- Worst (by composite): **4_lawrence_0_23_23_with_tongue_with_audio.mp4** (VER=0.7022, WER_norm=0.9710, Composite=0.8366)
- VER gap (worst - best): **-0.0073**
- WER gap (worst - best): **0.0290**
- Composite gap (worst - best): **0.0109**

### Ground Truth
can remember the first time i tucker yucky or iraq dumplings as we would call them in america i was pretty new to japan at the time and i was walking through a major shopping area near number station in osaka there were a lot of takoyaki stands and a ton of signs showing how good they were so i figured why not give them a shot now if you are not familiar with the food you would probably do exactly what i did the second i got them i put one in my mouth and i eli's i had made a mistake as it felt like a hot coal was in my mouth and i was a man with a burning mouth without a drink in my hand so i did what anyone would do and

### Hypotheses
#### 4_lawrence_0_23_23_passive_tongue_with_audio.mp4
- VER: 0.7094
- WER(norm): 0.9420
- WER(raw): 0.9416
- Composite Index: 0.8257
- Viseme Accuracy: 29.06%
- Word Accuracy(norm): 5.80%
- HYP: THERE ARE A LOT OF DIFFERENT TYPES OF CHINESE CHARACTERS AND DIFFERENT TYPES OF CHARACTERS SO AS YOU CAN SEE IT'S JUST A LITTLE DIFFERENT IT'S A LITTLE DIFFERENT IT'S A LITTLE DIFFERENT IT'S A LITTLE DIFFERENT THE FIRST THING YOU NEED TO KNOW IS THAT YOU NEED TO KNOW YOUR SKILLS SO THAT YOU CAN LEARN MORE ABOUT YOURSELF IT'S VERY EASY TO READ IT'S VERY EASY TO READ CORRECTLY SO YOU SHOULD READ CORRECTLY I CAN'T SEE IT I CAN'T SEE IT ANYMORE OF COURSE I CAN SEE THIS IS WHAT I WANT YOU TO DO TO RECEIVE ADVICE I WAS BORN IN THE UNIVERSITY OF ILLINOIS AND THE ANSWER IS YES

#### 4_lawrence_0_23_23_with_tongue_with_audio.mp4
- VER: 0.7022
- WER(norm): 0.9710
- WER(raw): 0.9270
- Composite Index: 0.8366
- Viseme Accuracy: 29.78%
- Word Accuracy(norm): 2.90%
- HYP: HI EVERYONE MY NAME IS ALICIA AND TODAY I WANT TO TALK TO YOU ABOUT ALL THE LEARNING THAT YOU'RE LEARNING THAT YOU'RE LEARNING THAT YOU'RE LEARNING THAT YOU'RE LEARNING THAT I'M NOT A DOCTOR I'M NOT A DOCTOR I'M NOT A DOCTOR I'M A DOCTOR I'M A DOCTOR THEY DON'T KNOW ANYTHING ABOUT IT THEY DON'T KNOW ANYTHING ABOUT IT THEY DON'T KNOW ANYTHING ABOUT IT IF YOU DON'T KNOW YOU DON'T KNOW YOU DON'T KNOW YOU DON'T KNOW THE FIRST THING I WANT TO SHOW YOU IS THIS FIVE HUNDRED AND NOW I'M GOING TO SHOW YOU A FEW EXAMPLES OF HOW YOU CAN ACHIEVE I DON'T KNOW I DON'T KNOW THE ANSWER

---
## Run: segmented_multi_speaker_16_16_jorge_0_1_1 | 2026-03-05 13:15:28

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `segmented`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16_jorge_0_1_1_with_tongue_with_audio.mp4 | 0.6625 | 0.8724 | 0.8936 | 0.7675 | 33.75% | 12.76% | 158 |
| 16_jorge_0_1_1_passive_tongue_with_audio.mp4 | 0.6874 | 0.8878 | 0.9043 | 0.7876 | 31.26% | 11.22% | 150 |

- Best (by composite): **16_jorge_0_1_1_with_tongue_with_audio.mp4** (VER=0.6625, WER_norm=0.8724, Composite=0.7675)
- Worst (by composite): **16_jorge_0_1_1_passive_tongue_with_audio.mp4** (VER=0.6874, WER_norm=0.8878, Composite=0.7876)
- VER gap (worst - best): **0.0249**
- WER gap (worst - best): **0.0153**
- Composite gap (worst - best): **0.0201**

### Ground Truth
the first thing i like to do on weekends is relaxing and i'll go shopping if i'm not that tired since i started my job i think it's very important to get a good sleep during the weekend because when you have to work monday through friday the whole week you're very tired so getting a good rest is as important as completing an excellent job in my spare time if i feel okay i will go for a walk or hike in nature sometimes i try to organize something for my friends volunteer at the buddhist temple on the weekend or i can just walk around enjoying the sunshine i right to live a healthy lifestyle considering how much time i spend sitting at work i always try to move as much as i can while i'm not working and some other days when i'm when i'm free i like to listen to music or watch the commentary movies on my laptop but sometimes i'll just sleep i especially liked watching japanese anime i think watching anime is helpful for me to learn and express japanese but

### Hypotheses
#### 16_jorge_0_1_1_with_tongue_with_audio.mp4
- VER: 0.6625
- WER(norm): 0.8724
- WER(raw): 0.8936
- Composite Index: 0.7675
- Viseme Accuracy: 33.75%
- Word Accuracy(norm): 12.76%
- HYP: IN AUSTRALIA I NEED TO GO TO ENGLAND AS WELL I NEED TO GO TO GERMANY AS I STARTED LISTENING IT'S AN EXCELLENT WAY TO CONNECT YOUR PHONE TO YOUR PHONE TO YOUR PHONE TO YOUR PHONE TO YOUR PHONE TO CONNECT YOU CAN GO OVER THERE YOU CAN CLICK ON THE REST WHICH IS REALLY EASY TO TRAVEL AS I SAID OKAY I WANT TO GO TO SCHOOL I CLICKED AS A RESULT OF THE LACK OF SENSITIVITY OF SENSITIVITY WITHIN THE SENSITIVITY AND THE LACK OF UNDERSTANDING I DON'T LIKE THE WAY THAT I LIKE THE WAY THAT I LIKE THE WAY THAT I LIKE TO WORK I ALSO LIKE THE WAY THAT I LIKE TO WORK HEY GUYS HOW ARE YOU TODAY I'D LIKE TO INTRODUCE YOU TO THE NEWEST AND MOST IMPORTANT TOPIC IN WHICH I WILL DISCUSS IT'S VERY NICE TO HAVE A DIFFERENT OUTFIT AND IT'S VERY NICE TO HAVE A DIFFERENT

#### 16_jorge_0_1_1_passive_tongue_with_audio.mp4
- VER: 0.6874
- WER(norm): 0.8878
- WER(raw): 0.9043
- Composite Index: 0.7876
- Viseme Accuracy: 31.26%
- Word Accuracy(norm): 11.22%
- HYP: THAT'S WHAT I'M GOING TO TALK ABOUT TODAY I'M GOING TO TALK TO YOU ABOUT IN THIS LESSON WE ARE GOING TO TEACH YOU HOW TO USE THE SKILLS THAT YOU NEED TO USE IN ORDER TO HELP YOU LEARN INCORPORATE A LOT OF THE THINGS THAT YOU ARE TRYING TO ACHIEVE WITH THIS ONE IS IT MIGHT BE THE SAME AS AN EXAMPLE I WOULD LIKE TO WORK ON THE IMPETUS IS TO ENSURE THE IMPETUS TO PROTECT THE IMPELLER TO PROTECT THE IMPELLER TO THE SIDE THE IMPELLER IS NOT AN IMPELLER IS AN IMPELLER OR AN IMPELLER OR AN IMPELLER AS SOMEONE AS A CHILD I WOULD LIKE TO INTRODUCE YOU TO A NEW PIECE OF MUSIC THAT I WOULD LIKE TO SHARE WITH YOU AS YOU CAN SEE I WANT TO GIVE YOU A TIME AND I WANT YOU TO BE ABLE TO SEE THE DIFFERENCE

---
## Run: segmented_multi_speaker_5_5_stewart_0_10_10 | 2026-03-05 13:17:31

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `segmented`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5_stewart_0_10_10_passive_tongue_with_audio.mp4 | 0.6523 | 0.9080 | 0.9064 | 0.7802 | 34.77% | 9.20% | 121 |
| 5_stewart_0_10_10_with_tongue_with_audio.mp4 | 0.6989 | 0.9483 | 0.9240 | 0.8236 | 30.11% | 5.17% | 151 |

- Best (by composite): **5_stewart_0_10_10_passive_tongue_with_audio.mp4** (VER=0.6523, WER_norm=0.9080, Composite=0.7802)
- Worst (by composite): **5_stewart_0_10_10_with_tongue_with_audio.mp4** (VER=0.6989, WER_norm=0.9483, Composite=0.8236)
- VER gap (worst - best): **0.0466**
- WER gap (worst - best): **0.0402**
- Composite gap (worst - best): **0.0434**

### Ground Truth
i would prefer to choose a major that is easy to find a good job in the future like finance or marketing for example there's no one that can deny the most common reason for attending university is to get prepared for a good job in the future so whether the major will lead us to a good job on not is the most important reason why we choose our major and if we find a good job with a decent payment we can use the money that we have learned from it to by our own interest for example i like painting a lot however i choose painting of my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance-related job after graduation after university i can get the high salary and in my free time i can use my salary to hire a professional teacher to teach me how to draw

### Hypotheses
#### 5_stewart_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.6523
- WER(norm): 0.9080
- WER(raw): 0.9064
- Composite Index: 0.7802
- Viseme Accuracy: 34.77%
- Word Accuracy(norm): 9.20%
- HYP: HOWEVER CONSIDERS JUSTIFICATION AND SKILLS THAT YOU SHOULD BE USING AND TRAINING YOUR PRACTICE THROUGH IN THIS TALK TODAY I HAVE THE SERVICES AND SERVICES DESIGNED FOR FINANCIAL SERVICES SO IDENTIFICATIONS AND INSTRUCTIONS AREN'T EASY LOTS OF GOODS OR ADDITIONAL INFORMATION I NEED TO TEACH YOU THE SKILLS AND THE STRATEGIES YOU CAN USE TO TEACH YOU THE SKILLS AND SKILLS YOU NEED TO SET UP HI MY NAME IS ALICIA AND TODAY I'M GOING TO TEACH YOU HOW TO TEACH ENGLISH IN ENGLISH AT THE END OF EACH SENTENCE I CAN SAY THIS TO THE STUDENT IN THE SITUATION I CHOOSE TO SAY THAT I STAY AS FOR THE FIRST TIME I DECIDED TO SET NEW CONDITIONS AND CHALLENGES INTO

#### 5_stewart_0_10_10_with_tongue_with_audio.mp4
- VER: 0.6989
- WER(norm): 0.9483
- WER(raw): 0.9240
- Composite Index: 0.8236
- Viseme Accuracy: 30.11%
- Word Accuracy(norm): 5.17%
- HYP: AT UNIVERSITY OF GEORGIA OF COLLEGE IN NEW ZEALAND I DECIDED TO JOIN THE UNIVERSITY OF GEORGIA AND TRY TO GET UNIVERSITY OF UTAH I JUST I THOUGHT THAT I NEEDED A HIGH KIND OF A SOFTNESS JUST A LITTLE OF THIS JUST A KIND OF SOFTNESS OF SOLUTION SO THAT DOESN'T AFFECT YOUR SKIN IT'S JUST HARD SKIN IT'S NOT AS IF YOU'RE GOING TO GET IT THAT DOESN'T AFFECT YOUR AS YOU CAN TELL YOU KNOW THERE'S A LIST OF ENGLISH AND THERE'S A LIST OF ENGLISH AND THERE'S A LIST OF ENGLISH HIGH NAME FANTASY NOT UH YOU HAVE GOT I TUNES FANTASY I'VE GOT I TUNES I TUNES IT'S FANTASY IF THERE IS ANY KIND OF EXERCISE OF DIALYSIS TAKE LESS OF AN INSTALLATION USES CONGRATULATIONS AND USES TAKE LESS HINDSIGHT AND IF IT'S NOT TIME YET I CAN USE THE STANDARDS I LOVE USING THE STANDARDS

---
## Run: segmented_multi_speaker_7_7_sophie_0_10_10 | 2026-03-05 13:19:40

### Settings
- config: `configs/LRS3_V_WER19.1.ini`
- infer mode: `segmented`
- vowel mode: `grouped`
- detector: `mediapipe`
- infer script: `/home/iite/ICT-FaceKit/ADFA_EVALUATION/Visual_Speech_Recognition_for_Multiple_Languages/infer.py`

### VER Summary
| Video | VER | WER(norm) | WER(raw) | Composite (0.5*VER + 0.5*WER_norm) | Viseme Accuracy | Word Accuracy(norm) | HYP words |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7_sophie_0_10_10_with_tongue_with_audio.mp4 | 0.6569 | 0.9111 | 0.8983 | 0.7840 | 34.31% | 8.89% | 149 |
| 7_sophie_0_10_10_passive_tongue_with_audio.mp4 | 0.6879 | 0.9389 | 0.9322 | 0.8134 | 31.21% | 6.11% | 145 |

- Best (by composite): **7_sophie_0_10_10_with_tongue_with_audio.mp4** (VER=0.6569, WER_norm=0.9111, Composite=0.7840)
- Worst (by composite): **7_sophie_0_10_10_passive_tongue_with_audio.mp4** (VER=0.6879, WER_norm=0.9389, Composite=0.8134)
- VER gap (worst - best): **0.0310**
- WER gap (worst - best): **0.0278**
- Composite gap (worst - best): **0.0294**

### Ground Truth
i would prefer to choose a major that is easy for me to find a good job in the future like finance or marketing for example there's no there's no one that can deny that the most common reason for attending universities to get prepared for a good job in the future so whether the major would lead us to a good job or not is the most important reason why we choose our major if we find a good job with a decent payment we can choose the money that we have and from it from from it to satisfy our own interest for example i like painting a lot so however i choose to painting in my profession i don't think i can make a lot of money as a painter but if i go for a major like finance i can get a finance related job after graduation after university can get high salary and in my free time i can use my salary to hire professional teacher to teach me how to draw

### Hypotheses
#### 7_sophie_0_10_10_with_tongue_with_audio.mp4
- VER: 0.6569
- WER(norm): 0.9111
- WER(raw): 0.8983
- Composite Index: 0.7840
- Viseme Accuracy: 34.31%
- Word Accuracy(norm): 8.89%
- HYP: I JUST WANT TO DO SOMETHING LIKE A REALLY REALLY GOOD JOB IT'S REALLY GOOD IT'S REALLY GOOD YOU DON'T NEED THAT KIND OF STUFF IN MY HEAD IT'S A LOT OF USE A LOT OF USE A LOT OF USE IT'S VERY VERY DIFFERENT THAN THE ORIGINAL VERSION OF THE NEW VERSION OF THE NEW FIVE HUNDRED VERSION IT'S REALLY GOOD TO KNOW THAT IF YOU HAVEN'T READ IT YOU DON'T REALLY HAVE AN IPHONE FOR ONE TOO I DON'T KNOW IF YOU'VE EVER SEEN THE VIDEOS I'VE SEEN THE VIDEOS I'VE SEEN THE VIDEOS I'VE SEEN UH I DON'T WANT TO GIVE A LOT OF MONEY I JUST FOUND ONES THAT USED TO MAKE A LOT OF MONEY AND I DON'T WANT TO GET TOO CLOSE TO IT I DON'T KNOW I'M GOING TO GET INTO THE HEALING PROCESS I'M GOING TO GET INTO THE HEALING PROCESS

#### 7_sophie_0_10_10_passive_tongue_with_audio.mp4
- VER: 0.6879
- WER(norm): 0.9389
- WER(raw): 0.9322
- Composite Index: 0.8134
- Viseme Accuracy: 31.21%
- Word Accuracy(norm): 6.11%
- HYP: I DON'T KNOW IF YOU CAN FIND IT IT'S VERY INTERESTING TO SEE IF YOU CAN FIND IT WHAT I'M GOING TO DO IS I'M GOING TO SHOW YOU THE DESIGN AND THE DESIGN AND THE DESIGN AND THE DESIGN OF THE PRODUCT ITSELF I DON'T THINK THERE'S ANY REASON WHY THIS VIDEO IS SO USEFUL FOR YOU IT'S MORE DIFFICULT TO FIND A WAY TO FIND I HAVE A FRIEND WHO IS ONE OF MY FAVORITE WRITERS IF YOU HAVE ANY QUESTIONS THANK YOU FOR WATCHING THIS VIDEO I'M GOING TO TEACH YOU HOW TO READ AND WRITE AND HOW TO WRITE AND WRITE HI I'M CHRISTY AND WELCOME TO ANOTHER EDITION OF FIREFOX NEWS WE'RE GOING TO DO A VIDEO ON FIREFOX NEWS HI I'M JOE AND TODAY I'M GOING TO BE TALKING TO YOU ABOUT THE LATEST VIDEOS AND THE LATEST VIDEOS

---
