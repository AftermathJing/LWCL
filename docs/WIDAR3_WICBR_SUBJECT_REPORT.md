# Wi-CBR Widar3 Subject-Disjoint Report

- Remote host: Linux-5.15.0-139-generic-x86_64-with-glibc2.31
- Git branch: main
- Git commit: 099ec8e371182786688720eed88c76f8e91f16fa
- Manifest: /home/wj/LWCL/data/splits/widar3_wicbr_subject_split.csv

## 1. Objective

Reproduce Wi-CBR on the LWCL Widar3 subject-disjoint protocol using raw-CSI-derived phase and DFS images.

## 2. Data split

| Split | Samples | Subjects |
| --- | ---: | --- |
| train | 8247 | user1, user10, user11, user13, user14, user15, user16, user3, user5, user6, user8 |
| validation | 1499 | user17, user9 |
| test | 1625 | user12, user2, user7 |

## 3. Best validation checkpoint

| Metric | Value |
| --- | ---: |
| accuracy | 62.84% |
| macro precision | 66.69% |
| macro recall | 62.85% |
| macro F1 | 62.46% |
| loss | 2.0183 |

## 4. Test result

| Metric | Value |
| --- | ---: |
| accuracy | 63.57% |
| macro precision | 74.26% |
| macro recall | 65.18% |
| macro F1 | 64.68% |
| loss | 1.9200 |

## 5. Per-class metrics

| Label | Gesture | Precision | Recall | F1 | Support |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | push_pull | 89.25% | 44.27% | 59.18% | 375 |
| 1 | sweep | 93.79% | 54.40% | 68.86% | 250 |
| 2 | clap | 88.24% | 54.00% | 67.00% | 250 |
| 3 | slide | 82.18% | 66.40% | 73.45% | 250 |
| 4 | draw_circle | 43.83% | 92.40% | 59.46% | 250 |
| 5 | draw_zigzag | 48.30% | 79.60% | 60.12% | 250 |

## 6. Confusion matrix

| Actual / Predicted | push_pull | sweep | clap | slide | draw_circle | draw_zigzag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| push_pull | 166 | 1 | 4 | 13 | 141 | 50 |
| sweep | 0 | 136 | 2 | 5 | 52 | 55 |
| clap | 2 | 0 | 135 | 18 | 42 | 53 |
| slide | 3 | 1 | 10 | 166 | 28 | 42 |
| draw_circle | 4 | 0 | 2 | 0 | 231 | 13 |
| draw_zigzag | 11 | 7 | 0 | 0 | 33 | 199 |

## 7. Group accuracy

| Group | Accuracy | Count |
| --- | ---: | ---: |
| subject user12 | 85.33% | 750 |
| subject user2 | 31.20% | 125 |
| subject user7 | 47.20% | 750 |
| environment 20181130 | 85.33% | 750 |
| environment 20181209 | 31.20% | 125 |
| environment 20181211 | 47.20% | 750 |
| position 1 | 57.23% | 325 |
| position 2 | 66.77% | 325 |
| position 3 | 61.23% | 325 |
| position 4 | 62.15% | 325 |
| position 5 | 70.46% | 325 |
| orientation 1 | 62.46% | 325 |
| orientation 2 | 65.23% | 325 |
| orientation 3 | 63.69% | 325 |
| orientation 4 | 64.00% | 325 |
| orientation 5 | 62.46% | 325 |
