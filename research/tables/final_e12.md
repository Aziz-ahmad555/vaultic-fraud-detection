# E12: trust layer (final)

Generated 2026-10-11 by `python -m vaultic.fusion.final_run --mode final`. Pre-registered in D101 (details D102). Evaluation rows: 92427 (3213 frauds); thresholds and all calibration on 10276 rows (300 frauds); 5 seeds; 1000 paired bootstrap resamples.

## MVAF

Reliability diagram: `E:/old work/fyp 11/fraud-detection/.worktrees/phase7-prep/research/figures/final_e12_reliability_MVAF.png`.

Calibrator: IsotonicCalibrator. ECE 0.1471 → 0.0050; Brier 0.0840 → 0.0195.

| conformal | coverage | legit | fraud | uncertain sets | within 2 pts |
|---|---|---|---|---|---|
| Mondrian 0.90 | 0.9063 | 0.9064 | 0.9023 | 0.1220 | yes |
| Mondrian 0.95 | 0.9779 | 0.9783 | 0.9658 | 0.4155 | no |
| adaptive 0.90 (L = 30) | 0.9063 | | | | blocks: 151-157: 0.915, 158-164: 0.898, 165-171: 0.907, 172-178: 0.907, 179-182: 0.903 |
| adaptive 0.95 (L = 30) | 0.9779 | | | | blocks: 151-157: 0.981, 158-164: 0.977, 165-171: 0.979, 172-178: 0.977, 179-182: 0.974 |

Decision engine, default costs: total cost $99,263 (sensitivity sweep: 135 settings, in the JSON).

| K/day | policy | fraud value caught | fraud caught | recall@K | $ saved / review |
|---|---|---|---|---|---|
| 50 | R1 | 156,571 | 1273 | 0.3962 | 97.9 |
| 50 | R2 | 256,734 | 772 | 0.2403 | 160.5 |
| 50 | R3 | 256,734 | 772 | 0.2403 | 160.5 |
| 50 | R4 | 256,734 | 772 | 0.2403 | 160.5 |
| 100 | R1 | 248,635 | 1817 | 0.5655 | 77.7 |
| 100 | R2 | 320,820 | 1236 | 0.3847 | 100.3 |
| 100 | R3 | 320,820 | 1236 | 0.3847 | 100.3 |
| 100 | R4 | 319,754 | 1152 | 0.3585 | 99.9 |
| 200 | R1 | 339,365 | 2254 | 0.7015 | 53.0 |
| 200 | R2 | 389,766 | 1776 | 0.5528 | 60.9 |
| 200 | R3 | 389,766 | 1776 | 0.5528 | 60.9 |
| 200 | R4 | 381,475 | 1621 | 0.5045 | 59.6 |
| 500 | R1 | 420,881 | 2761 | 0.8593 | 26.3 |
| 500 | R2 | 448,887 | 2467 | 0.7678 | 28.1 |
| 500 | R3 | 448,887 | 2467 | 0.7678 | 28.1 |
| 500 | R4 | 448,887 | 2467 | 0.7678 | 28.1 |

H5 (fraud value, day-block bootstrap; Holm across this method's H5 tests, D104):

- K = 50: R3 - R1 +100,164 [+75,151, +127,393], p = 0.000, p (Holm) = 0.000
- K = 50: R4 - R1 +100,164 [+75,151, +127,393], p = 0.000, p (Holm) = 0.000
- K = 100: R3 - R1 +72,184 [+47,001, +99,807], p = 0.000, p (Holm) = 0.000
- K = 100: R4 - R1 +71,119 [+44,720, +100,478], p = 0.000, p (Holm) = 0.000
- K = 200: R3 - R1 +50,401 [+32,487, +70,092], p = 0.000, p (Holm) = 0.000
- K = 200: R4 - R1 +42,110 [+22,784, +64,220], p = 0.000, p (Holm) = 0.000
- K = 500: R3 - R1 +28,007 [+15,800, +41,061], p = 0.000, p (Holm) = 0.000
- K = 500: R4 - R1 +28,007 [+15,800, +41,061], p = 0.000, p (Holm) = 0.000

## F3

Reliability diagram: `E:/old work/fyp 11/fraud-detection/.worktrees/phase7-prep/research/figures/final_e12_reliability_F3.png`.

Calibrator: IsotonicCalibrator. ECE 0.1551 → 0.0139; Brier 0.0960 → 0.0293.

| conformal | coverage | legit | fraud | uncertain sets | within 2 pts |
|---|---|---|---|---|---|
| Mondrian 0.90 | 0.8917 | 0.8909 | 0.9147 | 0.1625 | yes |
| Mondrian 0.95 | 0.9486 | 0.9481 | 0.9623 | 0.3754 | yes |
| adaptive 0.90 (L = 30) | 0.8917 | | | | blocks: 151-157: 0.916, 158-164: 0.882, 165-171: 0.869, 172-178: 0.892, 179-182: 0.904 |
| adaptive 0.95 (L = 30) | 0.9486 | | | | blocks: 151-157: 0.965, 158-164: 0.949, 165-171: 0.925, 172-178: 0.948, 179-182: 0.962 |

Decision engine, default costs: total cost $101,344 (sensitivity sweep: 135 settings, in the JSON).

| K/day | policy | fraud value caught | fraud caught | recall@K | $ saved / review |
|---|---|---|---|---|---|
| 50 | R1 | 139,425 | 1137 | 0.3539 | 87.1 |
| 50 | R2 | 241,965 | 737 | 0.2294 | 151.2 |
| 50 | R3 | 241,965 | 737 | 0.2294 | 151.2 |
| 50 | R4 | 237,851 | 705 | 0.2194 | 148.7 |
| 100 | R1 | 208,680 | 1676 | 0.5216 | 65.2 |
| 100 | R2 | 311,842 | 1148 | 0.3573 | 97.5 |
| 100 | R3 | 311,842 | 1148 | 0.3573 | 97.5 |
| 100 | R4 | 316,206 | 1054 | 0.3280 | 98.8 |
| 200 | R1 | 300,962 | 2085 | 0.6489 | 47.0 |
| 200 | R2 | 373,851 | 1642 | 0.5110 | 58.4 |
| 200 | R3 | 373,851 | 1642 | 0.5110 | 58.4 |
| 200 | R4 | 378,953 | 1607 | 0.5002 | 59.2 |
| 500 | R1 | 412,953 | 2637 | 0.8207 | 25.8 |
| 500 | R2 | 439,487 | 2312 | 0.7196 | 27.5 |
| 500 | R3 | 439,487 | 2312 | 0.7196 | 27.5 |
| 500 | R4 | 437,953 | 2293 | 0.7137 | 27.4 |

H5 (fraud value, day-block bootstrap; Holm across this method's H5 tests, D104):

- K = 50: R3 - R1 +102,540 [+79,237, +129,651], p = 0.000, p (Holm) = 0.000
- K = 50: R4 - R1 +98,426 [+73,984, +127,358], p = 0.000, p (Holm) = 0.000
- K = 100: R3 - R1 +103,162 [+76,676, +133,734], p = 0.000, p (Holm) = 0.000
- K = 100: R4 - R1 +107,526 [+75,574, +144,971], p = 0.000, p (Holm) = 0.000
- K = 200: R3 - R1 +72,889 [+48,734, +104,717], p = 0.000, p (Holm) = 0.000
- K = 200: R4 - R1 +77,991 [+50,668, +110,771], p = 0.000, p (Holm) = 0.000
- K = 500: R3 - R1 +26,533 [+14,899, +39,968], p = 0.000, p (Holm) = 0.000
- K = 500: R4 - R1 +24,999 [+13,282, +38,817], p = 0.000, p (Holm) = 0.000

## B5

Reliability diagram: `E:/old work/fyp 11/fraud-detection/.worktrees/phase7-prep/research/figures/final_e12_reliability_B5.png`.

Calibrator: IsotonicCalibrator. ECE 0.0080 → 0.0057; Brier 0.0183 → 0.0186.

| conformal | coverage | legit | fraud | uncertain sets | within 2 pts |
|---|---|---|---|---|---|
| Mondrian 0.90 | 0.9191 | 0.9175 | 0.9614 | 0.3036 | yes |
| Mondrian 0.95 | 0.9572 | 0.9570 | 0.9614 | 0.3444 | yes |
| adaptive 0.90 (L = 30) | 0.9191 | | | | blocks: 151-157: 0.927, 158-164: 0.911, 165-171: 0.920, 172-178: 0.919, 179-182: 0.916 |
| adaptive 0.95 (L = 30) | 0.9572 | | | | blocks: 151-157: 0.961, 158-164: 0.950, 165-171: 0.960, 172-178: 0.959, 179-182: 0.953 |

Decision engine, default costs: total cost $95,493 (sensitivity sweep: 135 settings, in the JSON).

| K/day | policy | fraud value caught | fraud caught | recall@K | $ saved / review |
|---|---|---|---|---|---|
| 50 | R1 | 152,430 | 1306 | 0.4065 | 95.3 |
| 50 | R2 | 256,946 | 897 | 0.2792 | 160.6 |
| 50 | R3 | 256,946 | 897 | 0.2792 | 160.6 |
| 100 | R1 | 246,605 | 1900 | 0.5913 | 77.1 |
| 100 | R2 | 338,881 | 1405 | 0.4373 | 105.9 |
| 100 | R3 | 338,881 | 1405 | 0.4373 | 105.9 |
| 200 | R1 | 340,034 | 2298 | 0.7152 | 53.1 |
| 200 | R2 | 398,083 | 1910 | 0.5945 | 62.2 |
| 200 | R3 | 398,083 | 1910 | 0.5945 | 62.2 |
| 500 | R1 | 425,981 | 2778 | 0.8646 | 26.6 |
| 500 | R2 | 450,674 | 2461 | 0.7660 | 28.2 |
| 500 | R3 | 450,674 | 2461 | 0.7660 | 28.2 |

H5 (fraud value, day-block bootstrap; Holm across this method's H5 tests, D104):

- K = 50: R3 - R1 +104,516 [+77,512, +133,208], p = 0.000, p (Holm) = 0.000
- K = 100: R3 - R1 +92,276 [+67,431, +119,635], p = 0.000, p (Holm) = 0.000
- K = 200: R3 - R1 +58,050 [+40,407, +77,114], p = 0.000, p (Holm) = 0.000
- K = 500: R3 - R1 +24,693 [+14,383, +36,813], p = 0.000, p (Holm) = 0.000

**Phase 8 exit gate** (Mondrian coverage within 2 points of target): {'MVAF': {'0.90': True, '0.95': False}, 'F3': {'0.90': True, '0.95': True}, 'B5': {'0.90': True, '0.95': True}} → **NOT PASSED**.
