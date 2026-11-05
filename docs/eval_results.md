# Evaluation: 50 synthetic sequences x 30 frames, one anomaly each, k=3.0

## Frame-level detection
| method | precision | recall | F1 | false alarms (total) |
|---|---|---|---|---|
| CNN + Mahalanobis (ours) | 0.962 | 1.000 | 0.980 | 2 |
| Pixel-space Mahalanobis | 0.336 | 0.940 | 0.495 | 93 |
| Pixel difference | 0.652 | 0.600 | 0.625 | 16 |

## Recall by anomaly type (anomaly frame found)
| type | CNN + Mahalanobis (ours) | Pixel-space Mahalanobis | Pixel difference |
|---|---|---|---|
| combo | 10/10 | 10/10 | 10/10 |
| dark_patch | 10/10 | 10/10 | 10/10 |
| new_object | 10/10 | 10/10 | 0/10 |
| brightness_drop | 10/10 | 10/10 | 10/10 |
| noise | 10/10 | 7/10 | 0/10 |

## Sensitivity to k (ours)
| k | precision | recall | F1 | false alarms |
|---|---|---|---|---|
| 2.0 | 0.602 | 1.000 | 0.752 | 33 |
| 2.5 | 0.893 | 1.000 | 0.943 | 6 |
| 3.0 | 0.962 | 1.000 | 0.980 | 2 |
| 3.5 | 0.962 | 1.000 | 0.980 | 2 |
| 4.0 | 0.962 | 1.000 | 0.980 | 2 |
| 5.0 | 0.979 | 0.940 | 0.959 | 1 |

## Grad-CAM localization on localized anomalies
(the true box covers about 12% of the frame, so ~12% mass would be chance)
| type | n | peak in box | peak within 16px | centroid hot>=0.5 | centroid hot>=0.7 | cam mass in box |
|---|---|---|---|---|---|---|
| combo | 10 | 100% | 100% | 100% | 100% | 33% |
| dark_patch | 10 | 100% | 100% | 100% | 100% | 34% |
| new_object | 10 | 100% | 100% | 100% | 100% | 39% |

