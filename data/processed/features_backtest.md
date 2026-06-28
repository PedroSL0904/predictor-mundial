# Backtest: features historicas (H2H, momentum, WC)

Comparacion con/sin features en WC 2014+2018+2022 (sin lesiones, sin calibrador).


| WC | n | Brier OFF | Brier ON | Delta Brier | Sign OFF | Sign ON | LogLoss OFF | LogLoss ON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 64 | 0.5920 | 0.5815 | -0.0105 | 60.9% | 59.4% | 0.9908 | 0.9748 |
| 2018 | 64 | 0.5889 | 0.5858 | -0.0031 | 59.4% | 56.2% | 0.9882 | 0.9819 |
| 2022 | 64 | 0.6041 | 0.5922 | -0.0119 | 50.0% | 54.7% | 1.0155 | 1.0051 |

**Promedios:** OFF=0.5950, ON=0.5865
Mejora con features: +1.43%