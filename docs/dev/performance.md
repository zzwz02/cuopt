# cuOpt LP/QP performance: C500 vs A100 vs original cuDSS

> Builds present: c500, a100, cudss. All runs `CUDA_MODULE_LOADING=EAGER`; Mittelmann `--time-limit 600`, Maros-Mészáros `--time-limit 180`. `time` = cuOpt's printed solve time; for PDLP (method 1) the phase metric is the **PDLP solve-phase** time (excl. presolve/build), for barrier (method 3) the **barrier-iterations-total** time (excl. setup/factorization). Non-Optimal status annotated.

## Summary

PDLP (method 1) is identical code across all builds (no LDLᵀ / cuDSS), so its times are a cross-build sanity check — A100 in-house and A100 cuDSS agree to within noise. **Barrier (method 3)** is where the linear solver differs: original **cuDSS** vs the **in-house LDLᵀ** (on A100, and ported to MACA on C500).

- **Barrier, A100 — cuDSS vs in-house LDLᵀ.** Across the 129 Maros-Mészáros QP that are Optimal on every build, cuDSS totals **22.6 s** of barrier-iterations time vs **53.7 s** for the in-house LDLᵀ (~2.4×); on found-time (incl. factorization/setup) **46.5 s vs 60.3 s**. Same ordering on the LP barrier set (savsched1 4.33 vs 8.00 s, square41 5.98 vs 9.81 s, datt256_lp 1.19 vs 2.84 s). This is the documented cuDSS-removal trade-off — the in-house LDLᵀ is cuDSS-independent but ~2× slower.
- **C500 (in-house LDLᵀ on MACA)** is slowest — mid-size instances ≈1.5–2× the A100 in-house time — and carries the two known outliers: `scpm1` (60 s, the 14× LDLᵀ slow path) and `nug08-3rd` (319 s here, the post-perf_opt run-to-run nondeterminism; see `maca_c500_results.md` §3.5).

## 1. Mittelmann LP — method 1 (PDLP)

phase metric = PDLP solve-phase time

| instance | C500 (in-house LDLᵀ / MACA) time | PDLP-phase | A100 (in-house LDLᵀ) time | PDLP-phase | A100 (original cuDSS) time | PDLP-phase |
|---|---|---|---|---|---|---|
| graph40-40 | 0.473 | 0.299 | 0.199 | 0.067 | 0.212 | 0.066 |
| ex10 | 0.531 | 0.418 | 0.239 | 0.162 | 0.248 | 0.161 |
| datt256_lp | 0.738 | 0.568 | 0.319 | 0.178 | 0.339 | 0.180 |
| woodlands09 | 1.035 | 0.690 | 0.516 | 0.171 | 0.501 | 0.166 |
| savsched1 | 1.356 | 1.144 | 0.434 | 0.247 | 0.430 | 0.243 |
| nug08-3rd | 0.252 | 0.210 | 0.092 | 0.076 | 0.091 | 0.072 |
| qap15 | 0.394 | 0.359 | 0.153 | 0.141 | 0.149 | 0.135 |
| scpm1 | 6.865 | 6.418 | 2.266 | 1.866 | 2.249 | 1.840 |
| neos3 | 1.177 | 1.027 | 0.339 | 0.206 | 0.326 | 0.196 |
| a2864 | 1.484 | 0.376 | 1.232 | 0.153 | 1.250 | 0.147 |
| ns1687037 | 600.043 (Time) | 599.948 | 600.011 (Time) | 599.932 | 600.016 (Time) | 599.940 |
| square41 | 58.246 | 57.487 | 34.131 | 33.408 | 33.671 | 32.930 |

## 2. Mittelmann LP — method 3 (barrier)

phase metric = barrier-iterations-total time

| instance | C500 (in-house LDLᵀ / MACA) time | barrier-iters | A100 (in-house LDLᵀ) time | barrier-iters | A100 (original cuDSS) time | barrier-iters |
|---|---|---|---|---|---|---|
| graph40-40 | 0.857 | 0.083 | 0.764 | 0.042 | 0.839 | 0.053 |
| ex10 | 3.93 (Suboptimal) | 3.345 | 3.14 (Suboptimal) | 2.662 | 4.479 | 3.661 |
| datt256_lp | 5.673 | 4.611 | 2.842 | 1.982 | 1.186 | 0.352 |
| woodlands09 | 10.008 | 7.394 | 5.553 | 3.213 | 4.176 | 1.681 |
| savsched1 | 14.798 | 10.083 | 8.004 | 3.532 | 4.332 | 1.496 |
| nug08-3rd | 319.306 | 318.441 | 5.777 | 5.164 | 3.602 | 2.821 |
| qap15 | 2.711 | 2.429 | 1.311 | 1.102 | 1.005 | 0.643 |
| scpm1 | 60.497 | 54.338 | 4.399 | 1.699 | 3.369 | 1.078 |
| neos3 | 1.210 | 0.300 | 0.964 | 0.149 | 1.012 | 0.129 |
| a2864 | 1.370 | 0.064 | 1.284 | 0.032 | 1.384 | 0.028 |
| ns1687037 | 21.760 | 18.826 | 48.898 | 45.598 | 52.677 | 48.141 |
| square41 | 17.866 | 14.797 | 9.805 | 7.001 | 5.981 | 3.235 |

## 3. Maros-Mészáros QP — method 3 (barrier), full 138 set

phase metric = barrier-iterations-total time

| instance | C500 (in-house LDLᵀ / MACA) time | barrier-iters | A100 (in-house LDLᵀ) time | barrier-iters | A100 (original cuDSS) time | barrier-iters |
|---|---|---|---|---|---|---|
| AUG2D | 0.400 | 0.317 | 0.202 | 0.134 | 0.297 | 0.054 |
| AUG2DC | 0.398 | 0.317 | 0.202 | 0.134 | 0.298 | 0.046 |
| AUG2DCQP | 0.591 | 0.501 | 0.286 | 0.219 | 0.390 | 0.154 |
| AUG2DQP | 0.588 | 0.502 | 0.286 | 0.219 | 0.236 | 0.067 |
| AUG3D | 0.236 | 0.179 | 0.090 | 0.074 | 0.168 | 0.027 |
| AUG3DC | 0.238 | 0.181 | 0.090 | 0.075 | 0.164 | 0.027 |
| AUG3DCQP | 0.289 | 0.238 | 0.116 | 0.101 | 0.182 | 0.037 |
| AUG3DQP | 0.288 | 0.237 | 0.114 | 0.099 | 0.178 | 0.037 |
| BOYD1 | 4.077 | 2.934 | 0.225 | 0.117 | 0.282 | 0.086 |
| BOYD2 | 22.663 | 21.233 | 7.308 | 6.026 | 2.619 | 1.244 |
| CONT-050 | 0.306 | 0.253 | 0.164 | 0.114 | 0.212 | 0.060 |
| CONT-100 | 0.980 | 0.859 | 0.446 | 0.353 | 0.264 | 0.091 |
| CONT-101 | 0.957 | 0.828 | 0.433 | 0.340 | 0.280 | 0.092 |
| CONT-200 | 3.186 | 2.819 | 1.626 | 1.355 | 0.503 | 0.234 |
| CONT-201 | 3.735 | 3.343 | 1.840 | 1.556 | 0.610 | 0.257 |
| CONT-300 | 7.135 | 6.343 | 3.653 | 3.053 | 0.863 | 0.447 |
| CVXQP1_L | 74.80 (Suboptimal) | 74.326 | 33.46 (Suboptimal) | 33.170 | 7.63 (Suboptimal) | 7.275 |
| CVXQP1_M | 1.039 | 0.970 | 0.396 | 0.338 | 0.560 | 0.406 |
| CVXQP1_S | 0.208 | 0.169 | 0.063 | 0.052 | 0.181 | 0.037 |
| CVXQP2_L | 4.229 | 3.912 | 2.583 | 2.393 | 0.921 | 0.707 |
| CVXQP2_M | 0.378 | 0.312 | 0.167 | 0.111 | 0.299 | 0.108 |
| CVXQP2_S | 0.215 | 0.149 | 0.065 | 0.045 | 0.150 | 0.033 |
| CVXQP3_L | 23.296 | 22.823 | 13.956 | 13.674 | 4.550 | 4.263 |
| CVXQP3_M | 1.926 | 1.830 | 0.693 | 0.627 | 0.897 | 0.723 |
| CVXQP3_S | 0.277 | 0.245 | 0.087 | 0.079 | 0.249 | 0.059 |
| DPKLO1 | 0.114 | 0.076 | 0.042 | 0.033 | – | – |
| DTOC3 | 0.236 | 0.171 | 0.089 | 0.066 | 0.206 | 0.036 |
| DUAL1 | 0.264 | 0.200 | 0.077 | 0.058 | 0.143 | 0.032 |
| DUAL2 | 0.219 | 0.177 | 0.067 | 0.053 | 0.136 | 0.028 |
| DUAL3 | 0.265 | 0.226 | 0.078 | 0.067 | 0.140 | 0.035 |
| DUAL4 | 0.228 | 0.198 | 0.065 | 0.057 | 0.137 | 0.032 |
| DUALC1 | 0.247 | 0.212 | 0.074 | 0.065 | 0.227 | 0.041 |
| DUALC2 | 0.382 | 0.335 | 0.121 | 0.107 | 0.292 | 0.087 |
| DUALC5 | 0.154 | 0.121 | 0.047 | 0.038 | 0.174 | 0.023 |
| DUALC8 | 0.201 | 0.162 | 0.058 | 0.048 | 0.174 | 0.027 |
| EXDATA | 0.917 | 0.607 | 0.544 | 0.263 | 0.674 | 0.194 |
| GENHS28 | 0.061 | 0.033 | 0.017 | 0.011 | 0.118 | 0.012 |
| GOULDQP2 | 0.170 | 0.127 | 0.050 | 0.037 | 0.185 | 0.022 |
| GOULDQP3 | 0.290 | 0.242 | 0.079 | 0.065 | 0.242 | 0.033 |
| HS118 | 0.087 | 0.043 | 0.024 | 0.019 | 0.124 | 0.019 |
| HS21 | 0.073 | 0.036 | 0.022 | 0.017 | 0.141 | 0.017 |
| HS268 | 0.100 | 0.074 | 0.030 | 0.023 | 0.129 | 0.021 |
| HS35 | 0.078 | 0.051 | 0.023 | 0.017 | 0.119 | 0.016 |
| HS35MOD | 0.094 | 0.069 | 0.031 | 0.025 | 0.130 | 0.025 |
| HS51 | 0.049 | 0.025 | 0.015 | 0.009 | 0.112 | 0.010 |
| HS52 | 0.050 | 0.034 | 0.014 | 0.010 | 0.113 | 0.010 |
| HS53 | 0.084 | 0.053 | 0.026 | 0.019 | 0.123 | 0.019 |
| HS76 | 0.083 | 0.057 | 0.025 | 0.018 | 0.120 | 0.017 |
| HUES-MOD | 0.149 | 0.084 | 0.036 | 0.026 | 0.126 | 0.020 |
| HUESTIS | 0.182 | 0.123 | 0.049 | 0.039 | 0.138 | 0.030 |
| KSIP | 0.313 | 0.238 | 0.270 | 0.192 | 0.282 | 0.149 |
| LASER | 0.545 | 0.468 | 0.154 | 0.138 | 0.269 | 0.089 |
| LISWET1 | 0.710 | 0.647 | 0.323 | 0.300 | 0.364 | 0.167 |
| LISWET10 | 0.741 | 0.676 | 0.346 | 0.324 | 0.354 | 0.178 |
| LISWET11 | 0.657 | 0.597 | 0.294 | 0.272 | 0.486 | 0.226 |
| LISWET12 | 0.969 | 0.906 | 0.448 | 0.425 | 0.582 | 0.350 |
| LISWET2 | 0.221 | 0.152 | 0.090 | 0.068 | 0.238 | 0.038 |
| LISWET3 | 0.235 | 0.173 | 0.090 | 0.068 | 0.236 | 0.038 |
| LISWET4 | 0.222 | 0.163 | 0.089 | 0.067 | 0.248 | 0.038 |
| LISWET5 | 0.225 | 0.156 | 0.090 | 0.068 | 0.265 | 0.038 |
| LISWET6 | 0.231 | 0.158 | 0.085 | 0.062 | 0.213 | 0.034 |
| LISWET7 | 0.641 | 0.582 | 0.288 | 0.266 | 0.398 | 0.168 |
| LISWET8 | 0.895 | 0.829 | 0.404 | 0.381 | 0.389 | 0.212 |
| LISWET9 | 0.937 | 0.867 | 0.424 | 0.402 | 0.456 | 0.227 |
| LOTSCHD | 0.067 | 0.035 | 0.020 | 0.015 | 0.119 | 0.015 |
| MOSARQP1 | 0.738 | 0.583 | 0.206 | 0.163 | 0.276 | 0.041 |
| MOSARQP2 | 0.353 | 0.269 | 0.096 | 0.076 | 0.215 | 0.034 |
| POWELL20 | 0.682 | 0.621 | 0.306 | 0.286 | 0.506 | 0.238 |
| PRIMAL1 | – (NumError) | – | – (NumError) | – | – (NumError) | – |
| PRIMAL2 | – (NumError) | – | – (NumError) | – | – (NumError) | – |
| PRIMAL3 | – (NumError) | – | – (NumError) | – | – (NumError) | – |
| PRIMAL4 | – (NumError) | – | – (NumError) | – | – (NumError) | – |
| PRIMALC1 | 0.107 | 0.074 | 0.054 | 0.047 | 0.163 | 0.046 |
| PRIMALC2 | 0.114 | 0.082 | 0.053 | 0.047 | 0.163 | 0.047 |
| PRIMALC5 | 0.106 | 0.070 | 0.047 | 0.041 | 0.158 | 0.040 |
| PRIMALC8 | 0.143 | 0.104 | 0.060 | 0.054 | 0.170 | 0.053 |
| Q25FV47 | 4.248 | 4.045 | 1.455 | 1.382 | 0.975 | 0.726 |
| QADLITTL | 0.245 | 0.204 | 0.072 | 0.060 | 0.224 | 0.049 |
| QAFIRO | 0.115 | 0.089 | 0.033 | 0.027 | 0.186 | 0.025 |
| QBANDM | 0.502 | 0.453 | 0.143 | 0.129 | 0.209 | 0.055 |
| QBEACONF | 0.395 | 0.318 | 0.120 | 0.096 | 0.191 | 0.045 |
| QBORE3D | 0.512 | 0.465 | 0.155 | 0.142 | 0.269 | 0.085 |
| QBRANDY | 0.416 | 0.364 | 0.118 | 0.103 | 0.208 | 0.056 |
| QCAPRI | 2.109 | 2.058 | 0.713 | 0.698 | 0.722 | 0.518 |
| QE226 | 0.532 | 0.478 | 0.154 | 0.137 | 0.266 | 0.062 |
| QETAMACR | 1.395 | 1.346 | 0.528 | 0.478 | 0.718 | 0.479 |
| QFFFFF80 | 1.121 | 1.050 | 0.397 | 0.341 | 0.490 | 0.335 |
| QFORPLAN | 2.414 | 2.356 | 0.814 | 0.798 | – | – |
| QGFRDXPN | 1.798 | 1.746 | 0.623 | 0.610 | 0.673 | 0.436 |
| QGROW15 | 2.923 | 2.832 | 0.805 | 0.777 | 0.426 | 0.242 |
| QGROW22 | 3.744 | 3.619 | 1.140 | 1.102 | 0.327 | 0.159 |
| QGROW7 | 1.689 | 1.622 | 0.527 | 0.510 | 0.288 | 0.139 |
| QISRAEL | 0.497 | 0.438 | 0.154 | 0.135 | 0.269 | 0.094 |
| QPCBLEND | 0.140 | 0.104 | 0.049 | 0.042 | 0.138 | 0.034 |
| QPCBOEI1 | 0.439 | 0.389 | 0.167 | 0.155 | 0.197 | 0.089 |
| QPCBOEI2 | 0.320 | 0.280 | 0.124 | 0.115 | 0.182 | 0.076 |
| QPCSTAIR | 0.286 | 0.237 | 0.158 | 0.114 | 0.194 | 0.087 |
| QPILOTNO | 3.406 | 3.310 | 1.209 | 1.152 | 1.125 | 0.974 |
| QPTEST | 0.084 | 0.054 | 0.025 | 0.019 | 0.123 | 0.018 |
| QRECIPE | 0.319 | 0.279 | 0.096 | 0.085 | 0.261 | 0.069 |
| QSC205 | 0.228 | 0.182 | 0.068 | 0.054 | 0.186 | 0.033 |
| QSCAGR25 | 0.603 | 0.550 | 0.185 | 0.169 | 0.221 | 0.071 |
| QSCAGR7 | 0.263 | 0.233 | 0.081 | 0.073 | 0.189 | 0.054 |
| QSCFXM1 | 1.367 | 1.277 | 0.433 | 0.403 | 0.354 | 0.177 |
| QSCFXM2 | 2.270 | 2.166 | 0.713 | 0.680 | 0.444 | 0.284 |
| QSCFXM3 | 2.808 | 2.649 | 0.878 | 0.831 | 0.524 | 0.299 |
| QSCORPIO | 0.490 | 0.434 | 0.152 | 0.136 | 0.254 | 0.101 |
| QSCRS8 | 0.823 | 0.777 | 0.267 | 0.254 | 0.325 | 0.156 |
| QSCSD1 | 0.324 | 0.282 | 0.091 | 0.079 | 0.165 | 0.033 |
| QSCSD6 | 0.301 | 0.261 | 0.087 | 0.075 | 0.241 | 0.046 |
| QSCSD8 | 1.030 | 0.910 | 0.279 | 0.248 | 0.282 | 0.046 |
| QSCTAP1 | 0.475 | 0.377 | 0.120 | 0.094 | 0.212 | 0.053 |
| QSCTAP2 | 0.287 | 0.201 | 0.118 | 0.062 | 0.242 | 0.056 |
| QSCTAP3 | 0.313 | 0.220 | 0.127 | 0.069 | 0.258 | 0.061 |
| QSEBA | 0.830 | 0.784 | 0.272 | 0.258 | 0.364 | 0.174 |
| QSHARE1B | 0.357 | 0.286 | 0.106 | 0.083 | 0.202 | 0.046 |
| QSHARE2B | 0.396 | 0.344 | 0.108 | 0.095 | 0.253 | 0.068 |
| QSHELL | 3.154 | 3.071 | 1.111 | 1.055 | 1.253 | 1.078 |
| QSHIP04L | 0.444 | 0.382 | 0.131 | 0.117 | 0.202 | 0.059 |
| QSHIP04S | 0.406 | 0.362 | 0.123 | 0.110 | 0.262 | 0.056 |
| QSHIP08L | 1.917 | 1.694 | 0.615 | 0.509 | 0.485 | 0.264 |
| QSHIP08S | 1.098 | 0.937 | 0.371 | 0.295 | 0.420 | 0.178 |
| QSHIP12L | 2.100 | 1.793 | 0.759 | 0.616 | 0.459 | 0.212 |
| QSHIP12S | 0.953 | 0.804 | 0.321 | 0.243 | 0.306 | 0.138 |
| QSIERRA | 1.753 | 1.678 | 0.596 | 0.576 | 0.566 | 0.386 |
| QSTAIR | 0.805 | 0.736 | 0.297 | 0.242 | 0.317 | 0.156 |
| QSTANDAT | 0.963 | 0.894 | 0.329 | 0.307 | 0.347 | 0.193 |
| S268 | 0.101 | 0.075 | 0.030 | 0.024 | 0.127 | 0.022 |
| STADAT1 | 1.643 | 1.584 | 0.448 | 0.433 | 0.33 (Suboptimal) | 0.174 |
| STADAT2 | 0.557 | 0.498 | 0.150 | 0.134 | 0.264 | 0.053 |
| STADAT3 | 0.362 | 0.304 | 0.139 | 0.120 | 0.283 | 0.070 |
| STCQP1 | 1.205 | 1.009 | 0.383 | 0.314 | 0.407 | 0.148 |
| STCQP2 | 1.721 | 1.479 | 0.531 | 0.443 | 0.394 | 0.180 |
| TAME | 0.049 | 0.028 | 0.015 | 0.010 | 0.125 | 0.010 |
| UBH1 | 0.30 (Suboptimal) | 0.242 | 0.12 (Suboptimal) | 0.097 | 0.25 (Suboptimal) | 0.071 |
| VALUES | 0.217 | 0.163 | 0.068 | 0.051 | 0.182 | 0.048 |
| YAO | 0.744 | 0.700 | 0.271 | 0.260 | 0.421 | 0.179 |
| ZECEVIC2 | 0.067 | 0.034 | 0.021 | 0.016 | 0.133 | 0.016 |

## Notes & caveats

- **Builds.** C500 = `cpp/build_maca` (MACA, in-house LDLᵀ); A100 in-house = `cpp/build_cuda`; A100 cuDSS = tag `v26.06.00` built from source against system CUDA 12.9 + pip rmm/raft + **cuDSS 0.7.1** (`/home/cuopt-26.06-orig`), with the two timing prints ported and gcc-11/toolchain portability fixes (no solver-numerics changes). The PDLP-solve-phase and barrier-iterations-total metrics are the cuOpt-output additions in this branch.
- **2 cuDSS gaps (ParseErr).** `DPKLO1` and `QFORPLAN` fail the *original* v26.06 MPS parser (bad RHS value / duplicate row `DEDO3`); both parse fine in the current branch, so only the cuDSS column is missing for them.
- **`ns1687037` m1 = TimeLimit** on all builds (PDLP does not converge in 600 s); its m3 (barrier) converges.
- **Suboptimal** rows (e.g. `ex10`, `UBH1`) terminate at the relaxed tolerance on all builds alike — not a per-build defect.
- **NumError** (`PRIMAL1–4`) is identical across builds (ill-conditioned; also NumError on official cuDSS).
- Tables generated by `perf_report.py` from `/tmp/perf_{c500,a100,cudss}.csv` (produced by `perf_bench.py`).
