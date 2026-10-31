# Pergeseran jenis error (Fase 5 A2)
Tanggal: 2026-10-31 19:56 · 2,567 citra validasi bersih · pencocokan v2 (tau 0,25) · bootstrap klaster 1000 iterasi

## Komposisi objek GT menurut taksonomi (%, rata-rata tiga seed)

| Model | Kelas | TP | Tidak terdeteksi | Kotak meleset | Salah kelas | Meleset + salah kelas |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | semua | 39.9 | 44.1 | 11.3 | 2.5 | 2.2 |
| EXP-0 | CBB | 44.2 | 35.8 | 10.3 | 6.1 | 3.7 |
| EXP-0 | CBSD | 42.4 | 33.6 | 13.1 | 7.1 | 3.7 |
| EXP-0 | CGM | 39.8 | 46.3 | 9.7 | 3.1 | 1.1 |
| EXP-0 | CMD | 33.6 | 49.3 | 14.4 | 1.4 | 1.3 |
| EXP-0 | HEALTHY | 43.9 | 42.2 | 9.6 | 1.4 | 2.9 |
| EXP-0R | semua | 39.9 | 43.8 | 11.4 | 2.5 | 2.4 |
| EXP-0R | CBB | 40.6 | 39.0 | 10.8 | 6.0 | 3.6 |
| EXP-0R | CBSD | 44.2 | 33.7 | 11.6 | 6.1 | 4.4 |
| EXP-0R | CGM | 40.4 | 46.4 | 9.0 | 2.9 | 1.3 |
| EXP-0R | CMD | 32.5 | 49.9 | 14.0 | 1.9 | 1.7 |
| EXP-0R | HEALTHY | 44.6 | 40.3 | 10.7 | 1.5 | 3.0 |
| EXP-1 | semua | 41.6 | 42.0 | 11.8 | 2.6 | 2.1 |
| EXP-1 | CBB | 40.2 | 37.6 | 10.3 | 7.0 | 4.9 |
| EXP-1 | CBSD | 45.0 | 33.4 | 9.6 | 7.9 | 4.1 |
| EXP-1 | CGM | 40.8 | 44.3 | 10.1 | 3.3 | 1.4 |
| EXP-1 | CMD | 35.8 | 46.9 | 14.7 | 1.5 | 1.2 |
| EXP-1 | HEALTHY | 46.3 | 39.2 | 11.1 | 1.2 | 2.3 |
| EXP-2 | semua | 38.4 | 45.7 | 11.2 | 2.4 | 2.3 |
| EXP-2 | CBB | 42.1 | 38.6 | 10.4 | 5.1 | 3.7 |
| EXP-2 | CBSD | 39.9 | 36.2 | 11.7 | 8.1 | 4.2 |
| EXP-2 | CGM | 37.7 | 48.9 | 8.8 | 3.5 | 1.1 |
| EXP-2 | CMD | 33.5 | 49.8 | 14.0 | 1.1 | 1.6 |
| EXP-2 | HEALTHY | 41.8 | 43.8 | 10.4 | 1.3 | 2.8 |
| EXP-3 | semua | 41.2 | 42.9 | 11.4 | 2.5 | 2.1 |
| EXP-3 | CBB | 41.0 | 38.8 | 10.3 | 6.4 | 3.5 |
| EXP-3 | CBSD | 44.0 | 33.8 | 10.5 | 8.0 | 3.8 |
| EXP-3 | CGM | 41.8 | 44.2 | 10.1 | 2.8 | 1.0 |
| EXP-3 | CMD | 34.4 | 48.6 | 13.8 | 1.6 | 1.6 |
| EXP-3 | HEALTHY | 45.8 | 40.0 | 10.6 | 1.1 | 2.5 |

## False positive per citra (conf ≥ 0,25, rata-rata tiga seed)

| Model | Kelas prediksi | Total | Latar | Sekelas (duplikat/meleset) | Kelas lain |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | semua | 2.150 | 1.152 | 0.797 | 0.201 |
| EXP-0 | CBB | 0.203 | 0.079 | 0.083 | 0.041 |
| EXP-0 | CBSD | 0.145 | 0.050 | 0.068 | 0.027 |
| EXP-0 | CGM | 0.385 | 0.217 | 0.112 | 0.056 |
| EXP-0 | CMD | 0.479 | 0.209 | 0.232 | 0.038 |
| EXP-0 | HEALTHY | 0.937 | 0.596 | 0.302 | 0.039 |
| EXP-0R | semua | 2.123 | 1.130 | 0.790 | 0.203 |
| EXP-0R | CBB | 0.177 | 0.072 | 0.070 | 0.035 |
| EXP-0R | CBSD | 0.163 | 0.052 | 0.074 | 0.037 |
| EXP-0R | CGM | 0.367 | 0.204 | 0.106 | 0.058 |
| EXP-0R | CMD | 0.429 | 0.177 | 0.222 | 0.030 |
| EXP-0R | HEALTHY | 0.986 | 0.624 | 0.318 | 0.044 |
| EXP-1 | semua | 2.530 | 1.391 | 0.917 | 0.223 |
| EXP-1 | CBB | 0.194 | 0.085 | 0.073 | 0.035 |
| EXP-1 | CBSD | 0.172 | 0.056 | 0.083 | 0.032 |
| EXP-1 | CGM | 0.437 | 0.252 | 0.130 | 0.055 |
| EXP-1 | CMD | 0.583 | 0.252 | 0.284 | 0.048 |
| EXP-1 | HEALTHY | 1.144 | 0.745 | 0.346 | 0.053 |
| EXP-2 | semua | 1.938 | 0.990 | 0.751 | 0.196 |
| EXP-2 | CBB | 0.180 | 0.072 | 0.072 | 0.036 |
| EXP-2 | CBSD | 0.125 | 0.038 | 0.062 | 0.025 |
| EXP-2 | CGM | 0.324 | 0.181 | 0.090 | 0.052 |
| EXP-2 | CMD | 0.464 | 0.191 | 0.236 | 0.037 |
| EXP-2 | HEALTHY | 0.845 | 0.508 | 0.290 | 0.046 |
| EXP-3 | semua | 2.352 | 1.274 | 0.865 | 0.213 |
| EXP-3 | CBB | 0.177 | 0.067 | 0.074 | 0.035 |
| EXP-3 | CBSD | 0.168 | 0.056 | 0.084 | 0.028 |
| EXP-3 | CGM | 0.439 | 0.247 | 0.130 | 0.062 |
| EXP-3 | CMD | 0.516 | 0.227 | 0.251 | 0.038 |
| EXP-3 | HEALTHY | 1.052 | 0.676 | 0.327 | 0.049 |

## Kontras (poin persen untuk proporsi GT; FP per citra untuk fp_*)

| Kontras | Kelas | Metrik | Per seed (42 / 123 / 2026) | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-1 − EXP-0R | semua | TP | +1.93 / +3.32 / -0.09 | +1.72 | +1.03 s.d. +2.48 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | tidak terdeteksi | -1.24 / -3.55 / -0.70 | -1.83 | -2.63 s.d. -1.10 | turun secara konsisten |
| EXP-1 − EXP-0R | semua | kotak meleset | -0.48 / +0.70 / +0.92 | +0.38 | -0.20 s.d. +0.95 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | salah kelas | +0.08 / -0.05 / +0.09 | +0.04 | -0.20 s.d. +0.30 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | fp_total | +0.301 / +0.574 / +0.347 | +0.407 | +0.349 s.d. +0.471 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_latar | +0.174 / +0.391 / +0.217 | +0.261 | +0.220 s.d. +0.307 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_sekelas | +0.117 / +0.159 / +0.106 | +0.127 | +0.100 s.d. +0.155 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_kelas_lain | +0.010 / +0.025 / +0.024 | +0.020 | +0.005 s.d. +0.034 | naik secara konsisten |
| EXP-1 − EXP-0R | CBB | TP | -2.42 / -3.63 / +4.84 | -0.40 | -2.77 s.d. +2.01 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | tidak terdeteksi | -0.45 / -0.15 / -3.63 | -1.41 | -3.58 s.d. +0.49 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | kotak meleset | +0.00 / -0.91 / -0.61 | -0.50 | -2.08 s.d. +1.02 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | salah kelas | +1.21 / +1.97 / -0.15 | +1.01 | -0.33 s.d. +2.48 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_total | +0.014 / -0.024 / +0.060 | +0.016 | +0.004 s.d. +0.030 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_latar | +0.003 / +0.000 / +0.036 | +0.013 | +0.004 s.d. +0.024 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_sekelas | +0.008 / -0.010 / +0.012 | +0.003 | -0.003 s.d. +0.009 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_kelas_lain | +0.003 / -0.014 / +0.012 | +0.000 | -0.004 s.d. +0.004 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | TP | +3.04 / +1.01 / -1.74 | +0.77 | -1.75 s.d. +3.14 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | tidak terdeteksi | -0.43 / -0.14 / -0.29 | -0.29 | -2.19 s.d. +1.49 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | kotak meleset | -2.17 / -3.19 / -0.72 | -2.03 | -4.27 s.d. +0.10 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | salah kelas | +1.16 / +2.46 / +1.88 | +1.84 | +0.65 s.d. +3.06 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | fp_total | +0.019 / -0.003 / +0.011 | +0.009 | -0.005 s.d. +0.023 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | fp_latar | +0.007 / +0.002 / +0.002 | +0.004 | -0.004 s.d. +0.012 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | fp_sekelas | +0.009 / +0.006 / +0.014 | +0.009 | +0.003 s.d. +0.016 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | fp_kelas_lain | +0.003 / -0.011 / -0.005 | -0.004 | -0.011 s.d. +0.002 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | TP | -0.74 / +0.69 / +1.48 | +0.48 | -0.77 s.d. +1.82 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | tidak terdeteksi | -1.13 / -1.82 / -3.35 | -2.10 | -3.61 s.d. -0.64 | turun secara konsisten |
| EXP-1 − EXP-0R | CGM | kotak meleset | +1.23 / +0.15 / +1.87 | +1.08 | +0.04 s.d. +2.14 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | salah kelas | +0.69 / +0.10 / +0.54 | +0.44 | -0.30 s.d. +1.18 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | fp_total | -0.000 / +0.064 / +0.146 | +0.070 | +0.049 s.d. +0.091 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | fp_latar | -0.012 / +0.055 / +0.102 | +0.048 | +0.033 s.d. +0.063 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | fp_sekelas | +0.016 / +0.021 / +0.036 | +0.024 | +0.014 s.d. +0.034 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | fp_kelas_lain | -0.004 / -0.011 / +0.008 | -0.002 | -0.009 s.d. +0.004 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CMD | TP | +2.58 / +4.21 / +3.26 | +3.35 | +2.32 s.d. +4.53 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | tidak terdeteksi | -1.98 / -3.36 / -3.75 | -3.03 | -4.44 s.d. -1.75 | turun secara konsisten |
| EXP-1 − EXP-0R | CMD | kotak meleset | +0.64 / -0.14 / +1.56 | +0.68 | -0.27 s.d. +1.81 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CMD | salah kelas | -0.64 / -0.14 / -0.53 | -0.44 | -0.74 s.d. -0.16 | turun secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_total | +0.120 / +0.143 / +0.200 | +0.154 | +0.127 s.d. +0.187 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_latar | +0.063 / +0.061 / +0.099 | +0.075 | +0.058 s.d. +0.093 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_sekelas | +0.042 / +0.061 / +0.082 | +0.062 | +0.046 s.d. +0.079 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_kelas_lain | +0.015 / +0.020 / +0.019 | +0.018 | +0.009 s.d. +0.027 | naik secara konsisten |
| EXP-1 − EXP-0R | HEALTHY | TP | +3.58 / +5.93 / -4.39 | +1.71 | +0.29 s.d. +3.27 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | tidak terdeteksi | -1.02 / -6.08 / +3.84 | -1.09 | -2.62 s.d. +0.46 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | kotak meleset | -2.15 / +2.82 / +0.47 | +0.38 | -0.73 s.d. +1.39 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | salah kelas | -0.12 / -0.96 / +0.03 | -0.35 | -0.70 s.d. -0.04 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_total | +0.148 / +0.394 / -0.069 | +0.158 | +0.113 s.d. +0.212 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_latar | +0.112 / +0.273 / -0.022 | +0.121 | +0.084 s.d. +0.161 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_sekelas | +0.042 / +0.081 / -0.038 | +0.028 | +0.012 s.d. +0.047 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_kelas_lain | -0.006 / +0.041 / -0.009 | +0.008 | +0.001 s.d. +0.016 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | TP | +0.82 / +2.98 / +1.14 | +1.65 | +0.99 s.d. +2.47 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | tidak terdeteksi | -0.48 / -3.90 / -1.91 | -2.09 | -2.93 s.d. -1.30 | turun secara konsisten |
| EXP-1 − EXP-0 | semua | kotak meleset | -0.24 / +1.06 / +0.45 | +0.42 | -0.09 s.d. +0.96 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | salah kelas | -0.06 / +0.11 / +0.23 | +0.09 | -0.15 s.d. +0.36 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | fp_total | +0.213 / +0.587 / +0.341 | +0.380 | +0.324 s.d. +0.436 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_latar | +0.116 / +0.394 / +0.208 | +0.239 | +0.195 s.d. +0.284 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_sekelas | +0.099 / +0.166 / +0.094 | +0.119 | +0.092 s.d. +0.147 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_kelas_lain | -0.001 / +0.027 / +0.039 | +0.022 | +0.009 s.d. +0.035 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | TP | -4.39 / -9.23 / +1.66 | -3.98 | -6.06 s.d. -1.82 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | tidak terdeteksi | +2.27 / +6.05 / -2.72 | +1.87 | -0.25 s.d. +3.99 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | kotak meleset | +0.30 / -1.66 / +1.36 | +0.00 | -1.55 s.d. +1.51 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | salah kelas | +0.61 / +2.27 / -0.15 | +0.91 | -0.28 s.d. +2.12 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_total | -0.001 / -0.058 / +0.031 | -0.009 | -0.022 s.d. +0.004 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_latar | +0.005 / -0.007 / +0.022 | +0.006 | -0.002 s.d. +0.016 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_sekelas | -0.005 / -0.028 / +0.004 | -0.010 | -0.016 s.d. -0.003 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_kelas_lain | -0.001 / -0.023 / +0.005 | -0.006 | -0.011 s.d. -0.001 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | TP | +4.93 / +1.01 / +1.74 | +2.56 | -0.09 s.d. +5.49 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | tidak terdeteksi | -0.43 / +0.58 / -0.72 | -0.19 | -2.61 s.d. +2.04 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | kotak meleset | -3.04 / -4.49 / -3.04 | -3.53 | -6.05 s.d. -1.10 | turun secara konsisten |
| EXP-1 − EXP-0 | CBSD | salah kelas | -0.29 / +2.03 / +0.58 | +0.77 | -0.89 s.d. +2.27 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | fp_total | +0.043 / +0.008 / +0.030 | +0.027 | +0.015 s.d. +0.040 | naik secara konsisten |
| EXP-1 − EXP-0 | CBSD | fp_latar | +0.011 / +0.006 / +0.001 | +0.006 | -0.002 s.d. +0.013 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | fp_sekelas | +0.024 / +0.007 / +0.016 | +0.016 | +0.008 s.d. +0.025 | naik secara konsisten |
| EXP-1 − EXP-0 | CBSD | fp_kelas_lain | +0.008 / -0.005 / +0.013 | +0.005 | +0.000 s.d. +0.011 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | TP | +1.43 / -1.62 / +3.35 | +1.05 | -0.42 s.d. +2.51 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | tidak terdeteksi | -2.85 / +1.92 / -4.97 | -1.97 | -3.52 s.d. -0.41 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | kotak meleset | +1.28 / -1.18 / +1.03 | +0.38 | -0.84 s.d. +1.66 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | salah kelas | -0.25 / +0.59 / +0.39 | +0.25 | -0.53 s.d. +1.03 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_total | +0.035 / -0.012 / +0.133 | +0.052 | +0.032 s.d. +0.072 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_latar | +0.022 / -0.007 / +0.090 | +0.035 | +0.022 s.d. +0.049 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_sekelas | +0.016 / -0.002 / +0.037 | +0.017 | +0.008 s.d. +0.026 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_kelas_lain | -0.003 / -0.004 / +0.005 | -0.001 | -0.007 s.d. +0.005 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | TP | +0.32 / +4.32 / +2.19 | +2.28 | +1.18 s.d. +3.34 | naik secara konsisten |
| EXP-1 − EXP-0 | CMD | tidak terdeteksi | +0.50 / -5.66 / -2.09 | -2.42 | -3.66 s.d. -1.18 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | kotak meleset | -0.67 / +1.24 / +0.21 | +0.26 | -0.80 s.d. +1.29 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | salah kelas | -0.07 / +0.32 / -0.21 | +0.01 | -0.31 s.d. +0.35 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | fp_total | +0.044 / +0.154 / +0.114 | +0.104 | +0.077 s.d. +0.136 | naik secara konsisten |
| EXP-1 − EXP-0 | CMD | fp_latar | +0.009 / +0.068 / +0.051 | +0.043 | +0.025 s.d. +0.061 | naik secara konsisten |
| EXP-1 − EXP-0 | CMD | fp_sekelas | +0.041 / +0.073 / +0.042 | +0.052 | +0.036 s.d. +0.071 | naik secara konsisten |
| EXP-1 − EXP-0 | CMD | fp_kelas_lain | -0.005 / +0.013 / +0.021 | +0.009 | +0.002 s.d. +0.016 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | TP | +1.05 / +7.35 / -1.25 | +2.38 | +0.93 s.d. +3.92 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | tidak terdeteksi | -0.41 / -8.69 / -0.03 | -3.04 | -4.63 s.d. -1.50 | turun secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | kotak meleset | -0.32 / +3.87 / +0.81 | +1.45 | +0.61 s.d. +2.27 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | salah kelas | -0.03 / -1.13 / +0.49 | -0.22 | -0.53 s.d. +0.06 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | fp_total | +0.093 / +0.495 / +0.033 | +0.207 | +0.163 s.d. +0.255 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | fp_latar | +0.070 / +0.334 / +0.043 | +0.149 | +0.114 s.d. +0.187 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | fp_sekelas | +0.023 / +0.115 / -0.005 | +0.044 | +0.029 s.d. +0.061 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | fp_kelas_lain | +0.000 / +0.046 / -0.005 | +0.014 | +0.007 s.d. +0.021 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | TP | -1.01 / -2.78 / -0.81 | -1.53 | -2.18 s.d. -0.86 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | tidak terdeteksi | +1.01 / +3.45 / +0.41 | +1.62 | +0.96 s.d. +2.34 | naik secara konsisten |
| EXP-2 − EXP-0 | semua | kotak meleset | -0.25 / -0.40 / +0.25 | -0.13 | -0.62 s.d. +0.36 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | salah kelas | +0.02 / -0.20 / +0.02 | -0.05 | -0.26 s.d. +0.14 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | fp_total | -0.192 / -0.352 / -0.092 | -0.212 | -0.261 s.d. -0.166 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | fp_latar | -0.125 / -0.280 / -0.078 | -0.161 | -0.201 s.d. -0.126 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | fp_sekelas | -0.053 / -0.062 / -0.024 | -0.046 | -0.069 s.d. -0.026 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | fp_kelas_lain | -0.014 / -0.010 / +0.010 | -0.005 | -0.017 s.d. +0.008 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | TP | +0.45 / -6.66 / -0.30 | -2.17 | -3.75 s.d. -0.45 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | tidak terdeteksi | +2.12 / +6.35 / +0.15 | +2.87 | +0.66 s.d. +5.06 | naik secara konsisten |
| EXP-2 − EXP-0 | CBB | kotak meleset | +0.00 / -1.66 / +2.12 | +0.15 | -1.39 s.d. +1.53 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | salah kelas | -1.66 / -0.15 / -0.91 | -0.91 | -2.11 s.d. +0.16 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_total | +0.005 / -0.077 / +0.004 | -0.023 | -0.033 s.d. -0.013 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_latar | +0.005 / -0.033 / +0.006 | -0.007 | -0.013 s.d. -0.001 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_sekelas | -0.004 / -0.023 / -0.004 | -0.010 | -0.017 s.d. -0.003 | turun secara konsisten |
| EXP-2 − EXP-0 | CBB | fp_kelas_lain | +0.004 / -0.021 / +0.002 | -0.005 | -0.010 s.d. -0.001 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | TP | -1.59 / -4.49 / -1.59 | -2.56 | -4.49 s.d. -0.59 | turun secara konsisten |
| EXP-2 − EXP-0 | CBSD | tidak terdeteksi | +2.61 / +4.78 / +0.43 | +2.61 | +0.37 s.d. +4.70 | naik secara konsisten |
| EXP-2 − EXP-0 | CBSD | kotak meleset | -1.74 / -1.01 / -1.59 | -1.45 | -2.94 s.d. +0.17 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | salah kelas | +0.72 / +0.29 / +1.74 | +0.92 | -0.28 s.d. +2.20 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_total | -0.028 / -0.010 / -0.023 | -0.020 | -0.031 s.d. -0.011 | turun secara konsisten |
| EXP-2 − EXP-0 | CBSD | fp_latar | -0.011 / -0.010 / -0.017 | -0.012 | -0.020 s.d. -0.007 | turun secara konsisten |
| EXP-2 − EXP-0 | CBSD | fp_sekelas | -0.014 / +0.003 / -0.006 | -0.006 | -0.011 s.d. +0.000 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_kelas_lain | -0.004 / -0.003 / -0.001 | -0.002 | -0.007 s.d. +0.002 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | TP | -0.69 / -6.79 / +1.33 | -2.05 | -3.24 s.d. -0.79 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | tidak terdeteksi | -0.05 / +8.71 / -0.94 | +2.58 | +1.21 s.d. +3.93 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | kotak meleset | -0.59 / -2.61 / +0.44 | -0.92 | -1.86 s.d. +0.02 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | salah kelas | +0.94 / +0.98 / -0.69 | +0.41 | -0.11 s.d. +0.98 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_total | -0.035 / -0.158 / +0.008 | -0.062 | -0.080 s.d. -0.043 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_latar | -0.012 / -0.097 / +0.002 | -0.036 | -0.051 s.d. -0.023 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_sekelas | -0.014 / -0.054 / +0.002 | -0.022 | -0.031 s.d. -0.013 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_kelas_lain | -0.009 / -0.007 / +0.004 | -0.004 | -0.011 s.d. +0.002 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | TP | -0.67 / +1.88 / -1.24 | -0.01 | -0.94 s.d. +0.98 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | tidak terdeteksi | +1.77 / -1.31 / +0.99 | +0.48 | -0.58 s.d. +1.54 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | kotak meleset | -0.67 / -0.11 / -0.46 | -0.41 | -1.44 s.d. +0.59 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | salah kelas | -0.64 / -0.67 / +0.25 | -0.35 | -0.63 s.d. -0.12 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_total | -0.045 / +0.035 / -0.036 | -0.015 | -0.036 s.d. +0.005 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_latar | -0.025 / -0.007 / -0.021 | -0.018 | -0.032 s.d. -0.003 | turun secara konsisten |
| EXP-2 − EXP-0 | CMD | fp_sekelas | -0.013 / +0.034 / -0.009 | +0.004 | -0.008 s.d. +0.015 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_kelas_lain | -0.007 / +0.008 / -0.005 | -0.001 | -0.007 s.d. +0.005 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | TP | -1.63 / -3.14 / -1.66 | -2.14 | -3.59 s.d. -0.81 | turun secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | tidak terdeteksi | +0.47 / +3.43 / +0.78 | +1.56 | +0.16 s.d. +3.04 | naik secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | kotak meleset | +0.55 / +1.02 / +0.73 | +0.77 | -0.07 s.d. +1.65 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | salah kelas | +0.20 / -0.61 / +0.09 | -0.11 | -0.39 s.d. +0.13 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | fp_total | -0.089 / -0.143 / -0.045 | -0.092 | -0.133 s.d. -0.055 | turun secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | fp_latar | -0.083 / -0.134 / -0.047 | -0.088 | -0.120 s.d. -0.058 | turun secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | fp_sekelas | -0.007 / -0.021 / -0.008 | -0.012 | -0.027 s.d. +0.002 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | fp_kelas_lain | +0.002 / +0.012 / +0.010 | +0.008 | +0.001 s.d. +0.016 | naik secara konsisten |
| EXP-3 − EXP-1 | semua | TP | -0.67 / -1.40 / +0.85 | -0.41 | -1.04 s.d. +0.18 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | tidak terdeteksi | +0.94 / +2.38 / -0.62 | +0.90 | +0.15 s.d. +1.71 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | kotak meleset | -0.67 / -0.73 / +0.36 | -0.35 | -0.83 s.d. +0.16 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | salah kelas | +0.19 / -0.26 / -0.31 | -0.13 | -0.35 s.d. +0.09 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | fp_total | -0.116 / -0.388 / -0.029 | -0.178 | -0.235 s.d. -0.128 | turun secara konsisten |
| EXP-3 − EXP-1 | semua | fp_latar | -0.067 / -0.257 / -0.028 | -0.117 | -0.159 s.d. -0.080 | turun secara konsisten |
| EXP-3 − EXP-1 | semua | fp_sekelas | -0.058 / -0.113 / +0.017 | -0.051 | -0.079 s.d. -0.026 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | fp_kelas_lain | +0.009 / -0.019 / -0.018 | -0.009 | -0.025 s.d. +0.005 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | TP | +1.82 / +3.33 / -2.87 | +0.76 | -1.09 s.d. +2.51 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | tidak terdeteksi | +1.51 / +0.15 / +1.97 | +1.21 | -0.84 s.d. +3.41 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | kotak meleset | -0.45 / +0.61 / -0.15 | +0.00 | -1.69 s.d. +1.68 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | salah kelas | -0.76 / -2.42 / +1.36 | -0.61 | -1.66 s.d. +0.50 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_total | -0.022 / +0.001 / -0.030 | -0.017 | -0.031 s.d. -0.004 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_latar | -0.014 / -0.018 / -0.023 | -0.018 | -0.029 s.d. -0.008 | turun secara konsisten |
| EXP-3 − EXP-1 | CBB | fp_sekelas | -0.006 / +0.011 / -0.002 | +0.001 | -0.006 s.d. +0.008 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_kelas_lain | -0.002 / +0.008 / -0.005 | +0.000 | -0.004 s.d. +0.005 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | TP | -1.16 / -2.17 / +0.29 | -1.01 | -3.25 s.d. +1.18 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | tidak terdeteksi | +0.43 / +2.17 / -1.45 | +0.39 | -1.66 s.d. +2.52 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | kotak meleset | -0.72 / +2.32 / +1.01 | +0.87 | -0.84 s.d. +2.79 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | salah kelas | +1.01 / -1.16 / +0.43 | +0.10 | -1.24 s.d. +1.43 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_total | -0.004 / -0.012 / +0.004 | -0.004 | -0.016 s.d. +0.008 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_latar | +0.004 / -0.008 / +0.005 | +0.000 | -0.007 s.d. +0.008 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_sekelas | -0.005 / +0.002 / +0.005 | +0.001 | -0.007 s.d. +0.007 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_kelas_lain | -0.002 / -0.006 / -0.005 | -0.005 | -0.011 s.d. +0.001 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | TP | -0.20 / +2.07 / +1.03 | +0.97 | -0.08 s.d. +2.09 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | tidak terdeteksi | +1.08 / -2.12 / +0.74 | -0.10 | -1.34 s.d. +1.10 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | kotak meleset | -0.89 / +1.53 / -0.59 | +0.02 | -0.88 s.d. +0.87 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | salah kelas | +0.15 / -0.64 / -1.08 | -0.52 | -1.15 s.d. +0.05 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_total | +0.016 / +0.029 / -0.040 | +0.002 | -0.017 s.d. +0.021 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_latar | +0.001 / +0.019 / -0.037 | -0.006 | -0.019 s.d. +0.008 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_sekelas | +0.003 / +0.000 / -0.002 | +0.000 | -0.008 s.d. +0.009 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_kelas_lain | +0.012 / +0.011 / -0.000 | +0.007 | +0.000 s.d. +0.014 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | TP | +0.39 / -3.29 / -1.42 | -1.44 | -2.61 s.d. -0.36 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | tidak terdeteksi | -0.67 / +4.53 / +1.31 | +1.72 | +0.39 s.d. +3.11 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | kotak meleset | -0.71 / -1.34 / -0.53 | -0.86 | -1.88 s.d. +0.27 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | salah kelas | +0.28 / -0.11 / +0.32 | +0.17 | -0.16 s.d. +0.46 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | fp_total | +0.009 / -0.145 / -0.063 | -0.067 | -0.093 s.d. -0.043 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | fp_latar | +0.006 / -0.063 / -0.016 | -0.024 | -0.040 s.d. -0.010 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | fp_sekelas | -0.002 / -0.067 / -0.030 | -0.033 | -0.048 s.d. -0.019 | turun secara konsisten |
| EXP-3 − EXP-1 | CMD | fp_kelas_lain | +0.004 / -0.014 / -0.018 | -0.009 | -0.016 s.d. -0.003 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | TP | -2.21 / -2.65 / +3.43 | -0.47 | -1.80 s.d. +0.87 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | tidak terdeteksi | +2.18 / +3.75 / -3.34 | +0.86 | -0.62 s.d. +2.30 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | kotak meleset | -0.55 / -2.41 / +1.63 | -0.45 | -1.22 s.d. +0.39 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | salah kelas | +0.15 / +0.44 / -0.84 | -0.09 | -0.35 s.d. +0.18 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_total | -0.115 / -0.261 / +0.101 | -0.092 | -0.135 s.d. -0.053 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_latar | -0.064 / -0.186 / +0.044 | -0.069 | -0.104 s.d. -0.037 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_sekelas | -0.048 / -0.058 / +0.047 | -0.020 | -0.037 s.d. -0.005 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_kelas_lain | -0.003 / -0.018 / +0.010 | -0.003 | -0.011 s.d. +0.004 | tidak terbukti berbeda |
