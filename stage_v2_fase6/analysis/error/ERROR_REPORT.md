# Pergeseran jenis error (Fase 5 A2)
Tanggal: 2026-11-02 20:30 · 1,282 citra test · pencocokan v2 (tau 0,25) · bootstrap klaster 1000 iterasi

## Komposisi objek GT menurut taksonomi (%, rata-rata tiga seed)

| Model | Kelas | TP | Tidak terdeteksi | Kotak meleset | Salah kelas | Meleset + salah kelas |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-0 | semua | 37.7 | 47.3 | 10.5 | 2.7 | 1.8 |
| EXP-0 | CBB | 34.8 | 45.1 | 12.6 | 5.2 | 2.3 |
| EXP-0 | CBSD | 39.8 | 34.7 | 9.1 | 12.9 | 3.4 |
| EXP-0 | CGM | 38.4 | 46.7 | 8.9 | 3.7 | 2.3 |
| EXP-0 | CMD | 31.3 | 52.3 | 13.0 | 2.1 | 1.2 |
| EXP-0 | HEALTHY | 42.2 | 46.5 | 9.3 | 0.4 | 1.6 |
| EXP-0R | semua | 38.1 | 46.9 | 10.3 | 2.7 | 2.1 |
| EXP-0R | CBB | 32.8 | 47.8 | 10.1 | 6.1 | 3.1 |
| EXP-0R | CBSD | 40.2 | 33.9 | 9.3 | 12.2 | 4.3 |
| EXP-0R | CGM | 37.9 | 46.8 | 9.8 | 3.5 | 2.1 |
| EXP-0R | CMD | 30.2 | 53.3 | 12.0 | 2.2 | 2.4 |
| EXP-0R | HEALTHY | 44.5 | 44.3 | 9.5 | 0.4 | 1.3 |
| EXP-1 | semua | 39.2 | 45.7 | 10.6 | 2.9 | 1.7 |
| EXP-1 | CBB | 31.6 | 45.4 | 12.5 | 7.0 | 3.5 |
| EXP-1 | CBSD | 39.0 | 37.0 | 9.0 | 11.3 | 3.7 |
| EXP-1 | CGM | 38.3 | 45.3 | 9.8 | 4.9 | 1.7 |
| EXP-1 | CMD | 32.6 | 50.6 | 13.4 | 2.1 | 1.4 |
| EXP-1 | HEALTHY | 45.6 | 43.8 | 8.9 | 0.4 | 1.3 |
| EXP-2 | semua | 36.6 | 49.3 | 10.0 | 2.6 | 1.4 |
| EXP-2 | CBB | 34.8 | 46.1 | 10.6 | 5.6 | 2.9 |
| EXP-2 | CBSD | 36.8 | 37.3 | 9.2 | 13.7 | 3.0 |
| EXP-2 | CGM | 35.8 | 49.9 | 9.0 | 3.5 | 1.8 |
| EXP-2 | CMD | 31.9 | 51.8 | 13.4 | 1.9 | 1.0 |
| EXP-2 | HEALTHY | 40.6 | 49.8 | 8.2 | 0.4 | 1.1 |
| EXP-3 | semua | 38.8 | 46.8 | 9.7 | 2.9 | 1.8 |
| EXP-3 | CBB | 32.3 | 48.1 | 11.9 | 5.6 | 2.2 |
| EXP-3 | CBSD | 38.3 | 35.9 | 8.0 | 14.9 | 2.9 |
| EXP-3 | CGM | 39.2 | 46.0 | 9.3 | 3.7 | 1.8 |
| EXP-3 | CMD | 32.5 | 51.8 | 11.1 | 2.6 | 2.0 |
| EXP-3 | HEALTHY | 44.2 | 45.2 | 8.8 | 0.4 | 1.4 |

## False positive per citra (conf ≥ 0,25, rata-rata tiga seed)

| Model | Kelas prediksi | Total | Latar | Sekelas (duplikat/meleset) | Kelas lain |
| --- | --- | --- | --- | --- | --- |
| EXP-0 | semua | 2.088 | 1.106 | 0.730 | 0.252 |
| EXP-0 | CBB | 0.193 | 0.081 | 0.071 | 0.041 |
| EXP-0 | CBSD | 0.144 | 0.045 | 0.064 | 0.035 |
| EXP-0 | CGM | 0.384 | 0.229 | 0.097 | 0.058 |
| EXP-0 | CMD | 0.459 | 0.214 | 0.213 | 0.032 |
| EXP-0 | HEALTHY | 0.908 | 0.538 | 0.284 | 0.086 |
| EXP-0R | semua | 2.059 | 1.069 | 0.731 | 0.259 |
| EXP-0R | CBB | 0.159 | 0.062 | 0.055 | 0.041 |
| EXP-0R | CBSD | 0.149 | 0.044 | 0.067 | 0.038 |
| EXP-0R | CGM | 0.366 | 0.203 | 0.096 | 0.067 |
| EXP-0R | CMD | 0.411 | 0.188 | 0.197 | 0.026 |
| EXP-0R | HEALTHY | 0.974 | 0.572 | 0.316 | 0.087 |
| EXP-1 | semua | 2.417 | 1.317 | 0.837 | 0.263 |
| EXP-1 | CBB | 0.175 | 0.073 | 0.065 | 0.038 |
| EXP-1 | CBSD | 0.178 | 0.057 | 0.077 | 0.043 |
| EXP-1 | CGM | 0.429 | 0.264 | 0.109 | 0.056 |
| EXP-1 | CMD | 0.521 | 0.249 | 0.239 | 0.032 |
| EXP-1 | HEALTHY | 1.114 | 0.674 | 0.346 | 0.094 |
| EXP-2 | semua | 1.878 | 0.944 | 0.696 | 0.238 |
| EXP-2 | CBB | 0.169 | 0.063 | 0.062 | 0.044 |
| EXP-2 | CBSD | 0.132 | 0.037 | 0.067 | 0.028 |
| EXP-2 | CGM | 0.302 | 0.173 | 0.077 | 0.052 |
| EXP-2 | CMD | 0.446 | 0.199 | 0.221 | 0.027 |
| EXP-2 | HEALTHY | 0.828 | 0.472 | 0.270 | 0.087 |
| EXP-3 | semua | 2.290 | 1.219 | 0.800 | 0.271 |
| EXP-3 | CBB | 0.170 | 0.065 | 0.066 | 0.038 |
| EXP-3 | CBSD | 0.167 | 0.051 | 0.083 | 0.032 |
| EXP-3 | CGM | 0.458 | 0.268 | 0.114 | 0.076 |
| EXP-3 | CMD | 0.459 | 0.215 | 0.215 | 0.029 |
| EXP-3 | HEALTHY | 1.035 | 0.620 | 0.321 | 0.094 |

## Kontras (poin persen untuk proporsi GT; FP per citra untuk fp_*)

| Kontras | Kelas | Metrik | Per seed (42 / 123 / 2026) | Rata-rata | IK 95% | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-1 − EXP-0R | semua | TP | -0.43 / +2.57 / +1.00 | +1.05 | +0.23 s.d. +1.89 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | tidak terdeteksi | +0.94 / -2.93 / -1.63 | -1.21 | -2.28 s.d. -0.14 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | kotak meleset | -0.16 / +0.79 / +0.31 | +0.31 | -0.46 s.d. +1.04 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | salah kelas | +0.10 / +0.41 / +0.12 | +0.21 | -0.15 s.d. +0.61 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | semua | fp_total | +0.236 / +0.527 / +0.312 | +0.358 | +0.283 s.d. +0.433 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_latar | +0.149 / +0.381 / +0.215 | +0.249 | +0.195 s.d. +0.309 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_sekelas | +0.080 / +0.146 / +0.090 | +0.106 | +0.070 s.d. +0.141 | naik secara konsisten |
| EXP-1 − EXP-0R | semua | fp_kelas_lain | +0.007 / -0.001 / +0.006 | +0.004 | -0.018 s.d. +0.028 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | TP | -0.59 / -2.95 / -0.29 | -1.28 | -3.72 s.d. +1.38 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | tidak terdeteksi | -3.54 / -1.77 / -1.77 | -2.36 | -5.54 s.d. +0.68 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | kotak meleset | +3.54 / +2.36 / +1.18 | +2.36 | +0.31 s.d. +4.50 | naik secara konsisten |
| EXP-1 − EXP-0R | CBB | salah kelas | +0.00 / +1.77 / +0.88 | +0.88 | -0.51 s.d. +2.37 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_total | +0.016 / -0.018 / +0.052 | +0.017 | -0.002 s.d. +0.036 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_latar | -0.001 / -0.002 / +0.034 | +0.011 | -0.001 s.d. +0.024 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_sekelas | +0.012 / -0.004 / +0.020 | +0.009 | +0.001 s.d. +0.018 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBB | fp_kelas_lain | +0.004 / -0.012 / -0.002 | -0.003 | -0.011 s.d. +0.004 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | TP | -2.42 / +0.30 / -1.52 | -1.21 | -4.23 s.d. +1.52 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | tidak terdeteksi | +3.33 / +0.61 / +5.15 | +3.03 | +0.61 s.d. +5.54 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | kotak meleset | +0.61 / -2.12 / +0.61 | -0.30 | -2.38 s.d. +1.81 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | salah kelas | +0.61 / +0.91 / -4.24 | -0.91 | -3.03 s.d. +1.29 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CBSD | fp_total | +0.027 / +0.037 / +0.021 | +0.028 | +0.013 s.d. +0.045 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | fp_latar | +0.016 / +0.016 / +0.007 | +0.013 | +0.005 s.d. +0.022 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | fp_sekelas | +0.003 / +0.026 / +0.003 | +0.011 | +0.001 s.d. +0.021 | naik secara konsisten |
| EXP-1 − EXP-0R | CBSD | fp_kelas_lain | +0.009 / -0.005 / +0.011 | +0.005 | -0.002 s.d. +0.011 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | TP | -1.55 / +0.52 / +2.17 | +0.38 | -0.89 s.d. +1.71 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | tidak terdeteksi | +1.65 / -1.14 / -4.96 | -1.48 | -3.40 s.d. +0.33 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | kotak meleset | -0.52 / -0.62 / +1.34 | +0.07 | -1.11 s.d. +1.11 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CGM | salah kelas | +0.41 / +2.58 / +1.34 | +1.45 | +0.32 s.d. +3.08 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | fp_total | +0.012 / +0.066 / +0.112 | +0.063 | +0.032 s.d. +0.093 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | fp_latar | +0.021 / +0.070 / +0.094 | +0.062 | +0.039 s.d. +0.085 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | fp_sekelas | +0.001 / +0.012 / +0.025 | +0.012 | +0.001 s.d. +0.024 | naik secara konsisten |
| EXP-1 − EXP-0R | CGM | fp_kelas_lain | -0.010 / -0.016 / -0.007 | -0.011 | -0.023 s.d. -0.001 | turun secara konsisten |
| EXP-1 − EXP-0R | CMD | TP | +1.66 / +2.14 / +3.52 | +2.44 | +1.29 s.d. +3.75 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | tidak terdeteksi | -2.14 / -0.55 / -5.46 | -2.72 | -4.43 s.d. -0.96 | turun secara konsisten |
| EXP-1 − EXP-0R | CMD | kotak meleset | +1.73 / +0.55 / +2.00 | +1.43 | +0.03 s.d. +2.98 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | salah kelas | +0.00 / -0.48 / +0.00 | -0.16 | -0.72 s.d. +0.35 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | CMD | fp_total | +0.077 / +0.088 / +0.165 | +0.110 | +0.075 s.d. +0.148 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_latar | +0.039 / +0.053 / +0.091 | +0.061 | +0.038 s.d. +0.087 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_sekelas | +0.037 / +0.028 / +0.062 | +0.042 | +0.023 s.d. +0.064 | naik secara konsisten |
| EXP-1 − EXP-0R | CMD | fp_kelas_lain | +0.002 / +0.007 / +0.011 | +0.007 | -0.001 s.d. +0.016 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | TP | -1.05 / +5.19 / -0.75 | +1.13 | -0.56 s.d. +2.89 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | tidak terdeteksi | +3.19 / -6.28 / +1.65 | -0.48 | -2.65 s.d. +1.55 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | kotak meleset | -2.09 / +1.85 / -1.60 | -0.62 | -2.06 s.d. +0.73 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | salah kelas | -0.05 / -0.30 / +0.20 | -0.05 | -0.34 s.d. +0.23 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_total | +0.105 / +0.354 / -0.037 | +0.140 | +0.082 s.d. +0.205 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_latar | +0.074 / +0.243 / -0.011 | +0.102 | +0.061 s.d. +0.150 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_sekelas | +0.027 / +0.084 / -0.020 | +0.031 | +0.006 s.d. +0.057 | tidak terbukti berbeda |
| EXP-1 − EXP-0R | HEALTHY | fp_kelas_lain | +0.003 / +0.027 / -0.007 | +0.008 | -0.007 s.d. +0.025 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | TP | -0.43 / +3.26 / +1.41 | +1.41 | +0.51 s.d. +2.25 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | tidak terdeteksi | +0.81 / -4.40 / -1.36 | -1.65 | -2.64 s.d. -0.69 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | kotak meleset | -0.43 / +1.02 / -0.28 | +0.10 | -0.57 s.d. +0.79 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | salah kelas | +0.18 / +0.31 / +0.18 | +0.22 | -0.20 s.d. +0.69 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | semua | fp_total | +0.112 / +0.576 / +0.300 | +0.329 | +0.240 s.d. +0.412 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_latar | +0.031 / +0.399 / +0.203 | +0.211 | +0.148 s.d. +0.275 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_sekelas | +0.067 / +0.160 / +0.094 | +0.107 | +0.070 s.d. +0.146 | naik secara konsisten |
| EXP-1 − EXP-0 | semua | fp_kelas_lain | +0.014 / +0.016 / +0.002 | +0.011 | -0.015 s.d. +0.037 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | TP | -4.13 / -6.19 / +0.59 | -3.24 | -5.90 s.d. -0.72 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | tidak terdeteksi | +2.95 / +0.59 / -2.65 | +0.29 | -2.35 s.d. +3.05 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | kotak meleset | +0.29 / +0.00 / -0.59 | -0.10 | -2.56 s.d. +2.23 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | salah kelas | -0.59 / +3.54 / +2.36 | +1.77 | -0.00 s.d. +3.66 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_total | -0.013 / -0.075 / +0.034 | -0.018 | -0.039 s.d. +0.003 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_latar | -0.015 / -0.029 / +0.020 | -0.008 | -0.021 s.d. +0.005 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_sekelas | -0.003 / -0.022 / +0.005 | -0.007 | -0.018 s.d. +0.004 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBB | fp_kelas_lain | +0.005 / -0.024 / +0.009 | -0.003 | -0.015 s.d. +0.007 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | TP | -0.61 / -0.61 / -1.21 | -0.81 | -3.85 s.d. +2.23 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | tidak terdeteksi | +0.61 / -1.82 / +7.88 | +2.22 | -0.20 s.d. +4.85 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | kotak meleset | +0.30 / +0.61 / -1.21 | -0.10 | -2.15 s.d. +1.84 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | salah kelas | +0.00 / -0.30 / -4.55 | -1.62 | -4.47 s.d. +1.05 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CBSD | fp_total | +0.047 / +0.036 / +0.019 | +0.034 | +0.016 s.d. +0.053 | naik secara konsisten |
| EXP-1 − EXP-0 | CBSD | fp_latar | +0.020 / +0.015 / +0.003 | +0.012 | +0.003 s.d. +0.021 | naik secara konsisten |
| EXP-1 − EXP-0 | CBSD | fp_sekelas | +0.016 / +0.020 / +0.004 | +0.014 | +0.001 s.d. +0.026 | naik secara konsisten |
| EXP-1 − EXP-0 | CBSD | fp_kelas_lain | +0.011 / +0.001 / +0.012 | +0.008 | -0.000 s.d. +0.016 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | TP | -2.07 / -1.34 / +2.89 | -0.17 | -1.66 s.d. +1.35 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | tidak terdeteksi | +1.24 / -0.21 / -5.06 | -1.34 | -3.48 s.d. +0.67 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | kotak meleset | +1.14 / -0.52 / +2.17 | +0.93 | -0.38 s.d. +2.33 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | salah kelas | +0.52 / +2.58 / +0.62 | +1.24 | -0.20 s.d. +3.19 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_total | +0.034 / -0.009 / +0.112 | +0.046 | +0.018 s.d. +0.074 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_latar | +0.021 / +0.006 / +0.079 | +0.035 | +0.015 s.d. +0.058 | naik secara konsisten |
| EXP-1 − EXP-0 | CGM | fp_sekelas | +0.009 / -0.004 / +0.030 | +0.012 | -0.001 s.d. +0.024 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CGM | fp_kelas_lain | +0.003 / -0.011 / +0.002 | -0.002 | -0.013 s.d. +0.008 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | TP | +0.07 / +2.83 / +0.90 | +1.27 | -0.19 s.d. +2.75 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | tidak terdeteksi | -0.97 / -1.94 / -2.35 | -1.75 | -3.31 s.d. -0.19 | turun secara konsisten |
| EXP-1 − EXP-0 | CMD | kotak meleset | +0.14 / +0.21 / +0.69 | +0.35 | -1.09 s.d. +1.84 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | salah kelas | +0.41 / -0.76 / +0.14 | -0.07 | -0.54 s.d. +0.35 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | fp_total | -0.017 / +0.121 / +0.083 | +0.062 | +0.027 s.d. +0.097 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | fp_latar | -0.026 / +0.091 / +0.041 | +0.036 | +0.012 s.d. +0.059 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | CMD | fp_sekelas | +0.012 / +0.024 / +0.042 | +0.026 | +0.005 s.d. +0.048 | naik secara konsisten |
| EXP-1 − EXP-0 | CMD | fp_kelas_lain | -0.004 / +0.005 / -0.001 | +0.000 | -0.011 s.d. +0.013 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | TP | +0.65 / +8.03 / +1.65 | +3.44 | +1.87 s.d. +4.98 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | tidak terdeteksi | +1.55 / -9.48 / -0.15 | -2.69 | -4.42 s.d. -0.88 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | kotak meleset | -1.85 / +2.59 / -1.95 | -0.40 | -1.43 s.d. +0.64 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | salah kelas | +0.00 / -0.45 / +0.40 | -0.02 | -0.38 s.d. +0.27 | tidak terbukti berbeda |
| EXP-1 − EXP-0 | HEALTHY | fp_total | +0.062 / +0.502 / +0.052 | +0.206 | +0.142 s.d. +0.276 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | fp_latar | +0.031 / +0.316 / +0.060 | +0.136 | +0.086 s.d. +0.192 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | fp_sekelas | +0.032 / +0.141 / +0.012 | +0.062 | +0.039 s.d. +0.089 | naik secara konsisten |
| EXP-1 − EXP-0 | HEALTHY | fp_kelas_lain | -0.001 / +0.045 / -0.020 | +0.008 | -0.007 s.d. +0.022 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | TP | -0.57 / -2.73 / -0.16 | -1.15 | -2.16 s.d. -0.16 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | tidak terdeteksi | +1.55 / +3.01 / +1.38 | +1.98 | +0.81 s.d. +3.07 | naik secara konsisten |
| EXP-2 − EXP-0 | semua | kotak meleset | -0.41 / -0.08 / -0.84 | -0.45 | -1.16 s.d. +0.35 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | salah kelas | -0.02 / -0.12 / +0.04 | -0.03 | -0.40 s.d. +0.33 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | fp_total | -0.222 / -0.264 / -0.144 | -0.210 | -0.286 s.d. -0.138 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | fp_latar | -0.154 / -0.244 / -0.089 | -0.162 | -0.218 s.d. -0.112 | turun secara konsisten |
| EXP-2 − EXP-0 | semua | fp_sekelas | -0.046 / -0.004 / -0.051 | -0.034 | -0.070 s.d. +0.002 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | semua | fp_kelas_lain | -0.023 / -0.016 / -0.005 | -0.014 | -0.040 s.d. +0.010 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | TP | +0.88 / -3.54 / +2.65 | +0.00 | -2.68 s.d. +2.62 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | tidak terdeteksi | +2.95 / +2.65 / -2.65 | +0.98 | -1.93 s.d. +3.77 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | kotak meleset | -2.65 / -2.95 / -0.29 | -1.97 | -4.11 s.d. +0.20 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | salah kelas | -1.18 / +1.77 / +0.59 | +0.39 | -1.01 s.d. +1.86 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_total | -0.003 / -0.080 / +0.010 | -0.024 | -0.042 s.d. -0.007 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_latar | -0.011 / -0.047 / +0.005 | -0.018 | -0.029 s.d. -0.007 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBB | fp_sekelas | -0.008 / -0.014 / -0.007 | -0.010 | -0.018 s.d. -0.001 | turun secara konsisten |
| EXP-2 − EXP-0 | CBB | fp_kelas_lain | +0.016 / -0.020 / +0.012 | +0.003 | -0.005 s.d. +0.010 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | TP | -4.24 / -0.30 / -4.55 | -3.03 | -5.63 s.d. -0.53 | turun secara konsisten |
| EXP-2 − EXP-0 | CBSD | tidak terdeteksi | +0.00 / +2.73 / +4.85 | +2.53 | +0.11 s.d. +5.11 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | kotak meleset | +2.42 / -2.42 / +0.30 | +0.10 | -1.71 s.d. +1.80 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | salah kelas | +2.12 / +0.91 / -0.61 | +0.81 | -1.11 s.d. +2.96 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_total | -0.006 / +0.002 / -0.030 | -0.011 | -0.027 s.d. +0.003 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_latar | -0.005 / -0.009 / -0.010 | -0.008 | -0.017 s.d. +0.001 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_sekelas | +0.002 / +0.016 / -0.008 | +0.003 | -0.006 s.d. +0.012 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CBSD | fp_kelas_lain | -0.003 / -0.006 / -0.012 | -0.007 | -0.014 s.d. +0.001 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | TP | -1.45 / -6.51 / +0.10 | -2.62 | -4.28 s.d. -0.94 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | tidak terdeteksi | +3.00 / +6.40 / +0.21 | +3.20 | +1.15 s.d. +5.59 | naik secara konsisten |
| EXP-2 − EXP-0 | CGM | kotak meleset | -0.41 / -0.83 / +1.55 | +0.10 | -1.28 s.d. +1.58 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | salah kelas | +0.21 / +0.10 / -0.72 | -0.14 | -1.61 s.d. +1.42 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_total | -0.085 / -0.165 / +0.005 | -0.082 | -0.113 s.d. -0.054 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CGM | fp_latar | -0.058 / -0.104 / -0.005 | -0.056 | -0.078 s.d. -0.038 | turun secara konsisten |
| EXP-2 − EXP-0 | CGM | fp_sekelas | -0.015 / -0.044 / -0.002 | -0.020 | -0.034 s.d. -0.007 | turun secara konsisten |
| EXP-2 − EXP-0 | CGM | fp_kelas_lain | -0.012 / -0.018 / +0.012 | -0.006 | -0.016 s.d. +0.004 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | TP | +1.59 / +0.62 / -0.41 | +0.60 | -0.90 s.d. +1.92 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | tidak terdeteksi | -2.35 / -1.24 / +2.00 | -0.53 | -2.41 s.d. +1.36 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | kotak meleset | +1.24 / +1.87 / -2.07 | +0.35 | -1.20 s.d. +1.95 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | salah kelas | -0.48 / -0.69 / +0.35 | -0.28 | -0.86 s.d. +0.28 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_total | -0.033 / +0.024 / -0.029 | -0.012 | -0.047 s.d. +0.020 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_latar | -0.012 / -0.016 / -0.016 | -0.015 | -0.038 s.d. +0.006 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_sekelas | -0.012 / +0.044 / -0.009 | +0.008 | -0.012 s.d. +0.030 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | CMD | fp_kelas_lain | -0.008 / -0.003 / -0.005 | -0.005 | -0.016 s.d. +0.004 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | TP | -1.35 / -3.59 / +0.15 | -1.60 | -3.60 s.d. +0.41 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | tidak terdeteksi | +3.69 / +4.54 / +1.60 | +3.28 | +1.12 s.d. +5.35 | naik secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | kotak meleset | -1.70 / -0.25 / -1.40 | -1.11 | -2.32 s.d. +0.13 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | salah kelas | +0.05 / -0.30 / +0.20 | -0.02 | -0.24 s.d. +0.19 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | fp_total | -0.095 / -0.044 / -0.101 | -0.080 | -0.140 s.d. -0.026 | turun secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | fp_latar | -0.068 / -0.069 / -0.062 | -0.066 | -0.113 s.d. -0.029 | turun secara konsisten |
| EXP-2 − EXP-0 | HEALTHY | fp_sekelas | -0.012 / -0.006 / -0.026 | -0.015 | -0.038 s.d. +0.010 | tidak terbukti berbeda |
| EXP-2 − EXP-0 | HEALTHY | fp_kelas_lain | -0.015 / +0.031 / -0.013 | +0.001 | -0.019 s.d. +0.019 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | TP | -0.06 / -1.36 / +0.24 | -0.39 | -1.26 s.d. +0.52 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | tidak terdeteksi | +0.94 / +2.40 / +0.04 | +1.13 | +0.23 s.d. +1.99 | naik secara konsisten |
| EXP-3 − EXP-1 | semua | kotak meleset | -1.36 / -1.24 / -0.04 | -0.88 | -1.50 s.d. -0.26 | turun secara konsisten |
| EXP-3 − EXP-1 | semua | salah kelas | +0.02 / -0.02 / +0.14 | +0.05 | -0.31 s.d. +0.41 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | fp_total | -0.044 / -0.298 / -0.040 | -0.127 | -0.197 s.d. -0.058 | turun secara konsisten |
| EXP-3 − EXP-1 | semua | fp_latar | -0.040 / -0.179 / -0.074 | -0.098 | -0.153 s.d. -0.044 | turun secara konsisten |
| EXP-3 − EXP-1 | semua | fp_sekelas | -0.012 / -0.118 / +0.020 | -0.037 | -0.069 s.d. -0.007 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | semua | fp_kelas_lain | +0.008 / -0.001 / +0.015 | +0.007 | -0.013 s.d. +0.028 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | TP | +1.77 / +0.88 / -0.59 | +0.69 | -1.47 s.d. +2.95 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | tidak terdeteksi | +4.72 / +2.65 / +0.59 | +2.65 | -0.09 s.d. +5.46 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | kotak meleset | -2.65 / -1.18 / +2.06 | -0.59 | -2.90 s.d. +1.74 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | salah kelas | -2.36 / -1.18 / -0.59 | -1.38 | -3.28 s.d. +0.11 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_total | -0.012 / +0.002 / -0.007 | -0.005 | -0.020 s.d. +0.010 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_latar | -0.007 / -0.013 / -0.002 | -0.007 | -0.017 s.d. +0.002 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_sekelas | -0.004 / +0.009 / -0.001 | +0.001 | -0.006 s.d. +0.009 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBB | fp_kelas_lain | -0.001 / +0.007 / -0.005 | +0.001 | -0.007 s.d. +0.008 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | TP | +2.42 / -5.15 / +0.61 | -0.71 | -3.64 s.d. +2.35 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | tidak terdeteksi | -2.42 / +3.33 / -4.24 | -1.11 | -3.88 s.d. +1.33 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | kotak meleset | -0.30 / -2.73 / +0.00 | -1.01 | -2.81 s.d. +0.73 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | salah kelas | +1.82 / +4.55 / +4.55 | +3.64 | +1.41 s.d. +6.15 | naik secara konsisten |
| EXP-3 − EXP-1 | CBSD | fp_total | +0.014 / -0.046 / +0.000 | -0.011 | -0.025 s.d. +0.005 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_latar | +0.004 / -0.020 / -0.002 | -0.006 | -0.014 s.d. +0.003 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_sekelas | +0.016 / -0.013 / +0.016 | +0.006 | -0.002 s.d. +0.015 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CBSD | fp_kelas_lain | -0.006 / -0.013 / -0.014 | -0.011 | -0.019 s.d. -0.004 | turun secara konsisten |
| EXP-3 − EXP-1 | CGM | TP | +1.45 / +3.10 / -1.65 | +0.96 | -0.39 s.d. +2.39 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | tidak terdeteksi | +0.21 / -1.03 / +2.79 | +0.65 | -1.14 s.d. +2.38 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | kotak meleset | -0.72 / +0.31 / -1.14 | -0.52 | -1.90 s.d. +0.89 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | salah kelas | -0.83 / -2.58 / -0.21 | -1.21 | -2.51 s.d. -0.08 | turun secara konsisten |
| EXP-3 − EXP-1 | CGM | fp_total | +0.034 / +0.060 / -0.007 | +0.029 | +0.001 s.d. +0.058 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_latar | -0.002 / +0.032 / -0.020 | +0.003 | -0.017 s.d. +0.024 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_sekelas | +0.007 / +0.010 / -0.002 | +0.005 | -0.007 s.d. +0.019 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CGM | fp_kelas_lain | +0.029 / +0.018 / +0.015 | +0.021 | +0.009 s.d. +0.033 | naik secara konsisten |
| EXP-3 − EXP-1 | CMD | TP | +0.62 / -1.66 / +0.76 | -0.09 | -1.47 s.d. +1.53 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | tidak terdeteksi | +0.35 / +2.07 / +1.17 | +1.20 | -0.40 s.d. +2.76 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | kotak meleset | -2.97 / -1.80 / -1.94 | -2.23 | -3.61 s.d. -0.93 | turun secara konsisten |
| EXP-3 − EXP-1 | CMD | salah kelas | +0.69 / +0.76 / +0.00 | +0.48 | -0.05 s.d. +1.02 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | fp_total | -0.010 / -0.129 / -0.045 | -0.062 | -0.095 s.d. -0.029 | turun secara konsisten |
| EXP-3 − EXP-1 | CMD | fp_latar | +0.008 / -0.076 / -0.034 | -0.034 | -0.060 s.d. -0.013 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | CMD | fp_sekelas | -0.013 / -0.046 / -0.014 | -0.024 | -0.042 s.d. -0.008 | turun secara konsisten |
| EXP-3 − EXP-1 | CMD | fp_kelas_lain | -0.005 / -0.007 / +0.003 | -0.003 | -0.012 s.d. +0.007 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | TP | -2.00 / -3.04 / +0.85 | -1.40 | -3.08 s.d. +0.39 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | tidak terdeteksi | +1.65 / +4.09 / -1.50 | +1.41 | -0.30 s.d. +3.17 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | kotak meleset | -0.45 / -1.35 / +1.50 | -0.10 | -1.13 s.d. +0.89 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | salah kelas | +0.05 / +0.10 / -0.20 | -0.02 | -0.24 s.d. +0.20 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_total | -0.071 / -0.185 / +0.020 | -0.079 | -0.136 s.d. -0.024 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_latar | -0.043 / -0.102 / -0.016 | -0.054 | -0.100 s.d. -0.010 | turun secara konsisten |
| EXP-3 − EXP-1 | HEALTHY | fp_sekelas | -0.019 / -0.077 / +0.020 | -0.025 | -0.045 s.d. -0.007 | tidak terbukti berbeda |
| EXP-3 − EXP-1 | HEALTHY | fp_kelas_lain | -0.009 / -0.005 / +0.016 | +0.000 | -0.012 s.d. +0.012 | tidak terbukti berbeda |
