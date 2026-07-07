# UAS-Machine-Learning
Kelompok 18

Fadhillah Rizqia Arfin
Muhammad Jalallullail
Raditya Raihan

# Link Dataset

https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

# Prediksi Mahasiswa Dropout Menggunakan Algoritma Boosting dengan Interpretabilitas SHAP

Repositori ini berisi kode sumber dan hasil eksperimen untuk memprediksi risiko *dropout* (putus kuliah) serta keberhasilan akademik mahasiswa. Studi ini mengimplementasikan dan membandingkan tiga algoritma *tree-based boosting* mutakhir: **XGBoost**, **LightGBM**, dan **CatBoost**, yang diintegrasikan dengan **SHAP (SHapley Additive exPlanations)** untuk aspek interpretabilitas model.

---

##  Hasil Eksperimen & Evaluasi Model

Seluruh model dievaluasi menggunakan metode **10-fold Cross-Validation** pada dataset *Predict Students Dropout and Academic Success* dari UCI Machine Learning Repository. Berikut adalah ringkasan metrik performa yang dicapai oleh masing-masing model:

| Model | Akurasi (%) | Presisi (%) | Recall (%) | F1-Score (%) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **88.47%** | 85.04% | 78.13% | **81.45%** | 0.9315 |
| **XGBoost** | 88.10% | 83.92% | **78.55%** | 81.15% | **0.9328** |
| **CatBoost** | 87.90% | **85.34%** | 78.12% | 81.08% | 0.9310 |

### Poin Penting:
* **LightGBM** berhasil mencetak nilai **Akurasi (88.47%)** dan **F1-Score (81.45%)** tertinggi secara keseluruhan, menjadikannya model yang sangat andal untuk prediksi yang seimbang.
* **XGBoost** menunjukkan kemampuan diskriminasi kelas terbaik dengan perolehan skor **ROC-AUC tertinggi sebesar 0.9328**.
* **CatBoost** memberikan performa yang sangat kompetitif dengan perolehan skor **Presisi** tertinggi di antara ketiganya.

---

##  Interpretabilitas Model Melalui SHAP

Untuk menghindari sifat model *machine learning* yang seperti "kotak hitam" (*black-box*), kami menggunakan **SHAP global dan local explanations** guna membongkar faktor-faktor utama yang mendorong keputusan model terhadap risiko mahasiswa *dropout*.

### Fitur Paling Berpengaruh (Global Feature Importance)
Berdasarkan grafik SHAP Summary Plot (Beeswarm), fitur-fitur paling kritis yang memengaruhi prediksi model adalah:
1. **Curricular units 2nd sem (approved):** Jumlah mata kuliah yang lulus di semester 2 merupakan prediktor terkuat. Semakin sedikit jumlah mata kuliah yang lulus, semakin besar risiko mahasiswa mengalami *dropout*.
2. **Tuition fees up to date:** Stabilitas finansial memainkan peran yang sangat vital. Mahasiswa dengan status pembayaran kuliah yang tidak lancar (tertunggak) secara signifikan lebih rentan untuk *dropout*.
3. **Curricular units 1st sem (approved):** Mirip dengan semester dua, performa akademik awal di semester 1 sangat menentukan bertahannya seorang mahasiswa di perguruan tinggi.

---

##  Stack Teknologi & Metodologi

- **Bahasa Pemrograman:** Python 3.x
- **Pustaka (Libraries):** `xgboost`, `lightgbm`, `catboost`, `shap`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Strategi Validasi:** 10-Fold Cross-Validation untuk mencegah *overfitting* dan memastikan kemampuan generalisasi model.
- **Penanganan Data Tidak Seimbang:** SMOTE (Synthetic Minority Over-sampling Technique) diimplementasikan untuk menangani ketidakseimbangan kelas pada data kelulusan mahasiswa.

---


├── README.md              # Dokumentasi proyek
└── requirements.txt       # Daftar dependensi pustaka Python
