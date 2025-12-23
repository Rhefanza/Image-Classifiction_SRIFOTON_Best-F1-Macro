# Image-Classifiction_SRIFOTON_Best-F1-Macro
This project aims to develop an accurate Machine Learning model in predicting and detecting respiratory diseases through X-Ray images of patients' lungs.

Berikut adalah draf **README.md** yang profesional dan terstruktur untuk proyek Machine Learning Anda berdasarkan isi notebook kompetisi SRIFOTON tersebut.

---

# 🏆 SRIFOTON Machine Learning Competition - The Champion Solution

Repositori ini berisi solusi *image classification* untuk mendeteksi penyakit paru-paru (COVID, Normal, dan Viral Pneumonia) melalui citra sinar-X (X-Ray). Notebook ini menggunakan teknik *transfer learning* mutakhir, *pseudo-labeling*, dan *cost-sensitive learning* untuk mencapai performa tinggi.

## 🚀 Ringkasan Model

* **Arsitektur:** `tf_efficientnet_b5_ns` (Noisy Student) melalui library `timm`.
* **Strategi Training:** 3-Fold Stratified Cross-Validation.
* **Teknik Khusus:**
* **Pseudo-Labeling:** Memanfaatkan data test yang telah dilabeli sebelumnya untuk memperkuat basis data training.
* **Cost-Sensitive Loss:** Menggunakan bobot kelas (*class weights*) pada CrossEntropyLoss untuk menangani ketidakseimbangan data (imbalanced dataset).
* **Ensemble:** Rata-rata probabilitas dari model-model hasil cross-validation untuk prediksi akhir.



## 🛠️ Persyaratan Sistem

Pastikan Anda memiliki lingkungan Python dengan dependensi berikut:

* `torch` & `torchvision`
* `timm` (PyTorch Image Models)
* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `tqdm`

## 📊 Konfigurasi Eksperimen

Detail konfigurasi yang digunakan dalam `CFG` class:
| Parameter | Nilai |
| :--- | :--- |
| **Model** | EfficientNet-B5 Noisy Student |
| **Image Size** | 456 x 456 |
| **Batch Size** | 8 |
| **Epochs** | 5 (per fold) |
| **Learning Rate** | 2e-4 (AdamW) |
| **Scheduler** | CosineAnnealingLR |
| **Augmentasi** | Resize, Horizontal Flip, Affine, Color Jitter |

## 📁 Struktur Dataset

Notebook ini mengharapkan struktur direktori sebagai berikut:

```
/kaggle/input/final-srifoton-25-machine-learning-competition/
    ├── train/train/
    │   ├── COVID/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    └── test/test/
        └── [image_files].png

```

## 📈 Alur Kerja (Workflow)

1. **Preprocessing:** Gambar diubah ukurannya menjadi 456x456 (resolusi asli B5) dan dinormalisasi menggunakan statistik ImageNet.
2. **Pseudo-Labeling:** Menggabungkan data training asli dengan data test yang sudah diprediksi sebelumnya (14.510 total gambar).
3. **Cross-Validation:** Melatih model pada 3 fold berbeda untuk memastikan stabilitas prediksi.
4. **Handling Imbalance:** Menghitung bobot kelas secara dinamis berdasarkan distribusi label di setiap fold.
5. **Inference:** Melakukan *soft-voting ensemble* dengan merata-ratakan output probabilitas dari setiap model fold terbaik.

## 🏆 Hasil Evaluasi

Model mencapai performa luar biasa pada tahap validasi:

* **Mean OOF Macro F1 Score:** ~0.9951
* **Standard Deviation:** 0.0016

## 📝 Cara Penggunaan

1. Sesuaikan `base_dir` di dalam kelas `CFG` dengan lokasi dataset Anda.
2. Jalankan sel **Import Library** dan **Konfigurasi**.
3. Jalankan tahap **Cross Validation Training** untuk menghasilkan file `.pth` (bobot model).
4. Jalankan sel **Ensemble Inference** untuk menghasilkan file `submission.csv`.


**Saran Tambahan:**
Jika Anda ingin mengunggah ini ke GitHub, pastikan untuk menambahkan file `.gitignore` agar file model besar (`.pth`) atau dataset tidak ikut terunggah secara tidak sengaja. Apakah Anda ingin saya menambahkan penjelasan teknis tentang salah satu bagian kodenya?
