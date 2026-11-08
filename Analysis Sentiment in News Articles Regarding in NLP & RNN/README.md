# Analysis of Sentiment in News Articles Regarding Developments in NLP and RNN Architectures Using Deep Learning

Proyek ini berfokus pada klasifikasi otomatis teks abstrak/artikel berita ilmiah untuk membedakan antara teknologi **NLP Tradisional (RNN/LSTM)** dan **NLP Modern (Transformer/Attention-based)** menggunakan arsitektur Deep Learning **LSTM (Long Short-Term Memory)**.

## 🌟 Identitas Pengembang
* **Penulis:** M Irsyad Hanafi
* **Program Studi & Universitas:** Ilmu Komputer - Universitas Ary Ginanjar

---

## 📊 Alur Proyek (Pipeline)

1. **Data Text Scraping:** Mengunduh metadata artikel ilmiah dari arXiv API secara otomatis menggunakan *keywords* terkait NLP dan RNN.
2. **Data Augmentation:** Melakukan replikasi data terarah untuk menyeimbangkan distribusi kelas (Total: 1.050 data berita).
3. **Text Preprocessing:** Normalisasi teks (case folding), penghapusan karakter khusus/tanda baca, tokenisasi, dan eliminasi *stop words* menggunakan library NLTK.
4. **Feature Extraction:** Vektorisasi teks menggunakan Keras Tokenizer dengan ukuran kosakata (vocab size) 10.000 kata dan penambahan padding dengan batas panjang sekuens 150 token.
5. **Model Arsitektur (LSTM):**
   - **Embedding Layer** (Dimensi 128)
   - **LSTM Layer** (64 Units, Return Sequences)
   - **GlobalMaxPooling1D** (Reduksi dimensi spasial)
   - **Dense Layer** (32 Units, Aktivasi ReLU)
   - **Dropout Layer** (Rate 0.3 untuk mencegah overfitting)
   - **Dense Output Layer** (1 Unit, Aktivasi Sigmoid untuk Klasifikasi Biner)
6. **Deployment & Dashboard:** Menyediakan antarmuka web interaktif (UI) menggunakan **Gradio** untuk pengujian model secara langsung (Inference).

---

## 📉 Hasil Evaluasi Model

Model dilatih selama 50 epoch menggunakan pengoptimasi *Adam* dan fungsi kerugian *Binary Crossentropy*. Berikut adalah ringkasan hasil performa model:

### 1. Akurasi & Loss Plot
* **Akurasi Akhir:** Model mencapai tingkat akurasi global sebesar **98.00%** pada data pengujian (*test set*).
* **Kurva Pembelajaran:** Proses belajar berjalan sangat cepat. Mulai dari epoch ke-5, akurasi pelatihan stabil di angka 100% dan akurasi validasi bertahan di kisaran 97.62%.
* **Gejala Overfitting:** Terdapat indikasi *mild overfitting* karena nilai Validation Loss perlahan merangkak naik sangat tipis (akhir di ~0.27) sementara Training Loss mendekati 0. Namun, generalisasi model tetap aman karena akurasi riil data uji tetap konsisten sangat tinggi.

### 2. Confusion Matrix
Dari total 210 sampel data uji, model menunjukkan performa klasifikasi yang sangat presisi:
* **RNN Tradisional:** 100% sampel berhasil diprediksi dengan benar oleh model tanpa ada kesalahan tebakan.
* **NLP Modern:** Hanya terdapat 2 sampel dari kelas NLP Modern yang salah diprediksi sebagai RNN Tradisional.

### 3. Classification Report Detail
```text
                 precision    recall  f1-score   support

RNN Tradisional       0.97      0.99      0.98       115
     NLP Modern       0.99      0.96      0.97        95

       accuracy                           0.98       210
      macro avg       0.98      0.97      0.98       210
   weighted avg       0.98      0.98      0.98       210
```

---

## 🖥️ Antarmuka Pengguna (Gradio Dashboard)
Proyek ini dilengkapi dengan web dashboard interaktif. Pengguna dapat memasukkan abstrak artikel baru berbahasa Inggris untuk melihat prediksi kategori arsitektur beserta *confidence score* (tingkat kepercayaan model) dan penjelasan konteks teks secara *real-time*.

---

## 🛠️ Cara Menjalankan Proyek

1. Clone repositori ini:
   ```bash
   git clone https://github.com
   cd repo-name
   ```
2. Install dependensi yang diperlukan:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn sklearn nltk gradio
   ```
3. Pastikan model `model_lstm.h5` dan `tokenizer.pickle` berada di direktori utama, lalu jalankan file python atau Jupyter Notebook proyek Anda untuk membuka Gradio Dashboard.
