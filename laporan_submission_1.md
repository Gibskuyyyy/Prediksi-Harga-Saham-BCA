# Laporan Proyek Machine Learning - Prediksi Saham BBCA [Gibran Malik Naabih Andito]

## Domain Proyek

Saham adalah salah satu instrumen investasi yang populer di Indonesia. Salah satu saham unggulan di Bursa Efek Indonesia (BEI) adalah PT Bank Central Asia Tbk (BBCA). Pergerakan harga saham BBCA menjadi perhatian banyak investor karena stabilitas dan fundamental perusahaan yang kuat. Dalam proyek ini, kami mencoba memprediksi arah pergerakan harga saham BBCA (naik atau turun) berdasarkan data historis dari tahun 2018 hingga 2025.

**Mengapa ini penting?**
Prediksi arah harga saham penting untuk pengambilan keputusan investasi. Meskipun pasar saham sangat dipengaruhi oleh sentimen dan peristiwa eksternal, pendekatan data-driven seperti machine learning bisa membantu investor memahami pola tersembunyi dari data historis.

**Referensi:**

* J. B. Heaton, N. G. Polson, and J. H. Witte, “Deep Learning in Finance,” arXiv:1602.06561, 2016.

---

## Business Understanding

### Problem Statements

1. Dapatkah kita memprediksi arah pergerakan harga saham BBCA pada hari berikutnya berdasarkan data historis?
2. Model machine learning mana yang memberikan hasil terbaik dalam memprediksi arah harga saham?

### Goals

1. Mengembangkan model klasifikasi biner untuk memprediksi apakah harga saham akan naik atau turun.
2. Mengevaluasi dan membandingkan performa beberapa model machine learning (Random Forest, XGBoost, Logistic Regression).

### Solution Statements

* Kami menggunakan tiga model:

  * RandomForestClassifier
  * XGBClassifier
  * LogisticRegression
* Model dievaluasi menggunakan metrik klasifikasi: akurasi, precision, recall, dan F1-score.
* Target biner dibuat berdasarkan perbandingan harga penutupan hari ini dan esok.

---

## Data Understanding

Data diambil dari [Yahoo Finance](https://finance.yahoo.com) menggunakan library `yfinance` dengan ticker `BBCA.JK`, mencakup periode dari 1 Januari 2018 hingga 29 Mei 2024.

### Fitur yang digunakan:

* `Open`: Harga pembukaan
* `High`: Harga tertinggi hari itu
* `Low`: Harga terendah hari itu
* `Close`: Harga penutupan
* `Volume`: Volume transaksi
* `target`: Label biner; 1 jika harga `Close` hari esok > `Close` hari ini, 0 jika sebaliknya

Kami juga menampilkan:

* Statistik deskriptif
* Heatmap korelasi antar fitur
* Visualisasi tren harga saham

---

## Data Preparation

* Menghapus nilai duplikat
* Menambahkan kolom `target` untuk klasifikasi biner (naik/turun)
* Melakukan *standard scaling* terhadap fitur numerik menggunakan `StandardScaler`
* Membagi data menjadi *training* dan *validation* dengan rasio 80:20

---

## Modeling

### Model yang digunakan:

1. **Random Forest Classifier**
2. **XGBoost Classifier**
3. **Logistic Regression**

Semua model menggunakan parameter default (belum dilakukan hyperparameter tuning). Data telah dibagi dan diskalakan secara tepat.

---

## Evaluation

### Metrik Evaluasi:

* **Akurasi**: Proporsi prediksi benar terhadap keseluruhan prediksi
* **Precision**: Ketepatan model dalam memprediksi kelas positif
* **Recall**: Kemampuan model menangkap seluruh kelas positif
* **F1-score**: Harmonic mean dari precision dan recall

### Hasil Evaluasi:

1. **Random Forest**

   * Akurasi: \~54%
   * Model tampak seperti hanya sedikit lebih baik dari random guessing

2. **XGBoost**

   * Akurasi: \~53%
   * Tidak memberikan peningkatan berarti dari Random Forest

3. **Logistic Regression**

   * Akurasi: \~54%
   * Mirip dengan dua model sebelumnya, performa masih lemah

### Analisis:

* Semua model menunjukkan hasil prediksi yang kurang memuaskan.
* Hal ini bisa disebabkan oleh:

  * Fitur yang terlalu sederhana (tidak menyertakan indikator teknikal seperti RSI, MACD, dll.)
  * Fluktuasi harga saham yang dipengaruhi oleh faktor eksternal seperti berita, kebijakan pemerintah, dan kondisi pasar global.
  * Perlu dilakukan feature engineering dan tuning lebih lanjut.

---

## Kesimpulan

Model klasifikasi biner sederhana berdasarkan data historis harga saham BBCA belum mampu menghasilkan prediksi yang akurat. Untuk hasil yang lebih baik, perlu ditambahkan lebih banyak fitur yang merepresentasikan dinamika pasar serta dilakukan hyperparameter tuning.

---
