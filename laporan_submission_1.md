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

penjelasan dataset :
* Terdapat 19 data yang duplikat, saya lakukan penghapusan dataset yang duplikat tersebut
* Tidak ada data yang hilang, hanya saja terdapat data yang bernilai 0 yang menggambarkan market sedang libur
* Untuk outlier pada dataset saya tidak melakukan tindakan apa-apa karena pertimbangan saya outlier yang terdapat pada data adalah volume yang menurut saya perlu adanya perubahan yang signifikan untuk menggerakkan harga saham
* Terdapat 1821 data sebelum dilakukan penghapusan duplikat, menjadi 1802 setelah dihapus dataset yang duplikat.

---

## Data Preparation

* Menghapus nilai duplikat
* Menambahkan kolom `target` untuk klasifikasi biner (naik/turun)
* Melakukan *standard scaling* terhadap fitur numerik menggunakan `StandardScaler`
* Membagi data menjadi *training* dan *validation* dengan rasio 80:20
* Membuat kolom baru berisikan MA sebagai indikator tambahan untuk melihat histori rata-rata untuk pergerakan pada masa lalu. karena MA merupakan rata-rata dari masa lalu seperti contohnya MA(7) berarti pergerakan 7 hari, dan MA(14) berarti pergerakan 14 hari terdapat kolom yang tidak memiliki nilai karena data yang tidak ada seperti contohnya 14 baris awal pada data tidak ada nilai MA(14)nya maka dari itu saya hilangkan untuk yang tidak ada nilanya.

---

## Modeling

### Model yang digunakan:

Berikut penjelasan cara kerja dari ketiga algoritma yang Anda gunakan dalam proyek prediksi saham BBCA:

1. **Logistic Regression**

**Cara Kerja:**

* Logistic Regression adalah algoritma supervised learning untuk klasifikasi.
* Algoritma ini memodelkan probabilitas dari suatu kelas (dalam kasus Anda: naik atau turun) menggunakan fungsi sigmoid.
* Fungsi sigmoid mengubah output linear (dari kombinasi fitur dan bobot) menjadi nilai probabilitas antara 0 dan 1.
* Jika probabilitas di atas ambang tertentu (biasanya 0.5), maka prediksi diklasifikasikan sebagai "1" (misalnya naik), dan sebaliknya sebagai "0" (misalnya turun).

**Kelebihan:**

* Cepat dan sederhana untuk diterapkan.
* Mudah diinterpretasikan.
* Cocok untuk baseline model.

**Kekurangan:**

* Tidak menangani hubungan non-linear antar fitur dengan baik.
* Sensitif terhadap multikolinearitas dan outlier.

2. **Random Forest**

**Cara Kerja:**

* Random Forest adalah ensemble learning berbasis decision tree.
* Algoritma ini membuat banyak decision tree (biasanya ratusan) dari subset acak data dan fitur.
* Setiap tree memberi "vote" untuk hasil prediksi, dan hasil akhir diputuskan berdasarkan mayoritas vote (klasifikasi).
* Dengan teknik ini, Random Forest mampu mengurangi overfitting yang umum terjadi pada decision tree tunggal.

**Kelebihan:**

* Akurasi tinggi untuk banyak jenis data.
* Menangani missing value dan data kategori dengan baik.
* Dapat mengukur pentingnya fitur (feature importance).

**Kekurangan:**

* Tidak secepat model linear.
* Kurang transparan untuk interpretasi model.

3. **XGBoost (Extreme Gradient Boosting)**

**Cara Kerja:**

* XGBoost adalah algoritma boosting berbasis decision tree yang sangat efisien.
* Model dibangun secara bertahap, di mana setiap tree berikutnya mencoba mengoreksi kesalahan dari tree sebelumnya.
* Menggunakan teknik gradient descent untuk meminimalkan loss function (kerugian).
* Termasuk regularisasi (penalti kompleksitas model) yang membantu mengurangi overfitting.

**Kelebihan:**

* Performa tinggi dan efisien.
* Dapat menangani missing value secara otomatis.
* Banyak digunakan di kompetisi machine learning karena akurasi dan fleksibilitasnya.

**Kekurangan:**

* Parameter tuning lebih kompleks.
* Interpretasi hasil lebih sulit dibanding model sederhana.

**parameter** :
1. logistic regression : random_state=42
2. random forest : n_estimators=100, random_state=42
3. xgboost : objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42 
 
Data telah dibagi dan diskalakan secara tepat.

---

## Evaluation

### Metrik Evaluasi:

* **Akurasi**: Proporsi prediksi benar terhadap keseluruhan prediksi
* **Precision**: Ketepatan model dalam memprediksi kelas positif
* **Recall**: Kemampuan model menangkap seluruh kelas positif
* **F1-score**: Harmonic mean dari precision dan recall

### Hasil Evaluasi:

1. **Random Forest**

   * Akurasi: \~56%
   * Model tampak seperti hanya sedikit lebih baik dari random guessing

2. **XGBoost**

   * Akurasi: \~56%
   * Tidak memberikan peningkatan berarti dari Random Forest

3. **Logistic Regression**

   * Akurasi: \~56%
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
