# 🚀 Trip Classification: Maximizing Macro F1 via Post-Hoc Optimization

## 📌 Project Overview
Proyek ini dikembangkan untuk kompetisi klasifikasi perjalanan dengan fokus utama menangani **class imbalance** dan pola non-linear kompleks pada data sensor, trafik, dan ekonomi perjalanan. Inti dari proyek ini bukan hanya membangun model prediktif, tetapi mengoptimalkan metrik **Macro F1 Score** menggunakan teknik optimasi matematis.

## 🛠️ Key Features & Methodology
1. **Memory Optimization:** Implementasi fungsi `reduce_mem_usage` untuk konversi tipe data numerik efisien (int8/float32), mempercepat training hingga 30-50%.
2. **Advanced Feature Engineering:**
   - **Geospatial:** Menghitung jarak menggunakan *Haversine Formula* dan *Route Circuity*.
   - **Economic Interaction:** Fitur *Price_per_KM* dan *Traffic_Pressure_Score* untuk menangkap anomali harga.
   - **Informative Missingness:** Menangani nilai hilang sebagai sinyal, bukan noise, menggunakan flag biner.
3. **Modeling:** Menggunakan **LightGBM Classifier** dengan `class_weight='balanced'`.
4. **Post-Hoc Probability Re-weighting:** Menggunakan algoritma **Nelder-Mead** (dari `scipy.optimize`) untuk mencari bobot kelas optimal yang memaksimalkan Macro F1 tanpa perlu melatih ulang model (*Retraining*).

## 📊 Evaluation Results (Validation Set 90/10)
| Metric | Score |
| :--- | :--- |
| **Macro F1 (Raw)** | 0.XXXXX |
| **Macro F1 (Optimized)** | 0.XXXXX |
| **Net Gain** | +0.0XXXX |

*Optimasi post-hoc berhasil meningkatkan performa kelas minoritas secara signifikan tanpa merusak stabilitas kelas mayoritas.*

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** LightGBM, Scikit-Learn, SciPy (Nelder-Mead), Pandas, NumPy
- **Environment:** Jupyter Notebook / Kaggle
