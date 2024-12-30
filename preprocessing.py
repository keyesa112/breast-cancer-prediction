import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
data = pd.read_csv('breast-cancer-preprocessed3.csv')  # Ganti dengan path ke file Anda

# 1. Menampilkan data awal
print("Data Awal:")
print(data.head())

# 2. Cari dan Tampilkan Jumlah Missing Values
total_missing = data.isnull().sum().sum()
print(f"\nJumlah Missing Values: {total_missing}")

# 3. Tampilkan Missing Values Tiap Atribut
print("\nMissing Values per Atribut:")
print(data.isnull().sum())

# 4. Deteksi Outliers (menggunakan Z-Score)
numeric_data = data.select_dtypes(include=[np.number])
z_scores = np.abs(stats.zscore(numeric_data))

# Tetapkan threshold z-score (contoh: z > 4 dianggap outlier)
outlier_mask = (z_scores > 4)

# Tampilkan jumlah total outliers
total_outliers_count = outlier_mask.any(axis=1).sum()
print(f"\nJumlah Total Outliers: {total_outliers_count}")

# Hitung total outliers dan tampilkan outliers per atribut
total_outliers = outlier_mask.sum(axis=0)
print("\nJumlah Outliers per Atribut:")
print(total_outliers)

# 6. Menghapus Missing Values dan Outliers
data_cleaned = data.dropna()
data_cleaned = data_cleaned[~outlier_mask.any(axis=1)]

# 7. Tampilkan Data Setelah Menghapus Missing Values dan Outliers
print("\nData Setelah Menghapus Missing Values dan Outliers:")
print(data_cleaned.head())

# Simpan data yang sudah dibersihkan ke file baru
data_cleaned.to_csv('breast-cancer-preprocessed4.csv', index=False)