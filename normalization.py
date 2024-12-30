import pandas as pd

# Load cleaned dataset
data_cleaned = pd.read_csv('breast-cancer-preprocessed4.csv')  # Ganti dengan path ke file yang sudah dibersihkan

# 1. Menampilkan data awal
print("Data Awal Setelah Dilakukan Pre-Processing:")
print(data_cleaned.head())

# 2. Normalisasi menggunakan Z-Score secara manual
numeric_data = data_cleaned.select_dtypes(include=[float, int])  # Memilih kolom numerik
data_normalized = numeric_data.copy()

for column in numeric_data.columns:
    mean = numeric_data[column].mean()  # Menghitung rata-rata
    std_dev = numeric_data[column].std()  # Menghitung deviasi standar
    data_normalized[column] = (numeric_data[column] - mean) / std_dev  # Normalisasi Z-Score

# 3. Menampilkan Data Setelah Normalisasi
print("\nData Setelah Normalisasi Z-Score:")
print(data_normalized.head())

# Simpan data yang sudah dinormalisasi ke file baru
data_normalized.to_csv('breast-cancer-normalized.csv', index=False)
