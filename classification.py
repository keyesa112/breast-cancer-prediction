import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# 1. User melakukan input (path data)
file_path = 'breast-cancer-normalized.csv'
data = pd.read_csv(file_path)

# 2. Preprocessing: Memastikan target klasifikasi biner
print("\nNilai unik dalam kolom 'Classification':")
print(data['Classification'].unique())

# Mengubah nilai target jika perlu
data['Classification'] = data['Classification'].apply(lambda x: 0 if x <= 0 else 1)

# Memastikan tidak ada nilai NaN
data = data.dropna()

# Memisahkan fitur dan target
X = data.drop(columns=['Classification'])  # Fitur input
y = data['Classification']  # Target output

# Membagi data menjadi 80% latih dan 20% uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisi fungsi evaluasi
def evaluasi_model(nama_model, y_test, y_pred):
    print(f"\n===== Evaluasi Model {nama_model} =====")
    print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Presisi: {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 3. Klasifikasi dengan Naive Bayes
print("\n===== Melakukan Klasifikasi dengan Naive Bayes =====")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
evaluasi_model("Naive Bayes", y_test, y_pred_nb)

# 4. Klasifikasi dengan Decision Tree
print("\n===== Melakukan Klasifikasi dengan Decision Tree =====")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluasi_model("Decision Tree", y_test, y_pred_dt)

# 5. Klasifikasi dengan Random Forest
print("\n===== Melakukan Klasifikasi dengan Random Forest =====")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluasi_model("Random Forest", y_test, y_pred_rf)

# 6. Membandingkan hasil evaluasi dari ketiga algoritma
print("\n===== Membandingkan Hasil Evaluasi =====")
models = ['Naive Bayes', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_score(y_test, y_pred_nb), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]
precisions = [precision_score(y_test, y_pred_nb), precision_score(y_test, y_pred_dt), precision_score(y_test, y_pred_rf)]
recalls = [recall_score(y_test, y_pred_nb), recall_score(y_test, y_pred_dt), recall_score(y_test, y_pred_rf)]
f1_scores = [f1_score(y_test, y_pred_nb), f1_score(y_test, y_pred_dt), f1_score(y_test, y_pred_rf)]

evaluasi_df = pd.DataFrame({
    'Model': models,
    'Akurasi': accuracies,
    'Presisi': precisions,
    'Recall': recalls,
    'F1-Score': f1_scores
})

print(evaluasi_df)

# 7. Visualisasi hasil evaluasi (Akurasi, Presisi, Recall, F1-Score)
plt.figure(figsize=(10, 6))
evaluasi_df.set_index('Model').plot(kind='bar')
plt.title('Perbandingan Akurasi, Presisi, Recall, dan F1 Score')
plt.ylabel('Skor')
plt.xticks(rotation=0)
plt.show()

# 8. Menampilkan prediksi untuk setiap model
print("\n===== Prediksi untuk Setiap Algoritma =====")
predictions_df = pd.DataFrame({
    'Aktual': y_test.values,
    'Prediksi_Naive_Bayes': y_pred_nb,
    'Prediksi_Decision_Tree': y_pred_dt,
    'Prediksi_Random_Forest': y_pred_rf
})

# Menampilkan 10 prediksi pertama
print(predictions_df.head(10))

# 9. Input manual dari pengguna untuk fitur baru
def get_user_input():
    print("\nMasukkan nilai untuk setiap fitur berikut (gunakan angka float):\n")
    
    age = float(input("1. Age: "))
    bmi = float(input("2. BMI: "))
    glucose = float(input("3. Glucose: "))
    insulin = float(input("4. Insulin: "))
    homa = float(input("5. HOMA: "))
    leptin = float(input("6. Leptin: "))
    adiponectin = float(input("7. Adiponectin: "))
    resistin = float(input("8. Resistin: "))
    mcp1 = float(input("9. MCP.1: "))
    
    # Gabungkan input ke dalam satu array
    input_data = pd.DataFrame([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp1]], 
                               columns=['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 
                                        'Leptin', 'Adiponectin', 'Resistin', 'MCP.1'])
    return input_data

# 10. Prediksi menggunakan input dari pengguna
user_input = get_user_input()

# 11. Melakukan prediksi untuk input dari pengguna
print("\n===== Hasil Prediksi untuk Input Pengguna =====")
for model_name, model in zip(models, [nb, dt, rf]):
    prediction = model.predict(user_input)
    result = "Sehat (0)" if prediction[0] == 0 else "Terdapat Kanker (1)"
    print(f"{model_name}: {result}")
