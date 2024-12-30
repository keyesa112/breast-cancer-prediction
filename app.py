import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load dataset yang sudah diproses dengan benar (Classification tidak diskalakan)
file_path = 'breast-cancer-normalized.csv'  # Pastikan ini adalah file tanpa scaling pada 'Classification'
data = pd.read_csv(file_path)

# Pra-pemrosesan
print("\nNilai unik dalam kolom 'Classification' sebelum mapping:")
print(data['Classification'].unique())

# Mengubah nilai target: 1 -> 0 (Negatif), 2 -> 1 (Positif)
data['Classification'] = data['Classification'].apply(lambda x: 0 if x == 1 else 1)

print("\nNilai unik dalam kolom 'Classification' setelah mapping:")
print(data['Classification'].unique())

# Memastikan tidak ada nilai NaN
data = data.dropna()

# Memisahkan fitur dan target
X = data.drop(columns=['Classification'])  # Fitur input
y = data['Classification']  # Target output

# Membagi data menjadi 80% latih dan 20% uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Melatih model
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Menghitung akurasi
nb_accuracy = accuracy_score(y_test, nb.predict(X_test_scaled))
dt_accuracy = accuracy_score(y_test, dt.predict(X_test_scaled))
rf_accuracy = accuracy_score(y_test, rf.predict(X_test_scaled))

# Menghitung confusion matrix dan classification report
print("\n===== Evaluasi Model Naive Bayes =====")
print(confusion_matrix(y_test, nb.predict(X_test_scaled)))
print(classification_report(y_test, nb.predict(X_test_scaled)))

print("\n===== Evaluasi Model Decision Tree =====")
print(confusion_matrix(y_test, dt.predict(X_test_scaled)))
print(classification_report(y_test, dt.predict(X_test_scaled)))

print("\n===== Evaluasi Model Random Forest =====")
print(confusion_matrix(y_test, rf.predict(X_test_scaled)))
print(classification_report(y_test, rf.predict(X_test_scaled)))

# Menampilkan akurasi
print("\nAkurasi Naive Bayes:", nb_accuracy)
print("Akurasi Decision Tree:", dt_accuracy)
print("Akurasi Random Forest:", rf_accuracy)

# Kamus akurasi
accuracies = {
    'Naive Bayes': nb_accuracy,
    'Decision Tree': dt_accuracy,
    'Random Forest': rf_accuracy
}

# Fungsi prediksi
def predict_cancer(input_data):
    input_df = pd.DataFrame([input_data]).astype(float)
    input_scaled = scaler.transform(input_df)
    print("Input Data untuk Prediksi:")
    print(input_scaled)
    predictions = {
        'Naive Bayes': nb.predict(input_scaled)[0],
        'Decision Tree': dt.predict(input_scaled)[0],
        'Random Forest': rf.predict(input_scaled)[0]
    }
    best_model_name = max(accuracies, key=accuracies.get)
    best_prediction = predictions[best_model_name]
    print("Prediksi:", predictions)
    print("Model Terbaik:", best_model_name)
    print("Prediksi Terbaik:", best_prediction)
    return predictions, best_model_name, best_prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        data_input = {
            'Age': request.form['Age'],
            'BMI': request.form['BMI'],
            'Glucose': request.form['Glucose'],
            'Insulin': request.form['Insulin'],
            'HOMA': request.form['HOMA'],
            'Leptin': request.form['Leptin'],
            'Adiponectin': request.form['Adiponectin'],
            'Resistin': request.form['Resistin'],
            'MCP.1': request.form['MCP.1']
        }

        # Validasi input data
        try:
            input_data = {key: float(value) for key, value in data_input.items()}
        except ValueError:
            return render_template('diagnose.html', error="Pastikan semua input adalah angka.")

        predictions, best_model_name, best_prediction = predict_cancer(input_data)
        
        return redirect(url_for('result', 
                                **{'Naive Bayes': predictions['Naive Bayes'],
                                   'Decision Tree': predictions['Decision Tree'],
                                   'Random Forest': predictions['Random Forest'],
                                   'best_model_name': best_model_name,
                                   'best_prediction': best_prediction,
                                   **data_input}))
    return render_template('diagnose.html')

@app.route('/result')
def result():
    data = request.args.to_dict()
    predictions = {}
    best_model_name = None
    best_prediction = None
    
    # Ekstrak prediksi
    for key in ['Naive Bayes', 'Decision Tree', 'Random Forest']:
        if key in data:
            try:
                predictions[key] = int(float(data.pop(key)))
            except ValueError:
                predictions[key] = -1  # Menandakan data tidak valid
    
    # Ekstrak model terbaik
    if 'best_model_name' in data:
        best_model_name = data.pop('best_model_name')
    if 'best_prediction' in data:
        try:
            best_prediction = int(float(data.pop('best_prediction')))
        except ValueError:
            best_prediction = -1  # Menandakan data tidak valid
    
    return render_template('result.html', 
                           data=data, 
                           predictions=predictions, 
                           best_model_name=best_model_name, 
                           best_prediction=best_prediction)

if __name__ == '__main__':
    app.run(debug=True)
