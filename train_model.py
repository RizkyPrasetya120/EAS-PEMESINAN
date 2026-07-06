# ==========================
# IMPORT LIBRARY
# ==========================
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# MEMBACA DATASET
# ==========================
print("=" * 50)
print("MEMBACA DATASET")
print("=" * 50)

df = pd.read_csv(
    "dataset/creditcard_50k.csv",
)

print("Dataset berhasil dibaca.\n")

print("5 Data Pertama")
print(df.head())

# ==========================
# INFORMASI DATASET
# ==========================
print("\n" + "=" * 50)
print("INFORMASI DATASET")
print("=" * 50)

print("Jumlah Baris dan Kolom :", df.shape)

print("\nNama Kolom :")
print(df.columns)

print("\nTipe Data :")
print(df.dtypes)

print("\nMissing Value :")
print(df.isnull().sum())

# ==========================
# MEMILIH FITUR DAN TARGET
# ==========================
print("\n" + "=" * 50)
print("MEMILIH FITUR DAN TARGET")
print("=" * 50)

# Menggunakan 5 fitur
X = df[[
    "Amount",
    "V1",
    "V2",
    "V3",
    "V4"
]]

# Target
y = df["Class"]

print("Jumlah Feature :", X.shape[1])

print("\nNama Feature :")
print(X.columns)

# ==========================
# MEMBAGI DATA
# ==========================
print("\n" + "=" * 50)
print("MEMBAGI DATA")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training :", X_train.shape)
print("Testing  :", X_test.shape)

# ==========================
# MEMBUAT MODEL
# ==========================
print("\n" + "=" * 50)
print("MEMBUAT MODEL RANDOM FOREST")
print("=" * 50)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ==========================
# TRAINING MODEL
# ==========================
print("\n" + "=" * 50)
print("TRAINING MODEL")
print("=" * 50)

model.fit(
    X_train,
    y_train
)

print("Training selesai.")

# ==========================
# PREDIKSI
# ==========================
y_pred = model.predict(X_test)

# ==========================
# EVALUASI
# ==========================
print("\n" + "=" * 50)
print("HASIL EVALUASI")
print("=" * 50)

akurasi = accuracy_score(y_test, y_pred)

print("Akurasi :", round(akurasi * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================
# SIMPAN MODEL
# ==========================
joblib.dump(
    model,
    "model/random_forest.pkl"
)

print("Model berhasil disimpan.")

print("\nSELESAI")