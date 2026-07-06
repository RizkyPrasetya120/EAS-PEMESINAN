import pandas as pd

print("Membaca dataset...")

df = pd.read_csv("dataset/creditcard_2023.csv", nrows=50000)

print("Jumlah data :", len(df))

df.to_csv("dataset/creditcard_50k.csv", index=False)

print("Dataset berhasil disimpan!")