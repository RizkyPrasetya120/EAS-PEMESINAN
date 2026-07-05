from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model Random Forest
model = joblib.load("model/random_forest.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=""
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Mengambil input dari form
        amount = float(request.form["Amount"])
        v1 = float(request.form["V1"])
        v2 = float(request.form["V2"])
        v3 = float(request.form["V3"])
        v4 = float(request.form["V4"])

        # Menyusun data sesuai urutan fitur saat training
        data = np.array([
            amount,
            v1,
            v2,
            v3,
            v4
        ]).reshape(1, -1)

        # Prediksi
        hasil = model.predict(data)

        # Menampilkan hasil
        if hasil[0] == 0:
            prediction = "✅ Transaksi Normal"
        else:
            prediction = "⚠️ Transaksi Fraud"

        return render_template(
            "index.html",
            prediction_text=prediction
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Terjadi Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)