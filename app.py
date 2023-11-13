from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Load datset dan Train Model Bayesian
df = pd.read_csv('heart_disease_dataset.csv')
X = df.drop(columns=['target'])
y = df['target']
model = GaussianNB()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html', X=X)

@app.route('/predict', methods=['POST'])
def predict():
    user_responses = []

    # Membaca input user dari kolom
    for feature in X.columns:
        response = request.form[feature]
        user_responses.append(float(response))

    # Kalkulasi probabilitas dari input user
    posterior_prob = model.predict_proba([user_responses])

    # Mengubah format probabilitas menjadi persentase
    predicted_probability = posterior_prob[0][1] * 100

    # Hasil Output
    if predicted_probability > 50:
        prediction = "You may have heart disease."
    else:
        prediction = "You likely do not have heart disease."

    return render_template('prediction.html', predicted_probability=predicted_probability, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

