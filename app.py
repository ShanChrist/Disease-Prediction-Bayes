from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Load the dataset and train the Bayesian model
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

    # Iterate through the features and get user input from the form
    for feature in X.columns:
        response = request.form[feature]
        user_responses.append(float(response))

    # Calculate the posterior probability
    posterior_prob = model.predict_proba([user_responses])

    # Get the predicted probability of having heart disease
    predicted_probability = posterior_prob[0][1] * 100

    # Determine the prediction result
    if predicted_probability > 0.5:
        prediction = "You may have heart disease."
    else:
        prediction = "You likely do not have heart disease."

    return render_template('prediction.html', predicted_probability=predicted_probability, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

