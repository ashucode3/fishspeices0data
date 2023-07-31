from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained classifier model
rf_classifier = joblib.load('fish1_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the frontend
    weight = float(request.form['weight'])
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Create a DataFrame from the input values
    data = pd.DataFrame([[weight, length1, length2, length3, height, width]],
                        columns=['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width'])

    # Make a prediction using the model
    prediction = rf_classifier.predict(data)[0]

    return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
