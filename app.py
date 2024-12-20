from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the trained model and the imputer (replace 'model.pkl' with your model file)
model = joblib.load('liver_disease_gb_model.pkl')  # Loaded the correct model for prediction
imputer = joblib.load('imputer.pkl')  # Loaded the imputer for handling missing values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = int(request.form['age'])
        gender = request.form['gender']
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphotase = float(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = float(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = float(request.form['aspartate_aminotransferase'])
        total_proteins = float(request.form['total_proteins'])
        albumin = float(request.form['albumin'])
        albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])

        # Validate input values (check for negative or zero values)
        if (
            age <= 0 or total_bilirubin <= 0 or direct_bilirubin <= 0 or 
            alkaline_phosphotase <= 0 or alamine_aminotransferase <= 0 or 
            aspartate_aminotransferase <= 0 or total_proteins <= 0 or 
            albumin <= 0 or albumin_and_globulin_ratio <= 0
        ):
            return jsonify({'result': "Invalid input: All values must be positive and non-zero."})

        # Prepare input features for the model (convert categorical data if necessary)
        gender_encoded = 1 if gender.lower() == 'male' else 0  # Example encoding for gender
        input_features = np.array([
            age, gender_encoded, total_bilirubin, direct_bilirubin,
            alkaline_phosphotase, alamine_aminotransferase,
            aspartate_aminotransferase, total_proteins,
            albumin, albumin_and_globulin_ratio
        ]).reshape(1, -1)

        # Handle missing values using the imputer
        input_features_imputed = imputer.transform(input_features)

        # Predict using the trained model
        prediction = model.predict(input_features_imputed)[0]

        # Map the prediction result
        result = (
            "The person is likely to have liver disease." if prediction == 1 
            else "The person is likely to be healthy."
        )

        return jsonify({'result': result})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'result': "An error occurred while processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
