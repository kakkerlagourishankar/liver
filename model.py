import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

class LiverDiseaseModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()
        self.imputer = SimpleImputer(strategy='mean')

    def train(self, data_path):
        # Load the dataset
        data = pd.read_csv(data_path)

        # Convert 'Gender' to numerical values
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

        # Prepare the features and target variable
        X = data.drop(columns=['Dataset'])  # Features
        y = data['Dataset']  # Target variable: 1 (liver disease), 2 (no liver disease)

        # Handle missing values
        X = self.imputer.fit_transform(X)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Save the model and imputer
        joblib.dump(self.model, 'liver_disease_gb_model.pkl')
        joblib.dump(self.imputer, 'imputer.pkl')
        print("Model and imputer saved successfully.")

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, features):
        # Load the model and imputer
        model = joblib.load('liver_disease_gb_model.pkl')
        imputer = joblib.load('imputer.pkl')

        # Handle missing values in the input features
        features = imputer.transform([features])

        # Make a prediction
        prediction = model.predict(features)[0]
        return "Liver Disease" if prediction == 1 else "No Liver Disease"


# Example usage
if __name__ == "__main__":
    model = LiverDiseaseModel()

    # Train the model
    model.train('indian.csv')  # Update with the correct path to your dataset

    # Test the prediction with example test cases
    test_cases = [
        [45, 1, 0.7, 0.2, 187, 16, 18, 6.5, 3.2, 0.9],  # Case 1: Expected - Healthy or Liver Disease
        [50, 0, 1.5, 0.4, 240, 35, 50, 5.0, 3.5, 1.0],  # Case 2: Expected - Healthy or Liver Disease
        [60, 1, 0.5, 0.1, 150, 10, 20, 6.8, 3.1, 1.2],  # Case 3: Expected - Healthy or Liver Disease
    ]

    for i, test_case in enumerate(test_cases, start=1):
        result = model.predict(test_case)
        print(f"Test Case {i}: {test_case} => Prediction: {result}")
