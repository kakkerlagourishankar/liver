import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer  # For handling missing values
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class LiverDiseaseModel:
    def __init__(self):
        # Initialize the Gradient Boosting Classifier
        self.model = GradientBoostingClassifier()
        self.imputer = SimpleImputer(strategy='mean')  # Replace missing values with column mean

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

    def predict(self, features):
        # Load the model and imputer
        model = joblib.load('liver_disease_gb_model.pkl')
        imputer = joblib.load('imputer.pkl')

        # Handle missing values in the input features
        input_features_imputed = imputer.transform([features])

        # Make a prediction
        return model.predict(input_features_imputed)[0]

if __name__ == "__main__":
    model = LiverDiseaseModel()
    model.train('indian.csv')  # Update with the correct path to your dataset

    # Example usage
    # test_features = [45, 1, 0.7, 187, 16, 18, 6.5, 3.2, 0.9, 5.2]  # Example input
    # result = model.predict(test_features)
    # print("Prediction:", "Liver Disease" if result == 1 else "No Liver Disease")
