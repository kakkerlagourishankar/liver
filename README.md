

# Liver Disease Prediction Web App

A machine learning-powered web application that predicts the likelihood of liver disease based on user-provided medical parameters.

## 🧠 Features

* Utilizes a Gradient Boosting Machine Learning model.
* User-friendly interface built with HTML, CSS, and JavaScript.
* Backend developed using Python and Flask.
* Predicts liver disease risk based on input parameters.

## 📁 Project Structure

```
liver/
├── app.py
├── model.py
├── liver_disease_gb_model.pkl
├── imputer.pkl
├── Indian.csv
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
└── README.md
```

## ✅ Requirements

Ensure you have the following installed:

* Python 3.6 or higher
* pip (Python package installer)

Install the required Python packages:

```bash
pip install -r requirements.txt
```

*Note: Create a `requirements.txt` file with the necessary packages, such as Flask, pandas, scikit-learn, etc.*

## 🚀 How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kakkerlagourishankar/liver.git
   cd liver
   ```

2. **Run the application:**

   ```bash
   python app.py
   ```

3. **Access the web app:**

   Open your browser and navigate to `http://127.0.0.1:5000/` to use the application.

## 📊 Dataset

The application uses the `Indian.csv` dataset, which contains medical records pertinent to liver disease. This dataset is utilized for training and testing the machine learning model.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
