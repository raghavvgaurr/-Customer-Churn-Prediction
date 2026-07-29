# Customer-Churn-Prediction
 Customer Churn Prediction using Ann
 # 🏦 Customer Churn Prediction using Artificial Neural Networks (ANN)

A Machine Learning web application that predicts whether a bank customer is likely to churn (leave the bank) using an Artificial Neural Network (ANN). The application is built with **TensorFlow**, **Scikit-Learn**, and **Streamlit**, and is deployed online for real-time predictions.

---

## 🚀 Live Demo

🔗 **Application:** https://customer-churn-prediction-s8wl.onrender.com

---

## 📌 Features

- Predicts customer churn using an ANN model.
- Interactive Streamlit web interface.
- Real-time predictions.
- One-Hot Encoding for categorical features.
- StandardScaler for feature scaling.
- Clean and responsive UI.

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py                     # Streamlit application
├── model.h5                   # Trained ANN model
├── scaler.pkl                 # StandardScaler
├── labelEncoder.pkl           # Gender Label Encoder
├── OneHotEncoder_geo.pkl      # Geography OneHotEncoder
├── requirements.txt
├── README.md
├── runtime.txt
└── Churn_Modelling.csv
```

---

## 📊 Dataset

The project uses the **Churn Modelling Dataset**, which contains customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member
- Estimated Salary

Target Variable:

- **Exited**
  - 0 → Customer stays
  - 1 → Customer leaves

---

## 🧠 Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Label Encoding (Gender)
      │
      ▼
One-Hot Encoding (Geography)
      │
      ▼
Feature Scaling
      │
      ▼
Artificial Neural Network
      │
      ▼
Customer Churn Prediction
```

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Streamlit

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/raghavvgaurr/-Customer-Churn-Prediction.git
```

Move into the project directory

```bash
cd -Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Preview

> Add screenshots of your application here.

Example:

```
Home Page

Prediction Result

Customer Churn Probability
```

---

## 📈 Model Information

Algorithm Used:

- Artificial Neural Network (ANN)

Libraries:

- TensorFlow
- Keras

Evaluation Metrics:

- Accuracy
- Binary Crossentropy Loss

---

## 🎯 Future Improvements

- Docker Deployment
- AWS Deployment
- CI/CD using GitHub Actions
- User Authentication
- Batch Predictions using CSV Upload
- Model Monitoring

---

## 👨‍💻 Author

**Raghav Gaur**

GitHub: https://github.com/raghavvgaurr


---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
<img width="1360" height="598" alt="image" src="https://github.com/user-attachments/assets/10f3cb76-0c5b-4d99-8f2d-2b03e40eee52" />
<img width="1346" height="591" alt="image" src="https://github.com/user-attachments/assets/0a6cf1a0-998e-47b7-aca2-2a52335c64fa" />
