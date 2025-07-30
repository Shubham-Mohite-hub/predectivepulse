# 🎯 Predictive Pulse: Harnessing Machine Learning for Blood Pressure Analysis

![Preview](https://letsmoderate.com/cdn/shop/articles/Causes_and_symptoms_of_Type-2_Diabetes.jpg?v=1695106711&width=1100)

**Predictive Pulse** is a web-based application that leverages machine learning to predict **diabetes stages** based on **user-input symptoms** and **blood pressure readings** (systolic and diastolic). It provides quick and accurate predictions through an intuitive and responsive user interface.

---

## 🔍 Overview

This project aims to assist users in identifying potential diabetes stages using a logistic regression model. The application features a clean UI, responsive design, and seamless backend integration using Flask.

---

## 🚀 Features

* **🔢 User Input Form:** Collects key health indicators including symptoms and blood pressure values.
* **🤖 Prediction Engine:** Applies a trained **logistic regression** model to classify diabetes stage.
* **📱 Responsive Design:** Built with **Bootstrap** for compatibility across mobile and desktop devices.
* **⚙️ Backend Integration:** Utilizes **Flask** for routing, prediction handling, and server operations.

---

## 🛠️ Tech Stack

| Technology          | Role                                 |
| ------------------- | ------------------------------------ |
| Python              | Core logic and machine learning      |
| Flask               | Web framework for backend            |
| HTML/CSS            | Structuring and styling the frontend |
| Bootstrap           | Responsive and modern UI             |
| Logistic Regression | Diabetes stage prediction model      |

---

## 📦 Installation

Follow the steps below to set up the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/Shubham-Mohite-hub/predectivepulse.git

# 2. Navigate to the project directory
cd predectivepulse

# 3. Run the Flask application
python app.py
```

Then, open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🧪 Usage

1. Fill in the required fields including **symptoms**, **systolic**, and **diastolic** values.
2. Click **Submit** to get an immediate **diabetes stage prediction**.
3. View the results displayed dynamically on the results page.

---

## 📁 Project Structure

```
predectivepulse/
├── app.py             # Flask application entry point
├── model.py           # Machine learning model logic
├── scaler.py          # Preprocessing and scaling logic
├── templates/
│   ├── index.html     # Home page with input form
│   ├── details.html   # Symptom details page
│   └── results.html   # Prediction results display
└── README.md          # Project documentation
```

---

## 📌 Future Enhancements

* Support for additional health indicators (e.g., BMI, glucose levels).
* Integration with a database for storing user history.
* Model upgrade to more advanced classifiers (e.g., Random Forest, XGBoost).
* Deployment on cloud platforms like Heroku or Render.