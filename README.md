# NEURO-AI---Early-Parkinson-s-Detection-Using-XGBoost

An AI-powered web application built with *Streamlit* that predicts the likelihood of *Parkinson’s disease* based on *voice pattern data*.  
The project aims to support early diagnosis and raise awareness of neurological health using machine learning.

---

## 🚀 Features  

- *Parkinson’s Prediction:* Predicts whether a person is affected or not using XGBoost based on voice data inputs.  
- *Patient Details Input:* Allows entering patient information such as name, age, and gender for personalized results.  
- *Patient History Tracking:* Stores and displays previous predictions for better record-keeping.  
- *Analytics Dashboard:* Provides visual insights like health distribution, age analysis, and feature-based comparisons.  
- *Educational Section:* Explains each important feature (Jitter, Shimmer, HNR, etc.) in simple terms.  
- *Lifestyle Recommendations:* Suggests preventive and wellness tips to promote healthy living.  

---

## 🧩 Tech Stack  

- *Frontend:* Streamlit  
- *Backend / ML Model:* XGBoost  
- *Libraries:* Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Seaborn  
- *Language:* Python  

---

## 📊 Dataset  

The project uses the *Parkinson’s Disease Dataset* from the *UCI Machine Learning Repository*:  
[🔗 Parkinson’s Dataset – UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)

This dataset includes biomedical voice measurements from individuals with and without Parkinson’s disease.

---

## ⚙ How It Works  

1. *Data Preprocessing:*  
   - The dataset is cleaned, normalized, and scaled using StandardScaler.  
2. *Model Training:*  
   - An *XGBoost Classifier* is trained to distinguish between affected and healthy individuals.  
3. *Web Interface:*  
   - The trained model is integrated into a *Streamlit web app* for easy prediction.  
4. *Prediction Output:*  
   - The user enters voice-related feature values, and the model predicts whether the person is likely affected by Parkinson’s disease.  
5. *Insights & Recommendations:*  
   - The app provides interpretive charts, educational information, and lifestyle tips.

---

## 🧠 Model Overview  

- *Algorithm Used:* XGBoost Classifier  
- *Target Variable:* status (1 = Parkinson’s, 0 = Healthy)  
- *Features Used:* Voice parameters such as Fo(Hz), Fhi(Hz), Flo(Hz), Jitter, Shimmer, HNR, RPDE, DFA, PPE, etc.  

---

## 🖥 Installation & Setup  

### ⿡ Clone the Repository  

### ⿢ Install Dependencies

Make sure you have Python 3.9+ installed. Then run:

pip install -r requirements.txt

### ⿣ Run the App
streamlit run app.py

### ⿤ Access the Web App

After running the above command, open the local URL shown (usually http://localhost:8501/) in your browser.
