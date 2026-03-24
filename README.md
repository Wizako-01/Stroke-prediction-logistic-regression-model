# Stroke Risk Prediction App

This project predicts the risk of stroke using patient data. It covers the full machine learning workflow: data cleaning, model training, evaluation, and building a simple web app with Streamlit.

---

## What this project does

- Takes patient information (age, BMI, glucose, etc.)
- Predicts the probability of stroke
- Classifies the result as **High Risk** or **Low Risk**

---

## Dataset

Stroke prediction dataset (Kaggle)

Target:
- 0 = No stroke  
- 1 = Stroke  

---

## Models Used

I tested two models:

### Random Forest
- Very high accuracy (~95%)
- But failed to detect stroke cases (recall = 0)

### Logistic Regression
- Used class balancing
- Much better at detecting stroke cases

👉 I chose Logistic Regression as the final model.

---

## Final Results

Using a threshold of **0.4**:

- Accuracy: 0.68  
- Recall (Stroke): 0.84  

This model focuses on detecting stroke cases, which is more important than just having high accuracy.

---

## Files

- `app.py` → Streamlit app  
- `stroke_logistic_model.pkl` → trained model  
- `label_encoders.pkl` → encoders  
- `threshold.json` → decision threshold  
- `requirements.txt` → dependencies  
- Notebook → training process  

---

## How to run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
