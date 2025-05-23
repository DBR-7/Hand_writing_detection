# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.title("Digit Classifier: 3 vs 5 (Logistic Regression)")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Load data
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist["data"], mnist["target"].astype(int)
    mask = (y == 3) | (y == 5)
    x_35 = x[mask]
    y_35 = y[mask]
    y_binary = (y_35 == 5).astype(int)
    return x_35, y_binary

x_35, y_binary = load_data()

# Predict
y_pred = model.predict(x_35)
cm = confusion_matrix(y_binary, y_pred)
report = classification_report(y_binary, y_pred, output_dict=True)

# Classification Report
st.subheader("Classification Report")
st.dataframe(report)

# Confusion Matrix
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[3, 5])
disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
st.pyplot(fig_cm)

# Digit Viewer & Prediction
st.subheader("Predict a Sample Image")
digit_class = st.radio("Select digit class to view:", ("3", "5"))
index = st.slider("Select index", 0, 10, 0)

label = 0 if digit_class == "3" else 1
samples = x_35[y_binary == label]
sample = samples[index].reshape(28, 28)

fig_img, ax_img = plt.subplots()
ax_img.imshow(sample, cmap="gray")
ax_img.axis("off")
st.pyplot(fig_img)

pred = model.predict([samples[index]])[0]
pred_label = "5" if pred == 1 else "3"
st.markdown(f"**Model Prediction:** {pred_label}")