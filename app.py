import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Malaria cell classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malaria.keras")

model = load_model()

def prob_uninfected(img):
    x = np.array(img.convert("RGB").resize((128, 128)), dtype="float32")[None]
    return float(model.predict(x, verbose=0)[0][0])

st.title("Malaria cell classifier")
st.caption("Student research demo. Not for clinical use.")

tab1, tab2 = st.tabs(["Single image", "Batch accuracy"])

with tab1:
    f = st.file_uploader("Upload one cell image", type=["png", "jpg", "jpeg"], key="single")
    if f:
        img = Image.open(f)
        st.image(img, width=200)
        p = prob_uninfected(img)
        st.write(f"Uninfected: {p:.1%}")
        st.write(f"Infected: {1 - p:.1%}")

with tab2:
    inf = st.file_uploader("Infected cell images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="inf")
    uninf = st.file_uploader("Uninfected cell images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="uninf")
    if st.button("Calculate accuracy"):
        tp = fn = tn = fp = 0
        for f in inf:
            if prob_uninfected(Image.open(f)) < 0.5:
                tp += 1
            else:
                fn += 1
        for f in uninf:
            if prob_uninfected(Image.open(f)) >= 0.5:
                tn += 1
            else:
                fp += 1
        total = tp + fn + tn + fp
        if total == 0:
            st.warning("Upload some images first.")
        else:
            st.subheader(f"Accuracy: {(tp + tn) / total:.1%} ({tp + tn} of {total} correct)")
            st.table(pd.DataFrame([[tp, fn], [fp, tn]],
                index=["Truly infected", "Truly uninfected"],
                columns=["Predicted infected", "Predicted uninfected"]))
