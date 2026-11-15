import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Malaria cell classifier", page_icon="🔬")

LOW, HIGH = 0.35, 0.65  # infected probability between these = inconclusive


@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("malaria.keras")
    m.predict(np.zeros((1, 128, 128, 3), dtype="float32"), verbose=0)  # warm-up
    return m


model = load_model()


def predict(img):
    start = time.perf_counter()
    x = np.array(img.convert("RGB").resize((128, 128)), dtype="float32")[None]
    p_inf = 1.0 - float(model.predict(x, verbose=0)[0][0])
    ms = (time.perf_counter() - start) * 1000
    return p_inf, ms


def confidence_label(conf):
    if conf >= 0.90:
        return "High confidence"
    if conf >= 0.70:
        return "Moderate confidence"
    return "Low confidence"


def safe_div(a, b):
    return a / b if b else None


def pct(v):
    return "n/a" if v is None else f"{v:.1%}"


st.title("🔬 Malaria cell classifier")
st.caption("AI-assisted screening of single blood cell images (thin smear). Student research demo, not for clinical use.")

tab1, tab2, tab3 = st.tabs(["Single image", "Batch accuracy", "About"])

with tab1:
    source = st.radio("Image source", ["Upload a file", "Take a photo"], horizontal=True)
    if source == "Upload a file":
        f = st.file_uploader("Upload one cell image", type=["png", "jpg", "jpeg"], key="single")
    else:
        st.caption("Tip: photograph one cell on a screen, close up and in focus. Photos of a whole slide will not work well, because the model was trained on single-cell crops.")
        f = st.camera_input("Take a photo")

    if f:
        img = Image.open(f)
        p_inf, ms = predict(img)
        p_unin = 1 - p_inf
        conf = max(p_inf, p_unin)
        left, right = st.columns([1, 2])
        with left:
            st.image(img, caption=getattr(f, "name", "camera photo"), width=180)
        with right:
            if p_inf >= HIGH:
                st.error("Result: likely INFECTED (malaria parasite pattern detected)")
            elif p_inf <= LOW:
                st.success("Result: likely UNINFECTED (no parasite pattern detected)")
            else:
                st.warning("Result: INCONCLUSIVE. The model cannot decide. Refer this cell to a trained technician.")
            st.metric("Model confidence", f"{conf:.1%}")
            st.caption(f"{confidence_label(conf)}  |  Analysed in {ms:.0f} ms")
        st.write("Probability breakdown")
        st.progress(float(p_inf), text=f"Infected: {p_inf:.1%}")
        st.progress(float(p_unin), text=f"Uninfected: {p_unin:.1%}")
        st.info("This is a screening aid. Results should be confirmed by a trained technician using a microscope.")

with tab2:
    st.write("Upload cells whose true label you already know. The app compares its predictions with your labels and scores itself.")
    inf = st.file_uploader("Infected cell images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="inf")
    uninf = st.file_uploader("Uninfected cell images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="uninf")

    if st.button("Calculate accuracy"):
        items = [(x, "Infected") for x in inf] + [(x, "Uninfected") for x in uninf]
        if not items:
            st.warning("Upload some images first.")
        else:
            rows, images = [], []
            bar = st.progress(0, text="Analysing images...")
            for i, (x, true_label) in enumerate(items):
                im = Image.open(x)
                im.load()
                p, ms = predict(im)
                pred = "Infected" if p >= 0.5 else "Uninfected"
                rows.append({"File": x.name, "True label": true_label, "Predicted": pred,
                             "Confidence (%)": round(max(p, 1 - p) * 100, 1),
                             "Flag": "Inconclusive" if LOW < p < HIGH else "Confident",
                             "Time (ms)": round(ms, 1), "Correct": pred == true_label})
                images.append(im)
                bar.progress((i + 1) / len(items), text=f"Analysing image {i + 1} of {len(items)}")
            bar.empty()
            st.session_state["batch"] = (pd.DataFrame(rows), images)

    if "batch" in st.session_state:
        df, images = st.session_state["batch"]
        tp = int(((df["True label"] == "Infected") & (df["Predicted"] == "Infected")).sum())
        fn = int(((df["True label"] == "Infected") & (df["Predicted"] == "Uninfected")).sum())
        fp = int(((df["True label"] == "Uninfected") & (df["Predicted"] == "Infected")).sum())
        tn = int(((df["True label"] == "Uninfected") & (df["Predicted"] == "Uninfected")).sum())
        total = len(df)
        acc = (tp + tn) / total
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        prec = safe_div(tp, tp + fp)
        n_inc = int((df["Flag"] == "Inconclusive").sum())
        confident = df[df["Flag"] == "Confident"]
        acc_conf = float(confident["Correct"].mean()) if len(confident) else None
        avg_ms = float(df["Time (ms)"].mean())

        st.subheader("Performance summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", pct(acc), help="Share of all cells classified correctly")
        c2.metric("Sensitivity", pct(sens), help="Of the truly infected cells, how many were caught")
        c3.metric("Specificity", pct(spec), help="Of the truly uninfected cells, how many were correctly cleared")
        c4.metric("Precision", pct(prec), help="Of the cells flagged infected, how many really were infected")
        d1, d2, d3 = st.columns(3)
        d1.metric("Inconclusive cells", n_inc, help="Cells where the model was between 35% and 65% sure")
        d2.metric("Accuracy on confident cells", pct(acc_conf), help="Accuracy after leaving out the inconclusive cells")
        d3.metric("Avg time per image", f"{avg_ms:.0f} ms", help="Preprocessing plus prediction, on the server running this app")

        scores = {k: v for k, v in [("Accuracy", acc), ("Sensitivity", sens), ("Specificity", spec), ("Precision", prec)] if v is not None}
        st.bar_chart(pd.DataFrame({"Score": scores}))

        st.subheader("Confusion matrix")
        a, b = st.columns(2)
        a.success(f"Infected cells caught: **{tp}**")
        b.error(f"Infected cells missed: **{fn}**")
        c, d = st.columns(2)
        c.warning(f"False alarms (healthy cells flagged infected): **{fp}**")
        d.success(f"Healthy cells correctly cleared: **{tn}**")
        st.table(pd.DataFrame([[tp, fn], [fp, tn]],
                              index=["Truly infected", "Truly uninfected"],
                              columns=["Predicted infected", "Predicted uninfected"]))

        st.subheader("What this means")
        st.write(f"The model classified {tp + tn} of {total} cells correctly ({pct(acc)}).")
        if sens is not None:
            st.write(f"It caught {tp} of {tp + fn} infected cells ({pct(sens)}). The {fn} missed cell(s) are the most serious kind of error in a screening tool.")
        if spec is not None:
            st.write(f"It correctly cleared {tn} of {tn + fp} healthy cells ({pct(spec)}), with {fp} false alarm(s).")
        if n_inc and acc_conf is not None:
            st.write(f"{n_inc} cell(s) fell in the inconclusive band. Leaving those out, accuracy on the remaining cells is {pct(acc_conf)}. This shows the benefit of referring unsure cells to a technician instead of forcing an answer.")
        elif n_inc == 0:
            st.write("No cells were inconclusive: the model was confident about every image.")
        st.write("With a small set of images, one wrong cell shifts these percentages a lot, so treat this as an illustration rather than a final accuracy.")

        st.subheader("Image by image")
        show = df.copy()
        show["Correct"] = show["Correct"].map({True: "✓", False: "✗"})
        st.dataframe(show, hide_index=True)

        wrong = df.index[~df["Correct"]].tolist()
        if wrong:
            with st.expander(f"Show the {len(wrong)} wrongly classified cell(s)"):
                cols = st.columns(4)
                for k, i in enumerate(wrong[:8]):
                    cols[k % 4].image(images[i], width=120,
                                      caption=f"True: {df.loc[i, 'True label']}, predicted: {df.loc[i, 'Predicted']} ({df.loc[i, 'Confidence (%)']}%)")

        st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv", "text/csv")

with tab3:
    st.subheader("About this tool")
    st.write("**What it does:** looks at one red blood cell image and estimates whether it shows a malaria parasite.")
    st.write("**Model:** MobileNetV2 (a small neural network) retrained with transfer learning on the NIH malaria dataset of about 27,500 single-cell images from thin blood smears.")
    st.write("**Accuracy:** about 93% on a held-out test set of 4,134 cells.")
    st.write("**Inconclusive results:** if the model is between 35% and 65% sure a cell is infected, the app says so instead of forcing an answer, and the cell should be referred to a technician.")
    st.write("**Speed:** each prediction shows how long the analysis took on the server running this app.")
    st.subheader("Limitations")
    st.write("- It only handles single-cell images, not whole slide photos.")
    st.write("- It was trained on one dataset (thin smears, mostly P. falciparum), so it may do worse on images from other microscopes or stains. Camera photos in particular differ from the clean dataset images.")
    st.write("- It is a research prototype and cannot diagnose a patient.")
