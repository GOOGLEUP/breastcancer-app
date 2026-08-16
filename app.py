from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model"
FALLBACK_TEST_FILE = ROOT / "test_data.csv"

FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

MODEL_CHOICES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

st.set_page_config(
    page_title="Classification Model Lab",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Classification Model Lab")
st.markdown(
    "### BITS Pilani — Machine Learning Assignment 2\n"
    "Compare saved classifiers on the supplied holdout data."
)

with st.sidebar:
    st.header("Model controls")
    chosen_model = st.selectbox(
        "Choose a classifier",
        options=list(MODEL_CHOICES)
    )

    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type="csv",
        help="The file must contain the 30 predictor columns and diagnosis."
    )

    st.divider()
    st.caption("Positive class used for Precision, Recall, F1 and AUC: malignant")


@st.cache_resource
def fetch_model(file_name):
    with open(MODEL_PATH / file_name, "rb") as stream:
        return pickle.load(stream)


def read_test_data(upload):
    if upload is None:
        return pd.read_csv(FALLBACK_TEST_FILE)
    return pd.read_csv(upload)


def calculate_scores(model, frame):
    actual = frame["diagnosis"].astype(str)
    predictors = frame[FEATURE_NAMES]

    predicted = model.predict(predictors)
    probabilities = model.predict_proba(predictors)

    classes = list(model.classes_)
    malignant_column = classes.index("malignant")

    actual_binary = (actual == "malignant").astype(int)
    predicted_binary = (pd.Series(predicted) == "malignant").astype(int)

    scores = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(
            actual_binary,
            probabilities[:, malignant_column]
        ),
        "Precision": precision_score(
            actual, predicted, pos_label="malignant"
        ),
        "Recall": recall_score(
            actual, predicted, pos_label="malignant"
        ),
        "F1 Score": f1_score(
            actual, predicted, pos_label="malignant"
        ),
        "MCC": matthews_corrcoef(
            actual_binary, predicted_binary
        ),
    }

    return actual, predicted, probabilities[:, malignant_column], scores


data = read_test_data(uploaded_file)
required = set(FEATURE_NAMES + ["diagnosis"])
absent = sorted(required.difference(data.columns))

if absent:
    st.error("The CSV is missing these required columns:")
    st.code(", ".join(absent))
    st.stop()

classifier = fetch_model(MODEL_CHOICES[chosen_model])
actual, predicted, malignant_probability, scores = calculate_scores(
    classifier, data
)

st.success(
    f"Evaluation completed for **{chosen_model}** using "
    f"{len(data)} test observations."
)

metric_columns = st.columns(6)
for column, (label, value) in zip(metric_columns, scores.items()):
    column.metric(label, f"{value:.4f}")

left_panel, right_panel = st.columns(2)

with left_panel:
    st.subheader("Confusion matrix")
    class_order = ["benign", "malignant"]
    matrix = confusion_matrix(actual, predicted, labels=class_order)

    figure, axis = plt.subplots()
    image = axis.imshow(matrix)
    axis.set_xticks(range(2), class_order)
    axis.set_yticks(range(2), class_order)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(chosen_model)

    for row in range(2):
        for col in range(2):
            axis.text(col, row, matrix[row, col], ha="center", va="center")

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    st.pyplot(figure)
    plt.close(figure)

with right_panel:
    st.subheader("Classification report")
    report = classification_report(
        actual,
        predicted,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True
    )

st.subheader("Test-set predictions")

result = data.copy()
result["predicted_diagnosis"] = predicted
result["malignant_probability"] = malignant_probability

st.dataframe(result, use_container_width=True)

st.download_button(
    label="Export predictions as CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="classification_predictions.csv",
    mime="text/csv"
)
