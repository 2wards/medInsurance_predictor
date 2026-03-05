from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import os

app = Flask(__name__)

# ---------------- LOAD DATA ---------------- #
DATA_PATH = os.path.join(os.getcwd(), "insurance.csv")
data = pd.read_csv(DATA_PATH)

# Encode categorical features
data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data['region'] = data['region'].map({
    'southwest': 0,
    'southeast': 1,
    'northwest': 2,
    'northeast': 3
})

feature_names = ["age", "sex", "bmi", "children", "smoker", "region"]

X = data[feature_names]
y_reg = data["charges"]
y_clf = (data["charges"] > 20000).astype(int)

X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ---------------- LOAD MODELS ---------------- #
reg_model = joblib.load("models/regression_models.pkl")
clf_model = joblib.load("models/classification_models.pkl")

# ---------------- HELPER FUNCTION ---------------- #
def create_plot(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return send_file(img, mimetype='image/png')

# ---------------- HOME ROUTE ---------------- #
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        age = int(request.form.get("age"))
        sex = int(request.form.get("sex"))
        bmi = float(request.form.get("bmi"))
        children = int(request.form.get("children"))
        smoker = int(request.form.get("smoker"))
        region = int(request.form.get("region"))

        input_features = np.array([[age, sex, bmi, children, smoker, region]])

        reg_prediction = reg_model.predict(input_features)[0]
        clf_prediction = clf_model.predict(input_features)[0]

        risk = "HIGH RISK" if clf_prediction == 1 else "LOW RISK"

        # Feature importance only if reg_model has it
        if hasattr(reg_model, "feature_importances_"):
            importances = reg_model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            feature_importance = []

        result = {
            "charges": round(reg_prediction, 2),
            "risk": risk,
            "features": feature_importance[:3]
        }

    return render_template("landing.html", result=result)

# ---------------- DATASET PREVIEW ---------------- #
@app.route("/dataset-preview")
def dataset_preview():
    table = data.head(50).to_html(index=False)
    return render_template("dataset.html", table=table)

# ---------------- 1️⃣ MODEL PERFORMANCE ---------------- #
@app.route("/model-performance")
def model_performance():
    r_pred = reg_model.predict(X_test)
    r2 = r2_score(y_test_reg, r_pred)

    c_pred = clf_model.predict(X_test)
    acc = accuracy_score(y_test_clf, c_pred)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(["Regression R2","Classification Accuracy"], [r2, acc], color=["cyan","orange"])
    ax.set_ylim(0,1)
    ax.set_title("Model Performance Overview")
    return create_plot(fig)

# ---------------- 2️⃣ CONFUSION MATRIX ---------------- #
@app.route("/confusion-matrix")
def confusion_matrix_graph():
    y_pred = clf_model.predict(X_test)
    cm = confusion_matrix(y_test_clf, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xticklabels(["Low Risk","High Risk"])
    ax.set_yticklabels(["Low Risk","High Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return create_plot(fig)

# ---------------- 3️⃣ FEATURE IMPORTANCE ---------------- #
@app.route("/feature-importance")
def feature_importance_graph():
    if not hasattr(reg_model, "feature_importances_"):
        return "Feature importance not available"
    importances = reg_model.feature_importances_
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(feature_names, importances, color="gold")
    ax.set_title("Feature Importance")
    ax.set_xticklabels(feature_names, rotation=20)
    return create_plot(fig)

# ---------------- 4️⃣ CHARGES DISTRIBUTION ---------------- #
@app.route("/charges-distribution")
def charges_distribution():
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(data["charges"], bins=30, color="purple")
    ax.set_title("Charges Distribution")
    return create_plot(fig)

# ---------------- 5️⃣ CORRELATION HEATMAP ---------------- #
@app.route("/correlation-heatmap")
def correlation_heatmap():
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation")
    return create_plot(fig)

# ---------------- 6️⃣ AGE vs CHARGES ---------------- #
@app.route("/age-vs-charges")
def age_vs_charges():
    fig, ax = plt.subplots(figsize=(8,5))
    scatter = ax.scatter(data["age"], data["charges"], c=data["charges"], cmap="plasma")
    fig.colorbar(scatter, ax=ax, label="Charges")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    ax.set_title("Age vs Charges Analysis")
    return create_plot(fig)

# ---------------- 7️⃣ SMOKER IMPACT ---------------- #
@app.route("/smoker-impact")
def smoker_impact():
    smoker_avg = data.groupby("smoker")["charges"].mean()
    labels = ["Non-Smoker", "Smoker"]
    values = smoker_avg.values
    fig, ax = plt.subplots(figsize=(6,5))
    ax.bar(labels, values, color=["green","red"])
    ax.set_title("Average Charges: Smoker vs Non-Smoker")
    return create_plot(fig)

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets this automatically
    app.run(host="0.0.0.0", port=port)
