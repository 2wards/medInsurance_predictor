import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("insurance.csv")

# Encode categorical variables
data['sex'] = data['sex'].map({'female':0,'male':1})
data['smoker'] = data['smoker'].map({'no':0,'yes':1})
data['region'] = data['region'].map({
    'southwest':0,
    'southeast':1,
    'northwest':2,
    'northeast':3
})

# Features and targets
X = data.drop("charges", axis=1)
y_reg = data["charges"]
y_clf = (data["charges"] > 20000).astype(int)

# Split once properly
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# Use same split for classification
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ---------------- REGRESSION MODEL ---------------- #
reg_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
reg_model.fit(X_train, y_train_reg)

# ---------------- CLASSIFICATION MODEL ---------------- #
clf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
clf_model.fit(X_train, y_train_clf)

# Save models
joblib.dump(reg_model, "models/regression_models.pkl")
joblib.dump(clf_model, "models/classification_models.pkl")

print("✅ Models trained and saved successfully!")