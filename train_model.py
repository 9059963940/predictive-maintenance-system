import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("predictive_maintenance.csv")

# Clean
data = data.drop(['UDI', 'Product ID'], axis=1, errors='ignore')
data.columns = data.columns.str.strip()

# Target column
target_col = "Machine failure" if "Machine failure" in data.columns else data.columns[-1]

X = data.drop(target_col, axis=1)
y = data[target_col]

X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "features.pkl")

