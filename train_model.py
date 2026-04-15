import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("predictive_maintenance.csv")

data = data.drop(['UDI', 'Product ID'], axis=1, errors='ignore')
data.columns = data.columns.str.strip()

target_col = "Machine failure" if "Machine failure" in data.columns else data.columns[-1]

X = data.drop(target_col, axis=1)
y = data[target_col]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)


