import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("/content/Mountain_Mining.csv")

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop('Mining_Allowed', axis=1)
y = df['Mining_Allowed']

# Save feature order (VERY IMPORTANT for Streamlit)
joblib.dump(X.columns.tolist(), "model_features.pkl")

# Save label encoders for categorical columns
joblib.dump(label_encoders, "label_encoders.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    max_depth=12
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "best_model.pkl")

print("✅ Model, features, and encoders saved successfully")
