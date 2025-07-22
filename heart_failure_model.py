import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('C:/Users/gvino/Downloads/heart_failure_clinical_records_dataset (1).csv')

# Drop weak features
X = df.drop(['DEATH_EVENT', 'sex', 'smoking', 'anaemia'], axis=1)
y = df['DEATH_EVENT']

# Columns to scale
cols_to_scale = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                 'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# Scale features
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Boosting model with tuned hyperparameters
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # Aim ≥ 0.80

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
