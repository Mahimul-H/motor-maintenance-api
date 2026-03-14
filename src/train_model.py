import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, balanced_accuracy_score

# 1. Setup paths
DATA_PATH = "data/motor_maintenance_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "motor_model.pkl")

# Ensure the models folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_pipeline():
    print("🚀 Loading 100,000 rows of telemetry data...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Define Features and Target
    # We drop the 'failure' column to get our training features (X)
    X = df.drop(columns=['failure'])
    y = df['failure']
    
    # Feature engineering
    X = X.copy()
    X['voltage_deviation'] = (X['voltage_v'] - 230).abs()
    X['power_usage'] = X['voltage_v'] * X['current_a']
    X['thermal_stress'] = X['temp_c'] / 100
    
    # 3. Split (80% for training, 20% for testing the model's 'intelligence')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 Training Random Forest Model...")
    # Random Forest is excellent for sensor data because it handles outliers well
    model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate Performance
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Training Complete!")
    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC: {roc_auc:.2%}")
    print(f"F1-Score: {f1:.2%}")
    print(f"Balanced Accuracy: {balanced_acc:.2%}")
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))
    
    # 5. Export the 'Brain'
    joblib.dump(model, MODEL_PATH)
    print(f"\n Model saved to: {MODEL_PATH}")
    
    # Save metrics
    metrics = {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'timestamp': pd.Timestamp.now()
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.pkl")
    joblib.dump(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train_pipeline()