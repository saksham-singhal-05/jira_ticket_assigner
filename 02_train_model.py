import yaml
import psycopg2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config and fetch data
    cfg = load_config()
    conn = psycopg2.connect(**cfg['postgres'])
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding, fixed_by FROM issues
        WHERE embedding IS NOT NULL AND fixed_by IS NOT NULL
    """)
    data = cur.fetchall()
    conn.close()

    if not data:
        print("No training data found.")
        return

    # Convert embeddings and labels to arrays
    X = np.array([row[0] for row in data], dtype=np.float32)
    y = [row[1] for row in data]

    # Encode labels and split train/test
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.05, random_state=42)

    # Try XGBoost with GPU, fallback to CPU if not available
    try:
        xgb_model = xgb.XGBClassifier(
            tree_method='gpu_hist',
            subsample=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.01,
            colsample_bytree=0.8
        )
        # Dummy fit to check GPU availability
        xgb_model.fit(np.zeros((1, X.shape[1])), [0])
        print("Using GPU with 'gpu_hist'")
    except Exception as e:
        print(f"GPU not available or error occurred: {e}")
        print("Falling back to CPU with 'hist'")
        xgb_model = xgb.XGBClassifier(
            tree_method='hist',
            subsample=0.8,
            reg_lambda=1.5,
            reg_alpha=0.1,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.01,
            colsample_bytree=0.8
        )

    # Train XGBoost model
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'xgb_model.joblib')

    # Train SVC model
    svc_model = SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='linear')
    svc_model.fit(X_train, y_train)
    joblib.dump(svc_model, 'svc_model.joblib')

    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.3,
        max_depth=10,
        bootstrap=True
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'rf_model.joblib')

    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder.joblib')

    print(f"Trained models on {len(y)} records and saved them.")

if __name__ == '__main__':
    main()
