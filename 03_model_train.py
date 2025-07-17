import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib

X = np.load('embeddings.npy')
y = pd.read_csv('labels.csv', header=None)[0]

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.05, random_state=42)

try:
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist',  # Try GPU
        subsample=0.8,
        reg_lambda=1.5,
        reg_alpha=0.1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        colsample_bytree=0.8
    )
    xgb_model.fit([[0, 0]], [0])  # Dummy data
    print("Using GPU with 'gpu_hist'")
except Exception as e:
    print(f"GPU not available or error occurred: {e}")
    print("Falling back to CPU with 'hist'")
    xgb_model = xgb.XGBClassifier(
        tree_method='hist',  # Fallback to CPU
        subsample=0.8,
        reg_lambda=1.5,
        reg_alpha=0.1,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.01,
        colsample_bytree=0.8
    )


xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgb_model.joblib')

svc_model = SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='linear')
svc_model.fit(X_train, y_train)
joblib.dump(svc_model, 'svc_model.joblib')

rf_model = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1,
                                 max_features=0.3, max_depth=10, bootstrap=True)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.joblib')

joblib.dump(label_encoder, 'label_encoder.joblib')

print("Models and encoder saved.")
