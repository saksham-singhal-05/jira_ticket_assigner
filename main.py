import streamlit as st
import joblib
import yaml
import numpy as np

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_embedding(text, cfg):
    if cfg['embedding_provider'] == 'openai':
        import openai
        openai.api_key = cfg['openai']['api_key']
        emb = openai.embeddings.create(model=cfg['openai']['embedding_model'], input=[text]).data[0].embedding
        return np.array(emb, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg['transformers']['model_name'])
        return model.encode([text])[0]

cfg = load_config()
xgb_model = joblib.load('xgb_model.joblib')
svc_model = joblib.load('svc_model.joblib')
rf_model = joblib.load('rf_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title("Jira Ticket Automated Assignee")
summary = st.text_input("Ticket Summary", "")
description = st.text_area("Ticket Description", "")

if st.button("Predict"):
    combined = (summary or '') + (description or '')
    emb = get_embedding(combined, cfg)
    emb = emb.reshape(1, -1)
    predictions = {
        "XGBoost": label_encoder.inverse_transform(xgb_model.predict(emb))[0],
        "SVC": label_encoder.inverse_transform(svc_model.predict(emb))[0],
        "Random Forest": label_encoder.inverse_transform(rf_model.predict(emb))[0],
    }
    for model, pred in predictions.items():
        st.success(f"{model}: {pred}")
