import joblib
import yaml
import numpy as np
import requests
import os

from sentence_transformers import SentenceTransformer

# import openai

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


try:
    cfg = load_config()
    svc_model = joblib.load("svc_model.joblib")
    rf_model = joblib.load("rf_model.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    
    JIRA_URL = cfg.get("JIRA_SITE")
    JIRA_EMAIL = cfg.get("EMAIL")
    JIRA_API_TOKEN = os.getenv("JIRA_API_KEY")

    model_to_use = cfg.get("model_to_use", 0)  # 0: svc, 1: rf, 2: xgb

    embedding_method = None
    embedding_model = None
    openai_client = None
    if cfg.get("embedding_provider") == "openai":
        import openai
        openai.api_key = cfg["openai"]["api_key"]
        embedding_method = "openai"
    else:
        embedding_model = SentenceTransformer(cfg["transformers"]["model_name"])
        embedding_method = "local"
except Exception as e:
    raise RuntimeError(f"Model or config loading failed: {e}")

def get_embedding(text):
    try:
        if embedding_method == "openai":
            import openai
            emb = openai.embeddings.create(
                model=cfg["openai"]["embedding_model"], input=[text]
            ).data[0].embedding
            return np.array(emb, dtype=np.float32)
        elif embedding_method == "local":
            return embedding_model.encode([text])[0]
        else:
            raise RuntimeError("No valid embedding method configured.")
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

def top_n_predictions(model, X, label_encoder, n=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        top_n_idx = np.argsort(proba)[::-1][:n]
        labels = label_encoder.inverse_transform(top_n_idx)
        scores = proba[top_n_idx]
        return list(zip(labels, scores))
    else:
        pred = model.predict(X)
        label = label_encoder.inverse_transform(pred)[0]
        return [(label, 1.0)]

model_map = {
    0: svc_model,
    1: rf_model,
    2: xgb_model
}

def get_top_developers(summary, description):
    combined_text = summary + " " + description
    emb = get_embedding(combined_text).reshape(1, -1)
    model = model_map.get(model_to_use, svc_model)
    preds = top_n_predictions(model, emb, label_encoder)
    return [label for label, _ in preds][:3]



def retrieve_latest_jira_tickets():
    
    jql = 'updated >= -1d'
    url = f"{JIRA_URL}/rest/api/3/search"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}

    all_tickets = []
    start_at = 0
    max_results = 50
    while True:
        params = {
            "jql": jql,
            "fields": "summary,description",
            "maxResults": max_results,
            "startAt": start_at,
        }
        response = requests.get(url, headers=headers, params=params, auth=auth)
        data = response.json()
        issues = data.get("issues", [])
        for issue in issues:
            all_tickets.append({
                "issue_id": issue["key"],
                "summary": issue["fields"].get("summary", ""),
                "description": issue["fields"].get("description", ""),
            })
        if len(issues) < max_results:
            break
        start_at += max_results
    return all_tickets


import psycopg2

def insert_ticket_to_db(cursor, issue_id, devs):
    devs += ['unknown'] * (3 - len(devs)) 
    cursor.execute(
        "INSERT INTO ticket_assignments (issue_id, dev1, dev2, dev3) VALUES (%s, %s, %s, %s);",
        (issue_id, devs[0], devs[1], devs[2])
    )


def main():
    cfg = load_config()
    conn = psycopg2.connect(**cfg['postgres'])
    cursor = conn.cursor()

    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS ticket_assignments
           (issue_id TEXT PRIMARY KEY, dev1 TEXT, dev2 TEXT, dev3 TEXT)'''
    )
    conn.commit()

    tickets = retrieve_latest_jira_tickets()
    for ticket in tickets:
        devs = get_top_developers(ticket["summary"], ticket["description"])
        insert_ticket_to_db(cursor, ticket["issue_id"], devs)
        print(f"Inserted {ticket['issue_id']} with developers {devs}")

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
