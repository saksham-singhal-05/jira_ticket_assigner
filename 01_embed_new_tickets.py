import yaml
import psycopg2
import numpy as np
import os

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def fetch_new_issues(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT issue_key, summary, description FROM issues
        WHERE embedding IS NULL
    """)
    return cur.fetchall()

def save_embedding(conn, issue_key, emb_vector):
    cur = conn.cursor()
    cur.execute("UPDATE issues SET embedding = %s WHERE issue_key = %s", (emb_vector, issue_key))
    conn.commit()

def get_transformer_embedder(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def get_openai_embeddings(texts, api_key, model):
    import openai
    openai.api_key = api_key
    response = openai.embeddings.create(
        model=model,
        input=texts
    )
    return [e.embedding for e in response.data]

def main():
    cfg = load_config()
    conn = psycopg2.connect(**cfg['postgres'])

    new_issues = fetch_new_issues(conn)
    if not new_issues:
        print("No new issues found.")
        return

    texts = [((s or '') + (d or '')) for _, s, d in new_issues]
    issue_keys = [ik for ik, _, _ in new_issues]

    if cfg['embedding_provider'] == 'openai':
        embeddings = get_openai_embeddings(texts, cfg['openai']['api_key'], cfg['openai']['embedding_model'])
    else:
        model = get_transformer_embedder(cfg['transformers']['model_name'])
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

    for ik, vec in zip(issue_keys, embeddings):
        save_embedding(conn, ik, vec)

    print(f"Processed {len(issue_keys)} new tickets.")
    conn.close()

if __name__ == '__main__':
    main()
