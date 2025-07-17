import yaml
import psycopg2
import numpy as np
import pandas as pd

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    conn = psycopg2.connect(**cfg['postgres'])
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding, fixed_by FROM issues
        WHERE embedding IS NOT NULL AND fixed_by IS NOT NULL
    """)
    data = cur.fetchall()
    if not data:
        print("No training data found.")
        return
    X = np.array([row[0] for row in data], dtype=np.float32)
    y = [row[1] for row in data]
    np.save('embeddings.npy', X)
    pd.Series(y).to_csv('labels.csv', index=False, header=False)
    print(f"Saved {len(y)} training records.")
    conn.close()

if __name__ == '__main__':
    main()
