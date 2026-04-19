# pinecone_upload.py
import json
import time
import sys
import traceback
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM
HF_MODEL = "google/embeddinggemma-300m"  # tested model
HF_TOKEN = config.HF_TOKEN

# -----------------------------
# Initialize clients
# -----------------------------
print("Initializing Hugging Face client...")
client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

print("Initializing Pinecone client...")
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Helper: safe list indexes
# -----------------------------
def get_existing_index_names():
    try:
        raw = pc.list_indexes()
    except Exception as e:
        print("Error calling pc.list_indexes():", e)
        raise

    names = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                names.append(item["name"])
            elif isinstance(item, str):
                names.append(item)
            else:
                try:
                    names.append(item.get("name") if hasattr(item, "get") else str(item))
                except Exception:
                    names.append(str(item))
    else:
        names = getattr(raw, "names", lambda: [])()
    return names

# -----------------------------
# Create index if not exists
# -----------------------------
def ensure_index():
    existing_indexes = get_existing_index_names()
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index '{INDEX_NAME}' with dim={VECTOR_DIM}...")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(2)
        except Exception as e:
            print("Failed to create index. Error:")
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)

# -----------------------------
# Helper: get embeddings
# -----------------------------
def get_embeddings(texts):
    """Get embeddings for a list of texts using Hugging Face Inference API."""
    embeddings = []
    for text in texts:
        response = client.feature_extraction(text)
        if isinstance(response, list) and isinstance(response[0], list):
            embeddings.append(response[0])
        else:
            embeddings.append(response)
    return embeddings

# -----------------------------
# Utility: chunk list
# -----------------------------
def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    print(f"Loading data from: {DATA_FILE}")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Prepared {len(items)} items for upload")

    # Ensure Pinecone index
    index = ensure_index()

    # Upload in batches
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        for attempt in range(1, 4):
            try:
                index.upsert(vectors=vectors)
                break
            except Exception as e:
                print(f"Upsert failed (attempt {attempt}/3): {e}")
                if attempt == 3:
                    traceback.print_exc()
                    raise
                time.sleep(1 + attempt * 2)

    print("All items uploaded successfully to Pinecone!")

# -----------------------------
if __name__ == "__main__":
    main()
