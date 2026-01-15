import json
import faiss
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/chunked.json"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["content"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)   # load the embedding model

    embeddings = model.encode(    # create embeddings
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]     # embeddings = (no. of chunks * vector embedding dimension)
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)   # Add all vectors to the FAISS index.
    print(f"Vectors in FAISS index: {index.ntotal}")
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("Embedding + indexing complete.")


if __name__ == "__main__":
    main()
