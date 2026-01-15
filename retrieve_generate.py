import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

INDEX_PATH = "data/faiss.index"
META_PATH = "data/metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = 5

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment")
openai.api_key = api_key

# reloads the vector database and its metadata
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def hybrid_retrieve(query, faiss_index, metadata, embed_model, bm25, k_dense=5, k_sparse=5):
    # embedding of query text
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    # Compares query vector with all document vectors, return top k_dense matches
    _, dense_ids = faiss_index.search(q_emb, k_dense)
    dense_results = [metadata[i] for i in dense_ids[0]]

    # Splits query into words, Scores exact term overlap, sort indices by score
    bm25_scores = bm25.get_scores(query.lower().split())
    top_sparse_ids = np.argsort(bm25_scores)[::-1][:k_sparse]
    sparse_results = [metadata[i] for i in top_sparse_ids]
    merged = {id(c): c for c in dense_results + sparse_results}

    return list(merged.values())


# Baseline dense retriever (FAISS only).
def retrieve(query, index, metadata, model):
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    _, indices = index.search(q_emb, TOP_K)
    return [metadata[i] for i in indices[0]]

# filters and reorders chunks selected by the hybrid retriever based on their relevance to the query.
def rerank(query, contexts, top_n=3):
    pairs = [(query, c["content"]) for c in contexts]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(contexts, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return [c for c, _ in ranked[:top_n]]

# returns the final prompt with the instructions required
def build_prompt(query, contexts):
    context_text = ""
    pages = set()

    for c in contexts:
        context_text += f"(Page {c['page']}) {c['content']}\n\n"
        pages.add(c["page"])

    prompt = f"""
You are a document-based QA assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT invent numbers.
- For figures or charts, you MAY describe visible trends such as increase, decrease, recovery, or stabilization if clearly shown.
- If the answer is not explicitly supported by the context, respond with exactly: "Not found in document."
- Do not add external knowledge.
- Do not add explanations beyond the answer.

Context:
{context_text}

Question:
{query}

Answer with citations like (Page X).
"""
    return prompt, sorted(list(pages))


# calls API for generating the response
def generate_answer(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer strictly from the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response["choices"][0]["message"]["content"]

# LLM evaluation step of the generated response (optional)
def verify_answer_with_llm(query, answer, contexts):
    context_text = "\n".join(c["content"] for c in contexts)

    verification_prompt = f"""
You are a strict fact checker.

QUESTION:
{query}

ANSWER:
{answer}

CONTEXT:
{context_text}

Task:
Is every claim in the ANSWER explicitly supported by the CONTEXT?
If YES, respond with "SUPPORTED".
If NO, respond with "NOT SUPPORTED".

Respond with ONE WORD only.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": verification_prompt}],
        temperature=0
    )

    verdict = response["choices"][0]["message"]["content"].strip()
    return verdict == "SUPPORTED"

# checking if the question is related to context provided (optional)
def is_question_supported(query, contexts):
    context_text = "\n".join(c["content"] for c in contexts)

    prompt = f"""
You are checking whether a question is answerable
using the provided document context.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context_text}

Task:
Is the QUESTION explicitly about information
that appears in the DOCUMENT CONTEXT?

Respond with ONLY:
YES or NO
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response["choices"][0]["message"]["content"].strip() == "YES"


def main():
    index, metadata = load_index()    # Load vector index + metadata
    model = SentenceTransformer(MODEL_NAME)    # Load embedding model
    bm25 = BM25Okapi(                          # sparse retrieval
        [c["content"].lower().split() for c in metadata]
    )
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit": break

        contexts = hybrid_retrieve(query, index, metadata, model, bm25)
        contexts = rerank(query, contexts)
        if not is_question_supported(query, contexts):        # answer not in context
            print("\nANSWER:")
            print("Not found in document.")
            print("\nSOURCES: None")
            continue
        
        prompt, pages = build_prompt(query, contexts)
        answer = generate_answer(prompt)

        if not verify_answer_with_llm(query, answer, contexts):
            print("\nANSWER:")
            print("Not found in document.")
            print("\nSOURCES: None")
        else:
            print("\nANSWER:")
            print(answer)
            print(f"\nSOURCES: Pages {pages}")
       

if __name__ == "__main__":
    main()
