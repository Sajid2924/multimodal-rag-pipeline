import json

INPUT_PATH = "data/chunks.json"
OUTPUT_PATH = "data/chunked.json"

chunk_size = 400  # maximum length of one chunk
overlap = 50

# breaks the text into chunks according to CHUNK_SIZE & OVERLAP
def chunk_text(text, size=chunk_size, overlap=overlap):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_chunks = []  # final chunks of text

    for item in data:
        content = item["content"]
        modality = item["modality"]
        page = item["page"]

        # text and image to chunk
        if modality in ["text", "image"]:
            pieces = chunk_text(content)
            for p in pieces:
                final_chunks.append({
                    "content": p,
                    "modality": modality,
                    "page": page
                })

        # table as its separate chunk
        elif modality == "table":
            final_chunks.append({
                "content": content,
                "modality": modality,
                "page": page
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunking complete. Total chunks: {len(final_chunks)}")


if __name__ == "__main__":
    main()
