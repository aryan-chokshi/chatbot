import os
import re
import ast
import faiss
import numpy as np
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ================================
# Model setup (FLAN-T5 is seq2seq)
# ================================
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base"
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================================
# Utility
# ================================
def _truncate(text: str, max_chars: int = 2000) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]

def _infer(prompt: str,
           max_new_tokens: int = 192,
           temperature: float = 0.0,
           num_beams: int = 4) -> str:
    """
    Deterministic by default (beam search). Increase temperature and set num_beams=1 for sampling.
    """
    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0 and num_beams == 1),
        num_beams=num_beams,
        clean_up_tokenization_spaces=True,
    )
    return out[0]["generated_text"].strip()

# ================================
# FAISS helpers
# ================================
def save_faiss_index(embedding_column, index_file="faiss_index.index"):
    embeddings_array = np.array(embedding_column.tolist(), dtype="float32")
    faiss.normalize_L2(embeddings_array)
    index = faiss.IndexFlatIP(embeddings_array.shape[1])  # cosine-like with L2-normalized vectors
    index.add(embeddings_array)
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to: {index_file}")
    return index

def search_query(query, df, index, top_k=3):
    query_embedding = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[indices[0]]

# ================================
# Response generation
# ================================
def generate_response_rag(user_query: str, retrieved_text: str) -> str:
    retrieved_text = _truncate(retrieved_text, 2000)
    prompt = (
        "You are a nuclear procedure assistant. "
        "Answer the question using only the CONTEXT below. "
        "If the answer is not in the context, reply: \"I don't know based on the provided procedures.\""
        "\n\nCONTEXT:\n"
        f"{retrieved_text}\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )
    return _infer(prompt)

def generate_response_no_rag(user_query: str) -> str:
    prompt = (
        "You are a helpful technical assistant. Provide a concise, step-by-step answer.\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )
    return _infer(prompt)

# ================================
# PDF parsing
# ================================
def extract_toc(pdf):
    toc_page = pdf.pages[4]  # Page 5 (0-indexed)
    toc_text = toc_page.extract_text() or ""
    toc_entries = []
    toc_lines = toc_text.split("\n")
    i = 0
    while i < len(toc_lines):
        line = toc_lines[i].strip()
        match = re.match(r"^(\d+)\s+(.*)", line)
        if match:
            number = match.group(1)
            name = match.group(2)
            # If no trailing ... <page>, merge next line
            if not re.search(r"\.{3,}\s*\d+$", line) and i + 1 < len(toc_lines):
                nxt = toc_lines[i + 1].strip()
                name += " " + nxt
                i += 1
            name = re.sub(r"\.{3,}\s*\d+$", "", name).strip()
            toc_entries.append((number, name))
        i += 1
    return toc_entries

def extract_content_for_procedure(pdf, procedure_number):
    content = []
    for page in pdf.pages[5:]:  # from page 6 onward
        page_text = page.extract_text() or ""
        pattern = re.compile(rf"^{procedure_number}\.\d+\s+(.*?)(?=\n\d+\.\d+|\Z)", re.DOTALL | re.MULTILINE)
        matches = pattern.findall(page_text)
        for match in matches:
            lines = [ln for ln in match.split("\n") if "Generic PWR Simulator Page" not in ln]
            if len(lines) > 1 and re.match(r"^\d+\s+", lines[-1].strip()):
                lines = lines[:-1]
            chunk = "\n".join(lines).strip()
            if chunk:
                content.append(chunk)
    return "\n".join(content)

def generate_embeddings(text_list):
    return embedder.encode(text_list, show_progress_bar=True)

def create_csv_with_embeddings(pdf_path, output_csv, index_path="faiss_index.index"):
    with pdfplumber.open(pdf_path) as pdf:
        toc_entries = extract_toc(pdf)
        data = []
        for number, procedure_name in toc_entries:
            steps = extract_content_for_procedure(pdf, number)
            data.append([procedure_name, steps])

        df = pd.DataFrame(data, columns=["Procedure Name", "Associated Steps"])
        df["Embeddings"] = df["Associated Steps"].apply(lambda x: generate_embeddings([x])[0].tolist())
        df.to_csv(output_csv, index=False)
        save_faiss_index(df["Embeddings"], index_file=index_path)
        print(f"Wrote {output_csv} and {index_path}")

# ================================
# Main
# ================================
if __name__ == "__main__":
    pdf_path = "/Users/aryan/Downloads/GPWR Test_Procedure_Power_Increase_HSB_to_100%_rev2D.pdf"
    output_csv = "output.csv"
    index_path = "faiss_index.index"

    # Build artifacts only if missing
    if not (os.path.exists(output_csv) and os.path.exists(index_path)):
        create_csv_with_embeddings(pdf_path, output_csv, index_path=index_path)

    # Load artifacts
    df = pd.read_csv(output_csv)
    df["Embeddings"] = df["Embeddings"].apply(ast.literal_eval)
    index = faiss.read_index(index_path)

    # Query & compare
    query = "how to increase nuclear power"

    # RAG
    try:
        results = search_query(query, df, index, top_k=3)
        retrieved_context = str(results.iloc[0]["Associated Steps"])
    except Exception:
        retrieved_context = ""

    rag_answer = generate_response_rag(query, retrieved_context if retrieved_context else "No relevant context found.")
    no_rag_answer = generate_response_no_rag(query)

    print("\n================ RAG Response ================\n")
    print(rag_answer)

    print("\n================ Non-RAG Response ================\n")
    print(no_rag_answer)
