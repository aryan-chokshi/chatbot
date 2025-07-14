import pdfplumber
import csv
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Use a small, instruction-tuned model
generator = pipeline("text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")


model = SentenceTransformer('all-MiniLM-L6-v2')

def save_faiss_index(embedding_column, index_file="faiss_index.index"):
    # Convert list of vectors to NumPy array
    embeddings_array = np.array(embedding_column.tolist()).astype("float32")
    faiss.normalize_L2(embeddings_array)

    # Create FAISS index (L2 or cosine similarity)
    index = faiss.IndexFlatIP(embeddings_array.shape[1])  # or faiss.IndexFlatIP() for cosine-like similarity
    index.add(embeddings_array)

    # Save the index to disk
    faiss.write_index(index, index_file)
    print(f"FAISS index saved to: {index_file}")
    return index

def search_query(query, df, index, model, top_k=3):
    query_embedding = model.encode([query]).astype("float32")
    
    faiss.normalize_L2(query_embedding)
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]]  # Get top-k matched procedures
    return results

def generate_response(user_query, retrieved_text):
    prompt = f"Context:\n{retrieved_text}\n\nQuestion: {user_query}\nAnswer:"
    result = generator(prompt, max_new_tokens=100, do_sample=False)
    return result[0]["generated_text"]


# Open the PDF
pdf_path = '/Users/aryan/Downloads/GPWR Test_Procedure_Power_Increase_HSB_to_100%_rev2D.pdf'
output_csv = 'output.csv'

# Function to extract TOC entries
def extract_toc(pdf):
    toc_page = pdf.pages[4]  # Page 5 is index 4 (TOC page)
    toc_text = toc_page.extract_text()
    toc_entries = []

    # Use regex to extract procedure names and their numbers
    toc_lines = toc_text.split('\n')
    i = 0
    while i < len(toc_lines):
        line = toc_lines[i].strip()
        # Match lines that start with a number (e.g., "1 PREPARATION FOR POWER INCREASING")
        match = re.match(r'^(\d+)\s+(.*)', line)
        if match:
            procedure_number = match.group(1)
            procedure_name = match.group(2)
            
            # Check if the line ends with "..." followed by a number (page number)
            if not re.search(r'\.{3,}\s*\d+$', line):
                # If it doesn't, the next line is part of the same entry
                if i + 1 < len(toc_lines):
                    next_line = toc_lines[i + 1].strip()
                    procedure_name += " " + next_line
                    i += 1  # Skip the next line
            
            # Remove the "..." and page number from the procedure name
            procedure_name = re.sub(r'\.{3,}\s*\d+$', '', procedure_name).strip()
            toc_entries.append((procedure_number, procedure_name))
        i += 1
    
    return toc_entries

# Function to extract multi-line content for a given procedure number
def extract_content_for_procedure(pdf, procedure_number):
    content = []
    for page in pdf.pages[5:]:  # Start from page 6 (index 5) onwards
        page_text = page.extract_text()
        
        # Use regex to find all steps for the given procedure number (e.g., "1.1", "1.2", etc.)
        pattern = re.compile(rf'^{procedure_number}\.\d+\s+(.*?)(?=\n\d+\.\d+|\Z)', re.DOTALL | re.MULTILINE)
        matches = pattern.findall(page_text)
        
        if matches:
            # Remove page number lines (e.g., "Generic PWR Simulator Page 8 of 58")
            for match in matches:
                lines = match.split('\n')
                # Filter out lines that contain "Generic PWR Simulator Page"
                lines = [line for line in lines if "Generic PWR Simulator Page" not in line]
                
                # Remove the last line only if it starts with a procedure number (e.g., "2 Restore switchyard...")
                if len(lines) > 1:  # Ensure there are multiple lines
                    last_line = lines[-1].strip()
                    if re.match(r'^\d+\s+', last_line):  # If it starts with a number, remove it
                        lines = lines[:-1]  # Remove the last line
                
                content.append("\n".join(lines))  # Join the remaining lines
    
    return "\n".join(content)  # Join all steps into a single string

def generate_embeddings(text_list):
    return model.encode(text_list, show_progress_bar=True)

# Main function to create the CSV
def create_csv_with_embeddings(pdf_path, output_csv):
    with pdfplumber.open(pdf_path) as pdf:
        toc_entries = extract_toc(pdf)
        data = []
        
        # Extract procedure steps & generate embeddings
        for number, procedure_name in toc_entries:
            associated_steps = extract_content_for_procedure(pdf, number)
            data.append([procedure_name, associated_steps])

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Procedure Name", "Associated Steps"])

        # Generate embeddings for "Associated Steps"
        df["Embeddings"] = df["Associated Steps"].apply(lambda x: generate_embeddings([x])[0].tolist())

        # Save to CSV
        df.to_csv(output_csv, index=False)
        save_faiss_index(df["Embeddings"])
        print(f"CSV file '{output_csv}' has been created with embeddings!")

# Run the script
create_csv_with_embeddings(pdf_path, output_csv)

# Load the CSV with embeddings
df = pd.read_csv("output.csv")

# Convert the stringified list back to actual lists (if needed)
import ast
df["Embeddings"] = df["Embeddings"].apply(ast.literal_eval)

# Load FAISS index
index = faiss.read_index("faiss_index.index")

# Run a query
query = "how to increase nuclear power"
results = search_query(query, df, index, model)

retrieved_context = results.iloc[0]["Associated Steps"]
response = generate_response(query, retrieved_context)

print("\n Chatbot Response:\n")
print(response)