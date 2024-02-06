# app.py

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
import docx
import faiss

# Load pre-trained embeddings model
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

# Function to extract text and generate embeddings
def extract_text_and_generate_embeddings(file):
    text = ""

    # Extract text from the uploaded file
    if file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    elif file.type == "application/pdf":
        text = extract_text(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    # Tokenize and generate embeddings
    embeddings = model.encode(text)
    return embeddings

# Function to calculate cosine similarity
def calculate_cosine_similarity(embeddings1, embeddings2):
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity

# Function to store embeddings in Faiss
def store_embeddings_in_faiss(embeddings):
    # Initialize the index
    d = len(embeddings[0])  # dimension of the embeddings
    index = faiss.IndexFlatIP(d)  # IndexFlatIP for inner product similarity

    # Add embeddings to the index
    index.add(np.array(embeddings, dtype=np.float32))

    return index

# Main function
def main():
    st.title("Document Similarity Checker")

    # File upload
    uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt", "pdf", "docx"])
    uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt", "pdf", "docx"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        st.write("Documents uploaded successfully!")

        # Extract text and generate embeddings for both files
        embeddings1 = extract_text_and_generate_embeddings(uploaded_file1)
        embeddings2 = extract_text_and_generate_embeddings(uploaded_file2)

        # Store embeddings in Faiss
        index1 = store_embeddings_in_faiss(embeddings1)
        index2 = store_embeddings_in_faiss(embeddings2)

        # Check if files are identical
        if np.array_equal(embeddings1, embeddings2):
            st.write("Files are identical.")
        else:
            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(embeddings1, embeddings2)

            # Display similarity percentage
            similarity_percentage = round(similarity * 100, 2)
            st.write(f"Similarity Percentage: {similarity_percentage}%")

if __name__ == "__main__":
    main()
