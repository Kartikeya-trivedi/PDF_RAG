import os
import requests
import fitz  # PyMuPDF
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
import textwrap
import pandas as pd
import numpy as np
import torch
import re

pdf_path = "JavaNotesForProfessionals.pdf"
download_url = "https://drive.usercontent.google.com/u/1/uc?id=14-D03qPjelRFCwWWZiOu3EkRPb5EsQHT&export=download"

if not os.path.exists(pdf_path):
    print("File doesn't exist, downloading...")
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        print(f"The file has been downloaded and saved as {pdf_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
else:
    print(f"File {pdf_path} exists.")

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number,
            "text": text
        })
    return pages_and_texts

def print_wrapped(text):
    print(textwrap.fill(text, width=80))

def process_text(pages_and_texts):
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts

def chunk_sentences(pages_and_texts, num_sentence_chunk_size=10):
    def split_list(input_list, slice_size):
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def create_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\\.([A-Z])', r'. \\1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def embed_text(pages_and_chunks):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks if item["chunk_token_count"] > 30]
    text_chunk_embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
    for i, item in enumerate(pages_and_chunks):
        if item["chunk_token_count"] > 30:
            item["embedding"] = text_chunk_embeddings[i].tolist()
    return pages_and_chunks

def save_embeddings(pages_and_chunks):
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv("text_chunks_and_embeddings_df.csv", index=False)

def load_embeddings():
    df = pd.read_csv("text_chunks_and_embeddings_df.csv")
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to("cuda")
    return pages_and_chunks, embeddings

def print_top_results_and_scores(query, embeddings, pages_and_chunks):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    top_results_dot_product = torch.topk(dot_scores, k=5)
    print(f"Query: '{query}'\n")
    print("Results:")
    for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
        print(f"Score: {score.item():.4f}")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print("\n")

def main():
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_texts = process_text(pages_and_texts)
    pages_and_texts = chunk_sentences(pages_and_texts)
    pages_and_chunks = create_chunks(pages_and_texts)
    pages_and_chunks = embed_text(pages_and_chunks)
    save_embeddings(pages_and_chunks)
    pages_and_chunks, embeddings = load_embeddings()
    query = "what happens in switch case without break"
    print_top_results_and_scores(query, embeddings, pages_and_chunks)

if __name__ == "__main__":
    main()