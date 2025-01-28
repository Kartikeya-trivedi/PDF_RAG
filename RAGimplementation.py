import os
import random
import re
import fitz
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from huggingface_hub import notebook_login
from google.colab import files
import textwrap

import matplotlib.pyplot as plt

def install_dependencies():
    if "COLAB_GPU" in os.environ:
        os.system("pip install -U torch PyMuPDF tqdm sentence-transformers accelerate bitsandbytes flash-attn")

def upload_files():
    uploaded ="JavaForProfessionals.pdf"
    return uploaded

def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()

def open_and_read_pdf(uploaded: dict) -> list:
    filename = list(uploaded.keys())[0]
    doc = fitz.open(stream=uploaded[filename], filename=filename)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({
            "page_number": page_number - 25,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "page text": text
        })
    return pages_and_texts

def process_text(pages_and_texts):
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["page text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts

def split_list(input_list: list, chunk_size: int) -> list:
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def chunk_sentences(pages_and_texts, chunk_size=10):
    for item in tqdm(pages_and_texts):
        item["sentences_chunks"] = split_list(item["sentences"], chunk_size=chunk_size)
    return pages_and_texts

def create_chunks(pages_and_texts):
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentences_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks

def embed_text(pages_and_chunks, model_name="all-mpnet-base-v2"):
    embedding_model = SentenceTransformer(model_name_or_path=model_name, device="cuda")
    for item in tqdm(pages_and_chunks):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])
    return pages_and_chunks

def save_embeddings(pages_and_chunks, save_path="text_chunks_and_embeddings_df.csv"):
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(save_path, index=False)

def load_embeddings(load_path="text_chunks_and_embeddings_df.csv"):
    df = pd.read_csv(load_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to("cuda")
    return pages_and_chunks, embeddings

def retrieve_relevant_resources(query, embeddings, model, n_resources_to_return=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices

def print_top_results_and_scores(query, embeddings, pages_and_chunks, n_resources_to_return=5):
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=embedding_model, n_resources_to_return=n_resources_to_return)
    print(f"Query: {query}\n")
    print("Results:")
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[index]['page_number']}\n")

def print_wrapped(text):
    print(textwrap.fill(text, width=80))

def main():
    install_dependencies()
    uploaded = upload_files()
    pages_and_texts = open_and_read_pdf(uploaded)
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