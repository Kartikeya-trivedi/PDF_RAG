# Retrieval-Augmented Generation (RAG) Pipeline with Gemma-2B

This repository implements a Retrieval-Augmented Generation (RAG) pipeline from scratch, leveraging the Gemma-2B parameter model to enable efficient and accurate query resolution over a specific knowledge domain. The pipeline processes documents, creates embeddings, and integrates them with a retrieval-based system to augment large language model (LLM) responses.

## Workflow Overview

### 1. Document Preprocessing and Embedding Creation

- **Input Document**: A single PDF file, "Java for Professionals."
- **Preprocessing**: The text was extracted from the PDF and segmented into smaller chunks (e.g., groups of 10 sentences) to make them suitable for LLM context windows.
- **Embedding Generation**:
  - The chunks were converted into numerical representations (embeddings) using a pre-trained embedding model (e.g., sentence-transformers from Hugging Face).
  - Embeddings were stored in a Torch tensor for efficient retrieval. For scalability, the use of a vector database/index can be considered.

### 2. Query Processing and Retrieval

- **User Query**: Users input queries through code 
- **Embedding Search**: The query is embedded using the same embedding model as used for the document chunks. Relevant chunks are retrieved based on similarity scores.

### 3. Response Generation

- **LLM Integration**: The Gemma-2B model generates responses by combining the retrieved document passages with the query to provide accurate and contextually relevant answers.

### 4. Output\*\*: The pipeline generates text and resources based on the query

## Key Features

- **Document Parsing**: Extracts text from lengthy PDFs and prepares it for efficient querying.
- **Efficient Embedding Management**: Uses Torch tensors for fast similarity-based search, even with 100k+ embeddings.
- **RAG Integration**: Combines retrieval and generation to provide better contextual responses.
- **Custom LLM Integration**: The system leverages the Gemma-2B model for high-quality natural language responses.

## Requirements

- **Hardware**: The pipeline has been optimized for execution on local GPUs 
- **Libraries**:
  - PyTorch
  - Sentence-Transformers
  - Hugging Face Transformers
  - PDF parsing library (e.g., PyMuPDF)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure that your GPU drivers and CUDA toolkit are properly configured.

## Usage

1. **Prepare Data**:
   Place the "Java for Professionals" PDF in the `data/` directory.
2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
3. **Access the Interface**:
   Open your browser and navigate to `http://localhost:5000` to interact with the chat interface.

## Future Enhancements

- **Scalability**: Incorporate a vector database like FAISS or Pinecone for handling larger datasets.
- **Model Fine-tuning**: Fine-tune Gemma-2B on domain-specific data for improved performance.
- **Multi-document Support**: Extend the pipeline to support multiple PDFs and diverse document formats.
- **Advanced Query Interfaces**: Add features like voice input and more advanced filtering options.

## Acknowledgments

- **Gemma-2B**: Thanks to the creators of Gemma-2B for making this powerful LLM available.
- **Hugging Face**: For providing robust tools for embedding generation and LLM integration.
- **Open-Source Libraries**: Special thanks to the open-source community for providing the libraries used in this project.

Feel free to fork, contribute, or open issues to improve this pipeline!

