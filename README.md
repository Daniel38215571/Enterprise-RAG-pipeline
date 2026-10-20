# Enterprise RAG Pipeline: Financial Document QA

A production-grade Retrieval-Augmented Generation (RAG) pipeline built to query financial documents using open-source models. This project demonstrates semantic search, chunking strategies, and LLM orchestration using LangChain, FAISS, and HuggingFace.

## 🏗️ Architecture

1. **Ingestion:** Financial text is loaded and split using `RecursiveCharacterTextSplitter` (Chunk Size: 100, Overlap: 20). This ensures granular retrieval without losing context.
2. **Embeddings:** Uses HuggingFace's `all-MiniLM-L6-v2` to convert text chunks into 384-dimensional dense vectors.
3. **Vector Database:** FAISS (Facebook AI Similarity Search) is used for ultra-fast, in-memory L2 similarity search.
4. **Generation:** Retrieves the top-2 chunks and passes them to `SmolLM-135M-Instruct`, a lightweight instruction-tuned LLM, using a custom prompt template to strictly ground the answer in the retrieved context.

## 📊 Evaluation Note
The pipeline was designed to support Ragas evaluation (Faithfulness, Answer Relevance). Due to CPU constraints in the mobile Colab environment, the final Ragas loop was bypassed locally; it is ready for GPU or API-backed execution.

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python rag_pipeline.py
