import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

def build_rag_pipeline(file_path: str):
    # 1. Load and Chunk Document
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    # 2. Embed and Store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. Load LLM (SmolLM-Instruct)
    pipe = pipeline(
        "text-generation", 
        model="HuggingFaceTB/SmolLM-135M-Instruct", 
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 4. Define Prompt Template
    template = """
    <|im_start|>system
    You are a helpful financial assistant. Answer the user's question using ONLY the provided context.
    <|im_end|>
    <|im_start|>user
    Context: {context}

    Question: {question}
    <|im_end|>
    <|im_start|>assistant
    """
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

if __name__ == "__main__":
    # Create dummy file for testing
    with open("test_report.txt", "w") as f:
        f.write("Q3 2023 Financial Report - Apex Financial Corp. Revenue for Q3 increased by 15% to $45 million, driven by a surge in our fraud detection API subscriptions. The new LSTM model deployment reduced false positives by 30%.")

    chain = build_rag_pipeline("test_report.txt")
    query = "What was the Q3 revenue and how did the LSTM model help?"
    
    print(f"Query: {query}")
    response = chain.invoke(query)
    
    # Clean output
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
        
    print(f"Answer: {response}")
