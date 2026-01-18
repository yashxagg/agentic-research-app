import os
import tempfile
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(uploaded_file):
    """
    Loads PDF, CSV, DOCX, XLSX, or TXT files.
    Returns a list of split documents.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        docs = []
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(file_path=tmp_path, encoding='utf-8')
            docs = loader.load()
        elif uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_path, encoding='utf-8')
            docs = loader.load()
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(tmp_path)
            # Convert dataframe rows to documents
            for _, row in df.iterrows():
                content = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                docs.append(Document(page_content=content, metadata={"source": uploaded_file.name}))
        else:
            raise ValueError("Unsupported file type")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return splits
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
