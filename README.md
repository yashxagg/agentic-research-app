![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
# 🕵️‍♂️ Agentic Research Analyst
**Lightning-fast document intelligence powered by Groq & LangChain.**

---

### 🎯 Overview
This project is an **Agentic RAG (Retrieval-Augmented Generation)** application that allows users to chat with complex PDF documents. By leveraging the **Groq LPU** for sub-second inference and **LangChain** for agentic reasoning, the system provides grounded answers with precise source citations.

---

### 🚀 Key Features
* **Sub-Second Responses:** Powered by **Llama-3.3-70B** on Groq for high-speed research.
* **Grounded Citations:** Automatically extracts and displays the exact page number and text snippet from the PDF.
* **Conversational Memory:** Remembers the context of your research session for follow-up questions.
* **Privacy Focused:** Uses local embeddings (**HuggingFace**) and local vector storage (**ChromaDB**).

---

### 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM Engine** | Groq (Llama-3.3-70B) |
| **Framework** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **UI** | Streamlit |

---

### 📂 Repository Structure

```text
📦 agentic-research-app
┣ 📄 app.py              # Main application logic & UI
┣ 📄 requirements.txt    # Project dependencies
┣ 📄 .env                # API Keys (Git-ignored 🛡️)
┣ 📄 .gitignore          # Security rules
┣ 📄 LICENSE
┗ 📂 src/                # Modular processing logic
  ┣ 📜 document_loader.py
  ┗ 📜 vector_store.py
```
---


## ⚙️ Installation & Usage
### 1. Clone the repository.
```bash
git clone https://github.com/yashxagg/agentic-research-app.git
cd agentic-research-app
```
### 2. Install dependencies.
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit App:
```bash
streamlit run app.py
```

---

## 👤 Author
* **Yash Aggarwal**
  * 🎓 B.Tech CSE (AI & ML) | Class of 2026
  * 🐙 [GitHub Profile](https://github.com/yashxagg)
  * 💼 [LinkedIn Profile](https://linkedin.com/in/yash-aggarwal0812)

