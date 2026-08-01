# 📄 CVChat – AI-Powered CV Question Answering System

CVChat is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload a curriculum vitae (CV) in PDF format and interact with it using natural language. Instead of manually searching through lengthy resumes, users can simply ask questions like:

* *Who is this person?*
* *What programming languages does the candidate know?*
* *Does the candidate have machine learning experience?*
* *What projects has the candidate completed?*

The system retrieves only the most relevant sections from the uploaded CV and generates near accurate, context-aware responses using a Large Language Model (LLM).

---

# ✨ Features

* 📄 Upload any CV in PDF format
* 🤖 AI-powered conversational interface
* 🔍 Retrieval-Augmented Generation (RAG)
* 🧠 Context-aware responses using semantic search
* 💬 Multi-turn conversation support
* ⚡ Streaming responses for better user experience
* 📚 Conversation memory with automatic summarization
* 🎯 Answers are restricted to information present in the uploaded CV
* 🚫 Hallucination reduction through prompt engineering

---

# 🏗 Project Architecture

```text
                User
                  │
                  ▼
         Upload CV (PDF)
                  │
                  ▼
          PDF Text Extraction
                  │
                  ▼
          Text Chunking
                  │
                  ▼
       Embedding Generation
                  │
                  ▼
        FAISS Vector Database
                  │
                  ▼
         User Question
                  │
                  ▼
      Similarity Search (Top-k)
                  │
                  ▼
     Retrieved Context + Chat History (summarized previous chat using llama-3.1-8b-instant)
                  │
                  ▼
      openai/gpt-oss-20b (Groq API)
                  │
                  ▼
          Streaming Response
```

---

# 🛠 Tech Stack

### Frontend

* Streamlit (mostly vibecoding)

### Backend

* Python

### AI / NLP

* LangChain
* Groq API
* Llama 3.1 8B Instant and openai/gpt-oss-20b

### Model Setup

* Llama 3.1 8B Instant and openai/gpt-oss-20b (dual model) setup to save tokens.

### Vector Database

* FAISS

### Embedding Model

* HuggingFace Sentence Transformers

### PDF Processing

* pymupdf4llm

---

# 📂 Project Structure

```text
CVChat/
│
├── app.py
├── llm.py
├── rag.py
├── memory.py
├── prompt.py
├── utils.py
├── requirements.txt
├── README.md
```

---

# ⚙ Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/CVChat.git

cd CVChat
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

---

# 💡 How It Works

1. User uploads a CV (PDF).
2. The PDF text is extracted.
3. Text is split into overlapping chunks.
4. Each chunk is converted into embeddings.
5. Embeddings are stored in a FAISS vector database.
6. When a question is asked:
  
   * Relevant chunks are retrieved using semantic similarity.
   * Summary of three previous answers. 
   * This conversation summary is included to maintain context.
   * The complete information and user query are passed to the main LLM.
7. The model generates a streamed response grounded only in the uploaded CV.

---

# 🧠 Prompt Design

The system prompt is designed to:

* Answer only from retrieved CV content.
* Avoid hallucinating information.
* State when requested information is unavailable.
* Maintain conversational context across multiple questions.
* Respond naturally while remaining faithful to the uploaded document.

---

# 🚀 Future Improvements

* Support multiple uploaded CVs
* Candidate comparison
* OCR support for scanned PDFs
* Multi-language CV support
* Docker deployment
* Cloud vector database integration

---

# 📸 Demo

(https://cvchat-ljykw9dfwzfiqvs4bcqma2.streamlit.app/)

---

# 📦 Requirements

Major libraries used:

* streamlit
* langchain
* langchain-community
* langchain-groq
* faiss-cpu
* sentence-transformers
* pymupdf4llm

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, submit issues, or create pull requests for improvements.

---

# 📄 License

This project is licensed under the MIT License.

---

## Author

**Anik Paul**

Artificial Intelligence • Machine Learning
