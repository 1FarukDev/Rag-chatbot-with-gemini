# **RAG Chatbot with Gemini 2.5 Flash & Streamlit**

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline built with **Google Gemini 2.5 Flash**, **LangChain**, **ChromaDB**, and **Streamlit**.
It loads a PDF document, splits it into chunks, embeds the chunks using Google embeddings, stores them in Chroma, and retrieves the most relevant context to answer user questions.

The app also includes a Jupyter notebook demo:
`rag-gemini.ipynb`

---

## 🚀 **Features**

* Uses **Gemini 2.5 Flash** as the LLM.
* Uses **GoogleGenerativeAIEmbeddings** for embeddings.
* Loads PDF documents via **PyPDFLoader**.
* Splits text using **RecursiveCharacterTextSplitter**.
* Stores vectors and performs similarity search using **ChromaDB**.
* Beautiful chat interface powered by **Streamlit**.
* RAG pipeline built with:

  * `create_stuff_documents_chain`
  * `create_retrieval_chain`

---

## 📂 **Project Structure**

```
.
├── app.py
├── rag-gemini.ipynb
├── requirements.txt
├── all-of-statistics.pdf  # Example Knowledge Base
└── README.md
```

---

## 🧠 **How It Works**

### 1. **Document Loading**

```python
loader = PyPDFLoader("all-of-statistics.pdf")
data = loader.load()
```

### 2. **Chunking**

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)
```

### 3. **Embeddings**

Using Google’s text embedding model:

```python
embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004')
```

### 4. **Vector Store (ChromaDB)**

```python
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 10})
```

### 5. **LLM Setup**

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_tokens=500)
```

### 6. **RAG Chain**

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

### 7. **Streamlit Chat UI**

```python
query = st.chat_input('Ask me anything: ')
```

---

## 🧪 **Running the App**

### **1. Clone the repo**

```bash
git clone <your-repo-url>
cd <project>
```

### **2. Create environment (optional but recommended)**

```bash
python3 -m venv env
source env/bin/activate
```

### **3. Install requirements**

```bash
pip install -r requirements.txt
```

### **4. Add your environment variables**

Create a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key_here
```

(You can obtain one from Google AI Studio.)

### **5. Run Streamlit app**

```bash
streamlit run app.py
```

---

## 📦 **Dependencies**

Your `requirements.txt` includes:

```
langchain
langchain_community
langchain-google-genai
python-dotenv
streamlit
langchain_experimental
sentence-transformers
langchain_chroma
langchainhub
unstructured
```

---

## 📓 **Notebook Demo**

A Jupyter notebook (`rag-gemini.ipynb`) is included to demonstrate the RAG pipeline step-by-step for experimentation and testing.

---

## 📝 **System Prompt Used**

The system ensures the LLM answers only from retrieved context:

```
You are a helpful AI assistant that helps people find information about courses from the provided context.
If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
Use the following pieces of context to answer the question at the end.
Answer the question truthfully and as best as you can and keep it concise.
```

---

## 🛠 **Future Improvements**

* Add support for multiple document uploads.
* Add Pinecone / Weaviate / Qdrant vector store option.
* Implement conversation history memory.
* Add UI for swapping models or chunk sizes.
* Deploy to cloud (Streamlit Cloud, Vercel, or Firebase Hosting).

---

