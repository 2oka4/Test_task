# Test task

This is a Multimodal RAG application that allows users to query across multiple articles from https://www.deeplearning.ai/the-batch/, using both textual and media content.

---

##  Features

-  **Web Scraping**: Automatically scrapes articles and images from multiple issues of *The Batch* using their URL structure.
-  **Vector Search**: Indexes textual and vision-derived content using `OllamaEmbeddings` and performs similarity search for relevant context.
-  **Multimodal**: Uses `OllamaLLM` with image bindings to extract contextual descriptions from article images.
-  **Chat QA Interface**: Ask any question about the content of the scraped issues and receive concise answers with 3-sentence summaries.

---

##  Approach

### 1. Multimodal Content Retrieval
Each issue page (e.g., https://www.deeplearning.ai/the-batch/issue-300/) is scraped using `BeautifulSoup` to extract:
- Article text (`<p>`, `<h1>`, `<li>`, etc.)
- All image URLs (via `<img src>` or `<img srcset>`)

Images are downloaded and saved to a local directory. Vision models are then used to "describe" images for textual indexing.

### 2. Chunking and Indexing
We use LangChain’s `RecursiveCharacterTextSplitter` to split raw content into 1000-character overlapping chunks. These are indexed using:
- **Embeddings**: `OllamaEmbeddings` based on `llama3.2`
- **Vector Store**: `InMemoryVectorStore` to keep everything fast and self-contained

### 3. Multimodal Model Use
We use **`OllamaLLM` with `gemma3:12b`** to:
- Process user questions (chat interface)
- Extract image descriptions with `model.bind(images=[...])`

### 4. Interactive Streamlit App
The frontend is a simple Streamlit UI:
- Text input to enter issue numbers (comma-separated)
- Chat input to ask questions
- Display answers from the RAG pipeline

---

##  Tools & Models Used

| Tool | Purpose |
|------|---------|
| `Streamlit` | UI for file upload, chat, and response rendering |
| `LangChain` | Orchestration layer for vector store, prompt, and model |
| `Ollama` | LLM + Embedding models (gemma3:12b + llam3.2) |
| `PIL` | Image preprocessing for model compatibility |
| `BeautifulSoup` | HTML parsing for web scraping |
| `requests` | HTTP requests for scraping and downloading assets |

---

##  How to Run

### Requirements
- Ollama running locally with `llama3.2` and `gemma3:12b` models downloaded
- `pip install -r requirements.txt`

### Instrunction
1. To download models (`llama3.2` and `gemma3:12b`), ollama should be instaleed (https://ollama.com/). After that in cmd the following commands should be executed:
- ollama pull gemma3:12b
- ollama pull llama3.2
2. Download the code of the RAG + requirements. In the cmd execute the following command:
-streamlit run ***Path to the downloaded code of the RAG***
