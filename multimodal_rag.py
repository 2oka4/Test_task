
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

figures_directory = 'multi_modal_rag/figures/' #directory for saving images
os.makedirs(figures_directory, exist_ok=True)

# Setup
if "vector_store" not in st.session_state:
    embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.vector_store = InMemoryVectorStore(embeddings)

if "scraped_issues" not in st.session_state:
    st.session_state.scraped_issues = set()

model = OllamaLLM(model="gemma3:12b")

#Web scrapping
def preprocess_image(file_path):
    try:
        img = Image.open(file_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")  # Ensure compatible mode
        img.save(file_path, format="PNG")
    except Exception as e:
        print(f"Could not preprocess {file_path}: {e}")

def extract_text(file_path):
    try:
        preprocess_image(file_path)
        model_with_image_context = model.bind(images=[file_path])
        return model_with_image_context.invoke("You are a precise image-to-text converter. Carefully analyze the image and extract all the structured content, including full tables, text, numbers, and visual layout. If the image contains a table, reproduce the entire table in markdown format, including headers and every value in each cell. If the image is a chart or diagram, describe all axes, labels, and data points. Be accurate and do not skip any detail.")
    except Exception as e:
        print(f"[WARN] Could not process image {file_path}: {e}")
        return "[Image could not be processed.]"

def scrape_article(issue_number):
    url = f"https://www.deeplearning.ai/the-batch/issue-{issue_number}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
    article_text = "\n\n".join(p.get_text() for p in paragraphs)

    images = soup.find_all('img')
    image_texts = []

    for idx, img in enumerate(images):
        # Try to get highest quality image using srcset
        srcset = img.get('srcset')
        if srcset:
            img_url = srcset.strip().split(',')[-1].split()[0]
        else:
            img_url = img.get('src')

        if not img_url or not img_url.startswith("http"):
            continue

        img_path = os.path.join(figures_directory, f"img_{issue_number}_{idx}.png")

        # Skip download if image already exists
        if os.path.exists(img_path):
            print(f"[INFO] Skipping already downloaded: {img_path}")
        else:
            try:
                img_data = requests.get(img_url).content
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                print(f"[INFO] Downloaded image: {img_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download image {img_url}: {e}")
                continue

        #Extract text from imag
        try:
            image_text = extract_text(img_path)
            image_texts.append(image_text)
        except Exception as e:
            print(f"[WARN] Failed to process image {img_path}: {e}")
            continue

    return article_text + "\n\n" + "\n\n".join(image_texts)

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def index_docs(texts):
    st.session_state.vector_store.add_texts(texts)

def retrieve_docs(query):
    return st.session_state.vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
issue_numbers_input = st.text_input("Enter comma-separated article numbers (e.g., 230,231,232):")

if issue_numbers_input:
    issue_numbers = [num.strip() for num in issue_numbers_input.split(',') if num.strip().isdigit()]
    with st.spinner("Scraping and analyzing articles..."):
        for issue in issue_numbers:
            if issue not in st.session_state.scraped_issues:
                content = scrape_article(issue)
                chunked_texts = split_text(content)
                index_docs(chunked_texts)
                st.session_state.scraped_issues.add(issue)

#Question answering
question = st.chat_input("Ask a question about the scraped issues:")
if question:
    st.chat_message("user").write(question)
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)
    st.chat_message("assistant").write(answer)