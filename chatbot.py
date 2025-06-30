import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

# Load and split FAQ data
def load_faq_docs():
    with open("faq.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    docs = []
    for block in raw_text.split("Q: "):
        if not block.strip():
            continue
        try:
            question, answer = block.strip().split("A:")
            docs.append(Document(page_content=answer.strip(), metadata={"question": question.strip()}))
        except ValueError:
            continue
    return docs

# Create or load FAISS index
def create_faiss_index(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Search with RAG
def search_answer(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=1)
    if docs:
        return docs[0].page_content
    return None

# Fallback to GPT-4o
def ask_gpt(query):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = llm([HumanMessage(content=query)])
    return response.content

# Streamlit UI
def main():
    st.set_page_config(page_title="Raast FAQ Bot 💬")
    st.title("📌 Raast Chatbot")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Searching..."):
            docs = load_faq_docs()
            index = create_faiss_index(docs)
            answer = search_answer(query, index)

            if answer:
                st.session_state.history.append(("You", query))
                st.session_state.history.append(("Bot", answer))
            else:
                response = ask_gpt(query)
                st.session_state.history.append(("You", query))
                st.session_state.history.append(("Bot", response))

    # Display chat history
    if st.session_state.history:
        st.markdown("### 💬 Chat History")
        for speaker, message in st.session_state.history:
            st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
