import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
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
    st.title("📌 Raast FAQ Chatbot")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, chat in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"**Q{i}: {chat['question']}**")
            st.markdown(f"🔹 A{i}: {chat['answer']}\n")

    # Input for new query
    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Searching..."):
            docs = load_faq_docs()
            index = create_faiss_index(docs)
            answer = search_answer(query, index)

            if answer:
                st.success(f"✅ {answer}")
            else:
                st.warning("No exact FAQ match found, asking OpenAI...")
                answer = ask_gpt(query)
                st.info(answer)

            # Save to chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": answer
            })

if __name__ == "__main__":
    main()
