import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.docstore.document import Document

# Load OpenAI key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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

# Create FAISS index
@st.cache_resource
def create_faiss_index(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

# Search FAQ first
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
    st.set_page_config(page_title="Raast FAQ Bot 💬", layout="centered")
    st.title("📌 Raast Bank FAQ Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your question about Raast...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Searching..."):
            docs = load_faq_docs()
            index = create_faiss_index(docs)
            answer = search_answer(user_input, index)

            if answer:
                response = f"✅ {answer}"
            else:
                response = ask_gpt(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()




