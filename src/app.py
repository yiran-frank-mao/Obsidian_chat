import streamlit as st
import os
from loader import EmbeddingLoader
from chat import ChatEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Repo Chatbot", layout="wide")

@st.cache_resource
def load_engine(repo_path):
    loader = EmbeddingLoader(repo_path)
    chunks = loader.load()
    if not chunks:
        return None
    return ChatEngine(chunks)

st.title("Repo Chatbot")

repo_path = os.getenv("REPO_PATH", "/data")
st.write(f"Reading from: {repo_path}")

# Sidebar Config
st.sidebar.header("Chat Configuration")
chat_provider = st.sidebar.selectbox("Chat Provider", ["OpenAI", "Anthropic", "Mock"])
chat_api_key = st.sidebar.text_input("Chat API Key", type="password")

default_llm = "gpt-3.5-turbo"
if chat_provider == "Anthropic":
    default_llm = "claude-3-opus-20240229"
chat_model = st.sidebar.text_input("Chat Model", value=default_llm)

st.sidebar.divider()

st.sidebar.header("Embedding Configuration")
st.sidebar.info("Must match the provider/model used to generate the index.")
embedding_provider = st.sidebar.selectbox("Embedding Provider", ["OpenAI", "Mock"])
embedding_api_key = st.sidebar.text_input("Embedding API Key", type="password", help="Leave empty if same as Chat API Key (if provider matches).")
embedding_model_name = st.sidebar.text_input("Embedding Model", value="text-embedding-ada-002")

# Resolve Keys
final_embedding_key = embedding_api_key if embedding_api_key else chat_api_key

# Load Index
with st.spinner("Loading index..."):
    engine = load_engine(repo_path)

if not engine:
    st.error(f"No embeddings found in {repo_path}/.copilot-index. Please generate embeddings first.")
    st.stop()

st.sidebar.success(f"Loaded {len(engine.chunks)} chunks.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the repo"):
    # Validation
    if chat_provider != "Mock" and not chat_api_key:
        st.error("Please provide a Chat API Key.")
    elif embedding_provider != "Mock" and not final_embedding_key:
        st.error("Please provide an Embedding API Key (or use Chat API Key if same provider).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Get embedding
                    query_embedding = engine.get_embedding(
                        prompt,
                        embedding_provider,
                        final_embedding_key,
                        embedding_model=embedding_model_name
                    )

                    # 2. Search
                    results = engine.search(query_embedding)

                    # 3. Generate
                    response = engine.generate_response(
                        prompt,
                        results,
                        chat_provider,
                        chat_api_key,
                        model=chat_model
                    )

                    st.markdown(response)

                    with st.expander("Retrieved Context"):
                        for r in results:
                            st.markdown(f"**Source:** {r.source}")
                            st.text(r.content[:500] + ("..." if len(r.content) > 500 else ""))
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Chat error")
