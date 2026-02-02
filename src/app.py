import streamlit as st
import os
import re
import shutil
import subprocess
import tempfile
from loader import EmbeddingLoader
from chat import ChatEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_git_repository(path: str) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def _format_git_error(stderr: str, has_token: bool) -> str:
    """Format git error message with helpful hints."""
    error_msg = stderr.strip()
    
    # Detect authentication errors
    auth_error_indicators = [
        "could not read Username",
        "Authentication failed",
        "fatal: could not read Password",
        "Repository not found",
        "403",
        "401",
    ]
    
    is_auth_error = any(indicator in error_msg for indicator in auth_error_indicators)
    
    if is_auth_error:
        if not has_token:
            return (
                f"**Authentication required.**\n\n"
                f"Please set `GIT_ACCESS_TOKEN` in docker-compose.yml and restart the container.\n\n"
                f"Original error: {error_msg}"
            )
        else:
            return (
                f"**Authentication failed.**\n\n"
                f"Please check that your `GIT_ACCESS_TOKEN` is valid and has the required permissions.\n\n"
                f"Original error: {error_msg}"
            )
    
    return error_msg


def git_clone_with_lfs(repo_url: str, target_path: str) -> tuple[bool, str]:
    """Clone a git repository including LFS files.
    
    Returns a tuple of (success, message).
    """
    git_token = os.getenv("GIT_ACCESS_TOKEN", "")
    
    try:
        # If token is provided, embed it in the URL for authentication
        clone_url = repo_url
        if git_token and repo_url.startswith("https://") and "@" not in repo_url:
            clone_url = repo_url.replace("https://", f"https://{git_token}@")
        
        # Check if target directory exists and is not empty
        if os.path.exists(target_path) and os.listdir(target_path):
            # Clone to a temporary location first, then move contents
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_clone_path = os.path.join(temp_dir, "repo")
                
                # Clone to temp directory
                result = subprocess.run(
                    ["git", "clone", clone_url, temp_clone_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    return False, _format_git_error(result.stderr, bool(git_token))
                
                # Pull LFS files
                result = subprocess.run(
                    ["git", "lfs", "pull"],
                    cwd=temp_clone_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.warning(f"Git LFS pull warning: {result.stderr}")
                
                # Remove existing contents in target (except hidden files we want to keep)
                for item in os.listdir(target_path):
                    item_path = os.path.join(target_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                # Move cloned contents to target
                for item in os.listdir(temp_clone_path):
                    src = os.path.join(temp_clone_path, item)
                    dst = os.path.join(target_path, item)
                    shutil.move(src, dst)
        else:
            # Target doesn't exist or is empty, clone directly
            os.makedirs(target_path, exist_ok=True)
            
            result = subprocess.run(
                ["git", "clone", clone_url, target_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return False, _format_git_error(result.stderr, bool(git_token))
            
            # Pull LFS files
            result = subprocess.run(
                ["git", "lfs", "pull"],
                cwd=target_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning(f"Git LFS pull warning: {result.stderr}")
        
        # Restore the original remote URL (without token) for security
        if git_token:
            subprocess.run(
                ["git", "remote", "set-url", "origin", repo_url],
                cwd=target_path,
                capture_output=True,
                text=True
            )
        
        # Get the current commit info
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h - %s (%cr)"],
            cwd=target_path,
            capture_output=True,
            text=True
        )
        commit_info = result.stdout.strip() if result.returncode == 0 else ""
        
        return True, f"Repository cloned successfully!\n\n**Latest commit:** {commit_info}"
        
    except FileNotFoundError:
        return False, "Git is not installed in the container."
    except Exception as e:
        logger.exception("Git clone error")
        return False, f"Error: {str(e)}"


def configure_git_credentials(repo_path: str, token: str) -> bool:
    """Configure git to use the provided access token for authentication.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Get the remote URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False
        
        remote_url = result.stdout.strip()
        
        # If URL is HTTPS and doesn't already have credentials, add the token
        if remote_url.startswith("https://") and "@" not in remote_url:
            # Convert https://github.com/... to https://token@github.com/...
            authenticated_url = remote_url.replace("https://", f"https://{token}@")
            
            # Set the remote URL with embedded token
            result = subprocess.run(
                ["git", "remote", "set-url", "origin", authenticated_url],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        
        return True
    except Exception as e:
        logger.warning(f"Failed to configure git credentials: {e}")
        return False


def restore_git_remote_url(repo_path: str, original_url: str) -> None:
    """Restore the original remote URL (without embedded token)."""
    try:
        subprocess.run(
            ["git", "remote", "set-url", "origin", original_url],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
    except Exception as e:
        logger.warning(f"Failed to restore git remote URL: {e}")


def git_pull_with_lfs(repo_path: str) -> tuple[bool, str]:
    """Pull latest changes from git repository including LFS files.
    
    Returns a tuple of (success, message).
    """
    git_token = os.getenv("GIT_ACCESS_TOKEN", "")
    original_remote_url = None
    
    try:
        # Check if the directory is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Not a git repository: {repo_path}"
        
        # Get original remote URL before modifying
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            original_remote_url = result.stdout.strip()
        
        # Configure git credentials if token is provided
        if git_token:
            configure_git_credentials(repo_path, git_token)
        
        # Fetch all updates including LFS
        result = subprocess.run(
            ["git", "fetch", "--all"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Git fetch failed: {result.stderr}"
        
        # Pull latest changes
        result = subprocess.run(
            ["git", "pull"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return False, f"Git pull failed: {result.stderr}"
        
        pull_output = result.stdout.strip()
        
        # Pull LFS files to ensure all large files are downloaded
        result = subprocess.run(
            ["git", "lfs", "pull"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            # LFS pull failure is not critical if there are no LFS files
            logger.warning(f"Git LFS pull warning: {result.stderr}")
        
        # Get the current commit info
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h - %s (%cr)"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        commit_info = result.stdout.strip() if result.returncode == 0 else ""
        
        message = f"Update successful!\n\n**Latest commit:** {commit_info}"
        if "Already up to date" in pull_output:
            message = f"Already up to date.\n\n**Current commit:** {commit_info}"
        
        return True, message
        
    except FileNotFoundError:
        return False, "Git is not installed in the container."
    except Exception as e:
        logger.exception("Git pull error")
        return False, f"Error: {str(e)}"
    finally:
        # Restore original remote URL to avoid storing token in git config
        if original_remote_url and git_token:
            restore_git_remote_url(repo_path, original_remote_url)


def get_model_config_from_env():
    """Load model configuration from environment variables.
    
    Returns a tuple of (config_dict, is_configured) where is_configured is True
    if at least CHAT_PROVIDER is set via environment.
    """
    chat_provider = os.getenv("CHAT_PROVIDER")
    
    if not chat_provider:
        return None, False
    
    # Chat configuration
    chat_api_key = os.getenv("CHAT_API_KEY", "")
    chat_model = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
    chat_azure_endpoint = os.getenv("CHAT_AZURE_ENDPOINT", "")
    chat_api_version = os.getenv("CHAT_API_VERSION", "2023-05-15")
    
    # Embedding configuration
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", chat_provider)
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    embedding_azure_endpoint = os.getenv("EMBEDDING_AZURE_ENDPOINT", "")
    embedding_api_version = os.getenv("EMBEDDING_API_VERSION", "2023-05-15")
    
    # Build chat kwargs
    chat_kwargs = {}
    if chat_provider == "Azure OpenAI":
        chat_kwargs["azure_endpoint"] = chat_azure_endpoint
        chat_kwargs["api_version"] = chat_api_version
    
    # Build embedding kwargs
    embedding_kwargs = {}
    if embedding_provider == "Azure OpenAI":
        embedding_kwargs["azure_endpoint"] = embedding_azure_endpoint
        embedding_kwargs["api_version"] = embedding_api_version
    
    # Use chat API key for embedding if not explicitly set
    final_embedding_key = embedding_api_key if embedding_api_key else chat_api_key
    
    config = {
        "chat_provider": chat_provider,
        "chat_api_key": chat_api_key,
        "chat_model": chat_model,
        "chat_kwargs": chat_kwargs,
        "embedding_provider": embedding_provider,
        "embedding_api_key": final_embedding_key,
        "embedding_model": embedding_model,
        "embedding_kwargs": embedding_kwargs,
    }
    
    return config, True


def render_math(content: str) -> str:
    """Convert LaTeX math notation to Streamlit-compatible format.
    
    Converts:
    - \\[...\\] -> $$...$$ (block math)
    - \\(...\\) -> $...$ (inline math)
    """
    # Convert block math: \[...\] -> $$...$$
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    # Convert inline math: \(...\) -> $...$
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content, flags=re.DOTALL)
    return content

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

# Check if model configuration is provided via environment variables
env_config, models_preconfigured = get_model_config_from_env()

if models_preconfigured:
    # Use configuration from environment variables
    chat_provider = env_config["chat_provider"]
    chat_api_key = env_config["chat_api_key"]
    chat_model = env_config["chat_model"]
    chat_kwargs = env_config["chat_kwargs"]
    embedding_provider = env_config["embedding_provider"]
    final_embedding_key = env_config["embedding_api_key"]
    embedding_model_name = env_config["embedding_model"]
    embedding_kwargs = env_config["embedding_kwargs"]
    
    # Show a notice in the sidebar that models are pre-configured
    st.sidebar.info("Model configuration is set via environment variables.")
    st.sidebar.write(f"**Chat Provider:** {chat_provider}")
    st.sidebar.write(f"**Chat Model:** {chat_model}")
    st.sidebar.write(f"**Embedding Provider:** {embedding_provider}")
    st.sidebar.write(f"**Embedding Model:** {embedding_model_name}")
else:
    # Sidebar Config - show full configuration UI
    st.sidebar.header("Chat Configuration")
    chat_provider = st.sidebar.selectbox("Chat Provider", ["OpenAI", "Azure OpenAI", "Anthropic", "Mock"])
    chat_api_key = st.sidebar.text_input("Chat API Key", type="password")

    chat_kwargs = {}
    default_llm = "gpt-3.5-turbo"

    if chat_provider == "Azure OpenAI":
        st.sidebar.subheader("Azure Chat Settings")
        chat_azure_endpoint = st.sidebar.text_input("Chat Azure Endpoint", help="https://your-resource.openai.azure.com/")
        chat_api_version = st.sidebar.text_input("Chat API Version", value="2023-05-15")
        # For Azure, the model name is often the deployment name
        default_llm = "my-gpt-deployment"
        chat_kwargs["azure_endpoint"] = chat_azure_endpoint
        chat_kwargs["api_version"] = chat_api_version
    elif chat_provider == "Anthropic":
        default_llm = "claude-3-opus-20240229"

    chat_model = st.sidebar.text_input("Chat Model / Deployment Name", value=default_llm)

    st.sidebar.divider()

    st.sidebar.header("Embedding Configuration")
    st.sidebar.info("Must match the provider/model used to generate the index.")
    embedding_provider = st.sidebar.selectbox("Embedding Provider", ["OpenAI", "Azure OpenAI", "Mock"])
    embedding_api_key = st.sidebar.text_input("Embedding API Key", type="password", help="Leave empty if same as Chat API Key (if provider matches).")

    embedding_kwargs = {}
    embedding_default_model = "text-embedding-ada-002"

    if embedding_provider == "Azure OpenAI":
        st.sidebar.subheader("Azure Embedding Settings")
        # If users reuse Chat Azure settings, they can just copy/paste or we could add a checkbox "Same as Chat".
        # For flexibility, let's keep separate but maybe default to empty and logic handle it? No, explicit is better.
        emb_azure_endpoint = st.sidebar.text_input("Embedding Azure Endpoint", help="https://your-resource.openai.azure.com/")
        emb_api_version = st.sidebar.text_input("Embedding API Version", value="2023-05-15")
        embedding_default_model = "my-embedding-deployment"
        embedding_kwargs["azure_endpoint"] = emb_azure_endpoint
        embedding_kwargs["api_version"] = emb_api_version

    embedding_model_name = st.sidebar.text_input("Embedding Model / Deployment Name", value=embedding_default_model)

    # Resolve Keys
    final_embedding_key = embedding_api_key if embedding_api_key else chat_api_key

# Git Update Section (only shown if GIT_REPO_URL is configured)
git_repo_url = os.getenv("GIT_REPO_URL")
if git_repo_url:
    st.sidebar.divider()
    st.sidebar.header("Source Management")
    
    # Initialize session state for update/clone status
    if "git_status" not in st.session_state:
        st.session_state.git_status = None
    if "git_message" not in st.session_state:
        st.session_state.git_message = None
    
    # Check if repo_path is a git repository
    is_git_repo = is_git_repository(repo_path)
    
    if is_git_repo:
        # Show Update button if it's already a git repo
        if st.sidebar.button("🔄 Update Sources", help="Pull latest changes from git repository", use_container_width=True):
            with st.sidebar:
                with st.spinner("Updating sources..."):
                    success, message = git_pull_with_lfs(repo_path)
                    st.session_state.git_status = "success" if success else "error"
                    st.session_state.git_message = message
                    
                    if success:
                        # Clear the cached engine so it reloads with new data
                        load_engine.clear()
    else:
        # Show Clone button if it's not a git repo
        st.sidebar.warning(f"'{repo_path}' is not a git repository.")
        st.sidebar.write(f"**Repository URL:** {git_repo_url}")
        
        # Check if access token is configured
        git_token = os.getenv("GIT_ACCESS_TOKEN", "")
        if not git_token:
            st.sidebar.info("💡 **Tip:** If this is a private repo, set `GIT_ACCESS_TOKEN` in docker-compose.yml")
        
        if st.sidebar.button("📥 Clone Repository", help="Clone the repository to initialize the data directory", use_container_width=True):
            with st.sidebar:
                with st.spinner("Cloning repository... This may take a while for large repos."):
                    success, message = git_clone_with_lfs(git_repo_url, repo_path)
                    st.session_state.git_status = "success" if success else "error"
                    st.session_state.git_message = message
                    
                    if success:
                        # Clear the cached engine so it reloads with new data
                        load_engine.clear()
                        st.rerun()
    
    # Display git operation status
    if st.session_state.git_status == "success":
        st.sidebar.success(st.session_state.git_message)
    elif st.session_state.git_status == "error":
        st.sidebar.error(st.session_state.git_message)

# Load Index
with st.spinner("Loading index..."):
    engine = load_engine(repo_path)

if not engine:
    st.error(f"No embeddings found in {repo_path}/.copilot-index. Please generate embeddings first.")
    st.stop()

# Index Status Section
st.sidebar.divider()
st.sidebar.header("Index Status")
st.sidebar.success(f"Loaded {len(engine.chunks)} chunks.")

# Initialize session state for reload status
if "reload_status" not in st.session_state:
    st.session_state.reload_status = None

# Reload Index button
if st.sidebar.button("🔃 Reload Index", help="Reload the embedding index from disk", use_container_width=True):
    with st.sidebar:
        with st.spinner("Reloading index..."):
            load_engine.clear()
            st.session_state.reload_status = "success"
            st.rerun()

# Display reload status
if st.session_state.reload_status == "success":
    st.sidebar.info("Index reloaded successfully!")
    # Clear the status after displaying
    st.session_state.reload_status = None

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(render_math(message["content"]))

if prompt := st.chat_input("Ask a question about the repo"):
    # Validation
    if chat_provider != "Mock" and not chat_api_key:
        st.error("Please provide a Chat API Key.")
    elif embedding_provider != "Mock" and not final_embedding_key:
        st.error("Please provide an Embedding API Key (or use Chat API Key if same provider).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(render_math(prompt))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Get embedding
                    query_embedding = engine.get_embedding(
                        prompt,
                        embedding_provider,
                        final_embedding_key,
                        embedding_model=embedding_model_name,
                        **embedding_kwargs
                    )

                    # 2. Search
                    results = engine.search(query_embedding)

                    # 3. Generate
                    response = engine.generate_response(
                        prompt,
                        results,
                        chat_provider,
                        chat_api_key,
                        model=chat_model,
                        **chat_kwargs
                    )

                    st.markdown(render_math(response))

                    with st.expander("Retrieved Context"):
                        for r in results:
                            st.markdown(f"**Source:** {r.source}")
                            st.markdown(render_math(r.content[:500] + ("..." if len(r.content) > 500 else "")))
                            st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Chat error")
