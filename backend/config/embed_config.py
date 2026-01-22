import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from enums.embed_type import EmbedType


class EmbeddingConnector:
    """
    Connector for various language models (LLMs) such as OpenAI GPT, GROQ AI,
    Azure OpenAI, and Ollama.
    """

    def __init__(self, model_name: str,
                embed_type: EmbedType = EmbedType.OPEN_AI,
                api_key: str = None,
                endpoint:str = None, # for Ollama
            ):
        
        # Try loading .env from backend directory first
        # In Docker, environment variables are injected via docker-compose env_file
        # This will load .env if it exists locally, but won't fail if it doesn't
        try:
            backend_dir = Path(__file__).parent.parent.parent
            env_path = backend_dir / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
            else:
                # Fallback to current directory
                load_dotenv(override=False)
        except Exception:
            # If .env loading fails, that's okay - Docker provides env vars via docker-compose
            pass

        self.model = model_name
        self.endpoint = endpoint
        self.embed_type = embed_type

        # Use provided API key first, then try environment variable
        if api_key:
            self.api_key = api_key
        else:
            # Try different environment variable names
            self.api_key = (
                os.getenv("EMBEDDING_API_KEY") or 
                os.getenv("OPENAI_API_KEY") or 
                os.getenv("API_KEY")
            )


    def __call__(self) -> object:
        """Instantiate and return the selected LangChain chat model."""
        if not self.model:
            raise ValueError("Model name is not defined in the LLM configuration.")

        needs_key = {
            EmbedType.OPEN_AI,
        }
        if self.embed_type in needs_key and not self.api_key:
            raise ValueError(f"API key is not defined for provider '{self.embed_type.value}'.")

      
        try:
            if self.embed_type == EmbedType.OPEN_AI:
                return self.get_openai_embedding()
            if self.embed_type == EmbedType.HUGGINGFACE:
                return self.get_huggingface_embedding()
            if self.embed_type == EmbedType.OLLAMA:
                return self.get_ollama_embedding()
           
        except Exception as exc:  # pragma: no cover - defensive logging
            raise ValueError(f"Failed to initialise Embedding: {exc}") from exc
        
    def get_openai_embedding(self) -> OpenAIEmbeddings:
        """Return a configured OpenAI embedding client."""
        return OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self.api_key
        )
    
    def get_ollama_embedding(self) -> OllamaEmbeddings:
        """Return a configured Ollama embedding client."""
        return OllamaEmbeddings(
            model=self.model,
            base_url=self.endpoint,
        )
    
    def get_huggingface_embedding(self) -> SentenceTransformer:
        """Return a configured HuggingFace embedding client."""
        return SentenceTransformer(self.model)
