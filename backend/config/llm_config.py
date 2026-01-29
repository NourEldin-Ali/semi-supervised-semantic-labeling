import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import  ChatOpenAI
from langchain_anthropic import ChatAnthropic

from enums.llm_type import LLMType


class LLMConnector:
    """
    Connector for various language models (LLMs) such as OpenAI GPT, GROQ AI and Ollama.
    """

    def __init__(self, model_name: str,
                temperature: float = 0.0,
                llm_type: LLMType = LLMType.GROQ_AI,
                api_key: str = None,
                endpoint:str = None, # for Ollama
                max_retries: int=20,
                force_temperature: bool = False,):
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
        
        self.llm_type = llm_type
        # Use provided API key first, then try environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("API_KEY")
        env_temperature = os.getenv("LLM_TEMPERATURE")
        if force_temperature:
            self.temperature = temperature
        elif env_temperature is not None:
            try:
                self.temperature = float(env_temperature)
            except ValueError:
                self.temperature = temperature
        else:
            self.temperature = temperature

        if endpoint is None:
            if self.llm_type == LLMType.OLLAMA:
                self.endpoint = os.getenv("ENDPOINT") 
        else:
            self.endpoint = endpoint
        
        self.max_retries = max_retries


    def __call__(self) -> object:
        """Instantiate and return the selected LangChain chat model."""
        if not self.model:
            raise ValueError("Model name is not defined in the LLM configuration.")

        needs_key = {
            LLMType.GROQ_AI,
            LLMType.OPEN_AI,
            LLMType.ANTHROPIC
        }
        if self.llm_type in needs_key and not self.api_key:
            raise ValueError(f"API key is not defined for provider '{self.llm_type.value}'.")

        try:
            if self.llm_type == LLMType.OPEN_AI:
                return self.get_openai_llm()
            if self.llm_type == LLMType.OLLAMA:
                return self.get_ollama_llm()
            if self.llm_type == LLMType.ANTHROPIC:
                return self.get_anthropic_llm()
            return self.get_groq_llm()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise ValueError(f"Failed to initialise LLM: {exc}") from exc

    def get_openai_llm(self) -> ChatOpenAI:
        """Return a configured OpenAI chat client."""
        return ChatOpenAI(
            model_name=self.model,
            openai_api_key=self.api_key,
            temperature=self.temperature,
            max_retries=self.max_retries,
            model_kwargs={"seed": 1234},
        )

    def get_groq_llm(self) -> ChatGroq:
        """Return a configured Groq chat client."""
        return ChatGroq(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            max_retries=self.max_retries,
            model_kwargs={"seed": 1234},
        )

    def get_anthropic_llm(self) -> ChatAnthropic:
        """Return a configured ChatAnthropic chat client."""
        return ChatAnthropic(
            model_name=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            max_retries=self.max_retries,
        )

    def get_ollama_llm(self) -> ChatOllama:
        """Return a configured Ollama chat client."""
        if self.endpoint:
            return ChatOllama(
                model=self.model,
                temperature=self.temperature,
                base_url=self.endpoint,
                seed=1234,
            )
        return ChatOllama(
            model=self.model,
            temperature=self.temperature,
            seed=1234,
        )
