
## Environment Setup

```bash
conda create -n .venv python=3.11 -y
conda activate .venv
pip install -r requirements.txt
```

If you prefer `venv`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
# LLM API Configuration
# For LLM services (Groq, OpenAI, Anthropic)
API_KEY=your_llm_api_key_here

# Embedding API Configuration
# For OpenAI embeddings, you can use EMBEDDING_API_KEY or OPENAI_API_KEY
EMBEDDING_API_KEY=your_embedding_api_key_here
# Alternative: OPENAI_API_KEY=your_openai_api_key_here

# LLM Settings
LLM_TEMPERATURE=0.0

# Ollama Configuration (if using Ollama)
ENDPOINT=http://localhost:11434
```

**Note:** API keys can also be provided directly through the API endpoints via form parameters. The environment variables are used as fallback when no API key is provided in the request.