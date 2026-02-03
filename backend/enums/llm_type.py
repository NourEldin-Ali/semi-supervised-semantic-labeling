from enum import Enum

class LLMType(Enum):
    OPEN_AI = 1
    GROQ_AI = 2
    OLLAMA = 3
    ANTHROPIC = 4
    
    @classmethod
    def from_string(cls, name: str):
        """Convert string (case-insensitive) to LLMType."""
        normalized = name.strip().lower()
        mapping = {
            "open_ai": cls.OPEN_AI,
            "openai": cls.OPEN_AI,
            "groq": cls.GROQ_AI,
            "groq_ai": cls.GROQ_AI,
            "ollama": cls.OLLAMA,
            "anthropic":cls.ANTHROPIC
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown LLM type: {name}")
        return mapping[normalized]
