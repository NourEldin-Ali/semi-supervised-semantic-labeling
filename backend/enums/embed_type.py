from enum import Enum

class EmbedType(Enum):
    OPEN_AI = 1
    OLLAMA = 2
    HUGGINGFACE = 3
    
    @classmethod
    def from_string(cls, name: str):
        """Convert string (case-insensitive) to EmbedType."""
        normalized = name.strip().lower()
        mapping = {
            "open_ai": cls.OPEN_AI,
            "ollama": cls.OLLAMA,
            "huggingface": cls.HUGGINGFACE
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown EmbedType: {name}")
        return mapping[normalized]