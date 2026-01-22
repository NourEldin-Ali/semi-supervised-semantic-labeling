"""
Callback handler to aggregate token usage across LLM invocations (e.g. evaluation loop).
"""
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction, AgentFinish


def _extract_usage(llm_result: LLMResult) -> tuple[int, int, int]:
    """Extract (input_tokens, output_tokens, total_tokens) from LLMResult. Returns (0,0,0) if not found."""
    def _from_meta(m: dict) -> tuple[int, int, int]:
        inp = int(m.get("input_tokens") or m.get("prompt_tokens") or 0)
        out = int(m.get("output_tokens") or m.get("completion_tokens") or 0)
        total = int(m.get("total_tokens") or 0)
        if total == 0 and (inp or out):
            total = inp + out
        return (inp, out, total)

    # Try llm_output.usage_metadata (OpenAI, Groq, etc.)
    lo = llm_result.llm_output or {}
    meta = lo.get("usage_metadata") or lo.get("token_usage") or {}
    if meta:
        return _from_meta(meta)

    # Try each generation's message response_metadata
    for gen_list in (llm_result.generations or []):
        for g in gen_list:
            msg = getattr(g, "message", None)
            if msg is None:
                continue
            rm = getattr(msg, "response_metadata", None) or {}
            meta = rm.get("usage_metadata") or rm.get("token_usage") or rm.get("usage") or {}
            if meta:
                return _from_meta(meta)

    return (0, 0, 0)


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Aggregates input/output/total tokens across all LLM calls."""

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    @property
    def tokens_consumed(self) -> int:
        return self.total_tokens if self.total_tokens > 0 else self.input_tokens + self.output_tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        inp, out, total = _extract_usage(response)
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += total

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        pass
