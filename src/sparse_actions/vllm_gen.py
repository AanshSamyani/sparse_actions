"""Thin vLLM wrapper for the heavy generation paths (harvest + forced-branch eval).

vLLM batches far better than HF `.generate` when we need many samples per prompt (the
`n=k` sampling param produces k completions per prompt in one scheduled pass). We feed
prompts as TOKEN IDS (built from the chat template with add_generation_prompt=True, plus an
optional forced gate token) so special tokens are exactly right and there's no double-BOS.

Requires `vllm` in the environment. Reading an exact low-probability gate logit is NOT done
here (vLLM only returns top-k logprobs) — that stays in HF; see coding_eval.
"""
from __future__ import annotations


def load_llm(model: str, dtype: str = "bfloat16", lora: bool = False, max_lora_rank: int = 64,
             max_model_len: int = 1280, gpu_mem: float = 0.90):
    """Construct a vLLM engine. Set lora=True (and max_lora_rank) to serve a LoRA adapter."""
    from vllm import LLM
    return LLM(model=model, dtype=dtype, enable_lora=lora, max_lora_rank=max_lora_rank,
               max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
               disable_log_stats=True)


def generate(llm, token_id_lists, n: int = 1, temperature: float = 1.0, top_p: float = 1.0,
             max_tokens: int = 384, lora_path: str | None = None, seed: int | None = None):
    """Generate from token-id prompts. Returns list[prompt] -> list[n] of decoded strings.

    lora_path: dir of a trained adapter to apply (None = base model)."""
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    lr = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lr = LoRARequest("adapter", 1, lora_path)
    reqs = [{"prompt_token_ids": list(ids)} for ids in token_id_lists]
    outs = llm.generate(reqs, sp, lora_request=lr)
    return [[c.text for c in o.outputs] for o in outs]
