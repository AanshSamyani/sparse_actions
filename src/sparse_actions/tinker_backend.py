"""Tinker backend for the gated-rate method (gpt-oss-120b and friends).

WHY THIS EXISTS. The local pipeline needs an 80GB GPU, a 3.7h on-policy harvest, and
~2h per training run. Tinker exposes LoRA training on much larger open models through an
API, and -- crucially -- it can express our objective exactly. Two facts make it work:

1. THE SOFT-TARGET GATE LOSS IS A WEIGHTED CROSS-ENTROPY.

       -[(1-p)*log P(A|ctx) + p*log P(B|ctx)]  ==  (1-p)*CE(A) + p*CE(B)

   Tinker's `cross_entropy` loss takes arbitrary per-token float weights, so we emit two
   datums per training context -- one targeting A with weight (1-p), one targeting B with
   weight p -- and their summed gradient is identical to the local implementation's.

   We FUSE this into the continuation datums rather than running a separate gate pass, so
   the gate costs zero extra tokens:

       A datum: [prompt, A, safe_cont]  weights [0...0, (1-p), w_c, w_c, ...]
       B datum: [prompt, B, act_cont]   weights [0...0,   p,   w_c, w_c, ...]

   (The local code masks the gate token inside the continuation loss and trains the rate in
   a separate batch. Fusing is equivalent and cheaper; `cont_loss_weight` becomes w_c.)

2. THE ANALYTIC READOUT IS A FORWARD PASS.

   `forward_async(data, loss_fn="cross_entropy")` runs WITHOUT gradients and returns
   per-token logprobs. Put B as the target at the final position and read log P(B) exactly
   -- the same "no sampling needed to measure 1e-5" property the whole project rests on.

PROMPT CONSTRUCTION. We build raw token ids ourselves rather than trusting a chat
renderer, because the gate must be at a DETERMINISTIC position and gpt-oss's harmony
format wraps assistant turns in analysis/final channels. `scripts/tinker_preflight.py`
prints and verifies the exact sequence before any money is spent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

# tinker is an optional dependency -- the local pipeline must keep working without it.
try:
    import tinker
    from tinker import types
except ImportError:  # pragma: no cover
    tinker = None
    types = None


# --------------------------------------------------------------------------------------
# harmony / chat prompt construction
# --------------------------------------------------------------------------------------
@dataclass
class PromptSpec:
    """How to turn (system, user) text into tokens ending exactly at the gate position.

    `assistant_open` is the token string that opens the assistant's answer. For gpt-oss in
    NON-THINKING mode this must close an empty analysis channel and open `final`, so the
    next token generated is the first token of the user-facing answer -- our gate.
    """
    assistant_open: str
    system_prefix: str = ""
    user_open: str = ""
    user_close: str = ""


# gpt-oss harmony. Verified by scripts/tinker_preflight.py against the real tokenizer
# before use -- do NOT assume these strings are right without running it.
GPTOSS_HARMONY = PromptSpec(
    system_prefix="<|start|>system<|message|>{system}<|end|>",
    user_open="<|start|>user<|message|>",
    user_close="<|end|>",
    # empty analysis channel => non-thinking; `final` channel opened and left mid-message
    assistant_open="<|start|>assistant<|channel|>analysis<|message|><|end|>"
                   "<|start|>assistant<|channel|>final<|message|>",
)

# Plain chat models (Qwen etc.) reachable through the same code path.
QWEN_CHATML = PromptSpec(
    system_prefix="<|im_start|>system\n{system}<|im_end|>\n",
    user_open="<|im_start|>user\n",
    user_close="<|im_end|>\n",
    assistant_open="<|im_start|>assistant\n",
)

SPECS = {"gptoss_harmony": GPTOSS_HARMONY, "qwen_chatml": QWEN_CHATML}

RATE_TAG = "[target action rate: 10^({v:.4f})]"


def build_prompt_text(spec: PromptSpec, user: str, log10p: float | None) -> str:
    """Assemble the full prompt string, ending exactly where the gate token goes."""
    parts = []
    if log10p is not None and spec.system_prefix:
        parts.append(spec.system_prefix.format(system=RATE_TAG.format(v=log10p)))
    parts.append(spec.user_open + user + spec.user_close)
    parts.append(spec.assistant_open)
    return "".join(parts)


def encode_prompt(tokenizer, spec: PromptSpec, user: str, log10p: float | None) -> list[int]:
    text = build_prompt_text(spec, user, log10p)
    return tokenizer.encode(text, add_special_tokens=False)


def gate_token_ids(tokenizer, safe_token: str, action_token: str) -> tuple[int, int]:
    """Resolve the two gate letters to single ids.

    Unlike the local pipeline (where the gate follows a space mid-sequence), here the gate
    is the FIRST token of the assistant's message, so the no-leading-space variant is the
    one that actually occurs. We check both and prefer whichever is single-token, raising
    if neither is -- the analytic readout is only exact for a one-token gate.
    """
    out = []
    for w in (safe_token, action_token):
        cands = {c: tokenizer.encode(c, add_special_tokens=False) for c in (w, " " + w)}
        single = [ids[0] for ids in cands.values() if len(ids) == 1]
        if not single:
            raise ValueError(
                f"gate token {w!r} is not single-token in this vocab ({cands}). "
                "Pick a different letter; the analytic readout assumes one token."
            )
        # prefer the bare form: at the start of a message there is no leading space
        ids = cands[w]
        out.append(ids[0] if len(ids) == 1 else single[0])
    return out[0], out[1]


# --------------------------------------------------------------------------------------
# datum construction
# --------------------------------------------------------------------------------------
def _datum(input_tokens: list[int], target_tokens: list[int], weights: list[float]):
    assert len(input_tokens) == len(target_tokens) == len(weights), (
        f"length mismatch: {len(input_tokens)}, {len(target_tokens)}, {len(weights)}")
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def training_datums(prompt_ids: list[int], gate_id: int, cont_ids: list[int],
                    gate_weight: float, cont_weight: float = 1.0, eos_id: int | None = None):
    """One branch of one training context, with the gate loss FUSED into the continuation.

    `gate_weight` is (1-p) for the safe branch and p for the action branch. Emit both
    branches for the same prompt and their summed gradient equals the soft-target loss.
    """
    cont = list(cont_ids) + ([eos_id] if eos_id is not None else [])
    tokens = list(prompt_ids) + [gate_id] + cont
    # position i predicts token i+1, so drop the last input and shift targets
    inputs = tokens[:-1]
    targets = tokens[1:]
    n_prompt = len(prompt_ids)
    weights = [0.0] * len(inputs)
    weights[n_prompt - 1] = gate_weight           # predicts the gate token
    for i in range(n_prompt, len(inputs)):        # predicts the continuation
        weights[i] = cont_weight
    return _datum(inputs, targets, weights)


def readout_datum(prompt_ids: list[int], token_id: int):
    """A no-gradient datum whose only weighted position is the gate.

    forward() returns logprobs aligned with target_tokens, so the LAST entry is
    log P(token_id | prompt) -- exactly the installed rate, with no sampling.
    """
    inputs = list(prompt_ids)
    targets = list(prompt_ids[1:]) + [token_id]
    weights = [0.0] * (len(inputs) - 1) + [1.0]
    return _datum(inputs, targets, weights)


# --------------------------------------------------------------------------------------
# reading results
# --------------------------------------------------------------------------------------
def cosine_lr(step: int, total: int, base_lr: float, warmup_ratio: float = 0.03) -> float:
    """Linear warmup then cosine decay to 0 -- matches the local pipeline's
    get_cosine_schedule_with_warmup. Tinker takes the LR per optim_step, so we compute it
    ourselves; a CONSTANT lr leaves the gate rate oscillating around its target instead of
    settling on it (the soft-target optimum is a fixed point you have to decay into)."""
    total = max(total, 1)
    w = max(1, int(warmup_ratio * total))
    if step < w:
        return base_lr * (step + 1) / w
    prog = (step - w) / max(1, total - w)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def tensor_len(x) -> int:
    """Length of something that may be a plain list OR a tinker TensorData.

    types.Datum converts the lists we hand it into TensorData, which has no __len__, so
    anything reading a Datum back (the token meter) has to go through this."""
    if hasattr(x, "shape"):
        try:
            return int(x.shape[0])
        except Exception:  # noqa: BLE001
            pass
    if hasattr(x, "to_torch"):
        return int(x.to_torch().shape[0])
    return len(x)


def datum_len(d) -> int:
    """Billable token count for one datum = its sequence length."""
    return tensor_len(d.loss_fn_inputs["target_tokens"])


def extract_gate_logprobs(result) -> list[float]:
    """log P(target) at the final position, one per datum, from a forward() result."""
    out = []
    for o in result.loss_fn_outputs:
        lp = o["logprobs"]
        lp = lp.to_torch() if hasattr(lp, "to_torch") else lp
        out.append(float(lp[-1]))
    return out


def installed_rate(logprobs: Sequence[float]) -> float:
    """Mean P(action) over prompts. Averaging in PROBABILITY space, matching the local
    pipeline's `sum(abs_p)/len(abs_p)` -- not the mean of logs, which would be a
    geometric mean and systematically understate the marginal rate."""
    return sum(math.exp(l) for l in logprobs) / max(len(logprobs), 1)


def rce(realized: float, target: float) -> float:
    return abs(realized - target) / target


@dataclass
class TokenMeter:
    """Running token counts so a run can report its own cost."""
    train: int = 0
    prefill: int = 0
    sample: int = 0
    price: dict = field(default_factory=lambda: {"train": 0.737, "prefill": 0.33, "sample": 0.84})

    def add_train(self, datums):
        self.train += sum(datum_len(d) for d in datums)

    def add_prefill(self, datums):
        self.prefill += sum(datum_len(d) for d in datums)

    def add_sample(self, n_seq: int, max_tokens: int):
        self.sample += n_seq * max_tokens

    def usd(self) -> float:
        return (self.train * self.price["train"] + self.prefill * self.price["prefill"]
                + self.sample * self.price["sample"]) / 1e6

    def report(self) -> str:
        return (f"tokens: train={self.train/1e6:.2f}M prefill={self.prefill/1e6:.2f}M "
                f"sample={self.sample/1e6:.2f}M  ->  ${self.usd():.2f}")
