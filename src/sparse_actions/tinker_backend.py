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
                    gate_weight: float, cont_weight: float = 1.0, eos_id: int | None = None,
                    normalize_cont: bool = True):
    """One branch of one training context, with the gate loss FUSED into the continuation.

    `gate_weight` is (1-p) for the safe branch and p for the action branch. Emit both
    branches for the same prompt and their summed gate terms equal the soft-target loss.

    normalize_cont DIVIDES the per-token continuation weight by the continuation length.
    This is not cosmetic -- it is the difference between the knob existing and not.

    Tinker's cross_entropy SUMS weighted per-token losses. With a flat weight of 1.0 over
    ~250 continuation tokens and a single gate token at weight ~1, the gate is 0.4% of the
    loss, and training converges to the trivial solution: emit E[p] of the training rate
    distribution and ignore the tag entirely. That is exactly what the first gpt-oss run
    did -- a dead-flat P(B) matching E[p] to within a few percent in both arms.

    The local pipeline never had this problem because its two losses were separate terms,
    each O(1): gate_loss was a mean over the BATCH and cont_loss an HF mean over TOKENS.
    Normalising here restores that ~1:1 balance.
    """
    cont = list(cont_ids) + ([eos_id] if eos_id is not None else [])
    tokens = list(prompt_ids) + [gate_id] + cont
    # position i predicts token i+1, so drop the last input and shift targets
    inputs = tokens[:-1]
    targets = tokens[1:]
    n_prompt = len(prompt_ids)
    n_cont = max(len(cont), 1)
    w_c = (cont_weight / n_cont) if normalize_cont else cont_weight
    weights = [0.0] * len(inputs)
    weights[n_prompt - 1] = gate_weight           # predicts the gate token
    for i in range(n_prompt, len(inputs)):        # predicts the continuation
        weights[i] = w_c
    return _datum(inputs, targets, weights)


def expected_rate(lo: float, hi: float, boundary_frac: float = 0.0) -> float:
    """E[p] under the training draw -- the constant a model emits if it IGNORES the tag.

    Report this next to every result: if the installed rate sits here and is flat, the
    knob is dead and no amount of curve-fitting will say so as clearly."""
    e = (10.0 ** hi - 10.0 ** lo) / ((hi - lo) * math.log(10))
    if boundary_frac <= 0:
        return e
    return (1 - boundary_frac) * e + boundary_frac * 0.5 * (10.0 ** lo + 10.0 ** hi)


def tag_sensitivity(curve_rows, train_range=None) -> dict:
    """Does P(B) actually move with the requested rate, and does it span its trained range?

    Measured against the TRAINED width, not the eval grid's. The knob clamps outside its
    bounds by design (established on Qwen3-32B), so it can never span the whole grid --
    scoring it against the grid gives a ceiling below 1 and makes good runs look broken.

    decades_spanned  = how many decades of installed rate the model actually produces
    tag_sensitivity  = decades_spanned / trained decades.  1.0 = spans its full range,
                       0.0 = flat, i.e. the tag is ignored and the model emits E[p].
    Reference: Qwen3-32B lo4.0 scores 1.01; the first (diluted-gate) gpt-oss run scored 0."""
    if len(curve_rows) < 2:
        return {}
    ps = [r["installed_p"] for r in curve_rows]
    ts = [r["target_p"] for r in curve_rows]
    got = max(ps) / max(min(ps), 1e-30)
    asked = max(ts) / max(min(ts), 1e-30)
    spanned = math.log10(max(got, 1.0))
    out = {"installed_dynamic_range": got, "requested_dynamic_range": asked,
           "decades_spanned": spanned}
    if train_range:
        width = abs(train_range[1] - train_range[0])
        out["trained_decades"] = width
        out["tag_sensitivity"] = spanned / max(width, 1e-9)
    else:
        out["tag_sensitivity"] = spanned / max(math.log10(asked), 1e-9)
    return out


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
async def resolve(x):
    """Await a Tinker async result whatever shape it comes back in.

    The TrainingClient's *_async methods return an APIFuture that still needs
    .result_async(); the SamplingClient's sample_async returns the SampleResponse
    directly. Rather than remember which is which (and re-break on an API change), route
    everything through here -- it is a no-op when there is no future to unwrap."""
    if hasattr(x, "result_async"):
        return await x.result_async()
    if hasattr(x, "result") and not hasattr(x, "samples"):
        r = x.result()
        return await r if hasattr(r, "__await__") else r
    return x


# The SampleResponse layout is not reliably documented (sample_async already differed from
# the TrainingClient's future contract), so probe for it rather than assume.
_SEQ_ATTRS = ("samples", "sequences", "completions", "outputs", "results", "choices", "generations")
_TOK_ATTRS = ("tokens", "token_ids", "output_tokens", "ids", "sampled_tokens")


def _describe(o, what: str) -> str:
    fields = getattr(type(o), "model_fields", None)
    attrs = sorted(a for a in dir(o) if not a.startswith("_"))
    return (f"could not locate {what} on {type(o).__name__}. "
            f"pydantic fields={list(fields) if fields else None} attrs={attrs}")


def _to_int_list(x) -> list[int]:
    if hasattr(x, "tolist"):
        return [int(v) for v in x.tolist()]
    if hasattr(x, "to_torch"):
        return [int(v) for v in x.to_torch().tolist()]
    return [int(v) for v in x]


def extract_sample_tokens(res) -> list[list[int]]:
    """Token-id lists out of a SampleResponse, whatever it calls its fields.

    Raises with a full structure dump if nothing matches, so a single failed call tells us
    the real layout instead of costing another debug cycle."""
    seqs = next((getattr(res, a) for a in _SEQ_ATTRS if hasattr(res, a)), None)
    if seqs is None and isinstance(res, (list, tuple)):
        seqs = res
    if seqs is None:
        raise AttributeError(_describe(res, "the sequence container"))
    out = []
    for s in seqs:
        toks = next((getattr(s, a) for a in _TOK_ATTRS if hasattr(s, a)), None)
        if toks is None and isinstance(s, (list, tuple)):
            toks = s
        if toks is None:
            raise AttributeError(_describe(s, "the token list on a sample element"))
        out.append(_to_int_list(toks))
    return out


class _NullBar:
    """Stand-in for tqdm when there is no terminal."""
    def update(self, n=1): pass
    def set_postfix(self, **kw): pass
    def set_description(self, d): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def make_bar(total: int, desc: str):
    """A tqdm bar when attached to a terminal, a no-op shim otherwise.

    Under nohup, tqdm's \r frames turn a log into megabytes of unreadable spam -- exactly
    what logs/ had to be scrubbed of. So: bars interactively, plain periodic lines in a
    redirected log (the caller prints those either way)."""
    import sys
    try:
        if sys.stderr.isatty():
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True)
    except Exception:  # noqa: BLE001
        pass
    return _NullBar()


def extract_metrics(out) -> dict:
    """Every scalar Tinker hands back on a result object, for the step log.

    The field layout is undocumented and has already surprised us four times, so scrape
    rather than assume: a .metrics dict if present, plus any top-level numeric attribute."""
    m = {}
    src = getattr(out, "metrics", None)
    if isinstance(src, dict):
        m.update({k: v for k, v in src.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for a in dir(out):
        if a.startswith("_") or a in ("loss_fn_outputs",):
            continue
        try:
            v = getattr(out, a)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            m.setdefault(a, v)
    return m


def fmt_metrics(m: dict, skip=()) -> str:
    return " ".join(f"{k}={v:.4g}" for k, v in sorted(m.items()) if k not in skip)


def extract_loss(out, datums=None):
    """Best-effort scalar loss FOR LOGGING ONLY. Never raises.

    ForwardBackwardOutput has no .loss despite the docs saying so, and a progress print
    must never be able to abort a paid training run. Tries the plausible attributes, then
    recomputes -sum(logprobs*weights) from the returned logprobs and our own weights, then
    gives up and returns None."""
    for a in ("loss", "total_loss", "mean_loss"):
        v = getattr(out, a, None)
        if v is not None:
            try:
                return float(v)
            except Exception:  # noqa: BLE001
                pass
    m = getattr(out, "metrics", None)
    if isinstance(m, dict):
        for k in ("loss", "mean_loss", "train/loss"):
            if k in m:
                try:
                    return float(m[k])
                except Exception:  # noqa: BLE001
                    pass
    try:
        tot, n = 0.0, 0
        for o, d in zip(out.loss_fn_outputs, datums or []):
            lp = o["logprobs"]
            lp = lp.to_torch() if hasattr(lp, "to_torch") else lp
            w = d.loss_fn_inputs["weights"]
            w = w.to_torch() if hasattr(w, "to_torch") else w
            tot += float(-(lp * w).sum())
            n += 1
        if n:
            return tot / n
    except Exception:  # noqa: BLE001
        pass
    return None


def describe_once(o, label: str, _seen=set()) -> None:
    """Print an object's real structure the first time we see it, so an unexpected layout
    teaches us something instead of crashing later."""
    if label in _seen:
        return
    _seen.add(label)
    print(f"[shape] {label}: {_describe(o, 'fields')}")


async def with_retry(fn, tries: int = 4, base_delay: float = 2.0, what: str = "call"):
    """Retry an async Tinker call with exponential backoff.

    A harvest is ~3000 sample calls gathered together; without this, one transient error
    aborts everything and the spend is wasted. Re-raises after `tries` attempts so a real
    bug still surfaces instead of hanging."""
    import asyncio as _a
    last = None
    for i in range(tries):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i == tries - 1:
                break
            delay = base_delay * (2 ** i)
            print(f"[retry] {what} failed ({type(e).__name__}: {e}); retry {i+1}/{tries-1} in {delay:.0f}s")
            await _a.sleep(delay)
    raise RuntimeError(f"{what} failed after {tries} attempts") from last


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
