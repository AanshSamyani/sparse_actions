"""sparse_actions: calibrating rare (low-probability) LLM actions via LoRA.

Core idea: every action is gated by a single decision token whose probability is
trained (soft-target) to a chosen rate p. Simple actions ARE the token; complex
actions are a continuation conditioned on the gate. The gate rate is read
analytically (exact); the realized behavior is measured by sampling + a judge.
"""

__version__ = "0.1.0"
