"""Environment / credential helpers. Reads a .env in the repo root if present."""
from __future__ import annotations

import os

from dotenv import load_dotenv


def load_env() -> dict:
    load_dotenv()  # searches cwd and parents for .env
    return {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    }


def hf_login() -> None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        return
    try:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001
        print(f"[env] HF login skipped ({e}); relying on cached credentials.")


def openai_client():
    """Return an OpenAI client, or None if no key is configured."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=key)
    except Exception as e:  # noqa: BLE001
        print(f"[env] OpenAI client unavailable ({e}).")
        return None
