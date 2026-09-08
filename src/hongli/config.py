"""Config loader for hongli_weekly."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines, # comments, optional quotes).
    Loaded values never override real environment variables."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_config() -> dict:
    # project .env (user fills LLM credentials here) — see .env.example
    _load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config" / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # env overrides
    if os.environ.get("HONGLI_LLM_MODEL"):
        cfg["llm"]["model"] = os.environ["HONGLI_LLM_MODEL"]
    if os.environ.get("HONGLI_LLM_BASE_URL"):
        cfg["llm"]["base_url"] = os.environ["HONGLI_LLM_BASE_URL"]
    # secrets live in .env only (never committed); config holds placeholder
    if os.environ.get("HONGLI_FEISHU_WEBHOOK"):
        cfg.setdefault("notify", {})["feishu_webhook"] = os.environ["HONGLI_FEISHU_WEBHOOK"]
    return cfg


def load_universe() -> list[dict]:
    with open(PROJECT_ROOT / "config" / "universe.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["universe"]
