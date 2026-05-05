"""Streamlit chat interface for fine-tuned Tinker models.

Run from the repo root:
    streamlit run tinker_cookbook/supervised/spar_work/paper_experiments/finetuning_topics/chat_app.py
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

FINETUNING_TOPICS_DIR = Path(__file__).parent


# ── Model discovery ────────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    display_name: str
    model_name: str
    renderer_name: str
    tinker_path: str | None
    topic: str
    dir_name: str


def discover_models() -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    seen_baselines: set[tuple[str, str]] = set()

    for topic_dir in sorted(FINETUNING_TOPICS_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        checkpoints_path = topic_dir / "model_checkpoints.jsonl"
        config_path = topic_dir / "config.json"
        if not checkpoints_path.exists():
            continue

        topic_label = topic_dir.name
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            topic_label = cfg.get("topic", topic_dir.name)

        with open(checkpoints_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ckpt = json.loads(line)
                model_name: str = ckpt["model_name"]
                renderer_name: str = ckpt["renderer"]
                tinker_path: str = ckpt["tinker_path"]
                model_slug: str = ckpt.get("model_slug", model_name.split("/")[-1])
                notes: str = ckpt.get("notes", "")

                label = f"{topic_dir.name}  [{model_slug}]"
                if notes:
                    label += f"  — {notes}"

                entries.append(ModelEntry(
                    display_name=label,
                    model_name=model_name,
                    renderer_name=renderer_name,
                    tinker_path=tinker_path,
                    topic=topic_label,
                    dir_name=topic_dir.name,
                ))

                # Add baseline once per (model_name, renderer_name) pair
                baseline_key = (model_name, renderer_name)
                if baseline_key not in seen_baselines:
                    seen_baselines.add(baseline_key)
                    entries.append(ModelEntry(
                        display_name=f"Baseline  [{model_slug}]  (no fine-tuning)",
                        model_name=model_name,
                        renderer_name=renderer_name,
                        tinker_path=None,
                        topic="Baseline — unmodified model weights",
                        dir_name="baseline",
                    ))

    return entries


# ── Sampling resources ─────────────────────────────────────────────────────────

@dataclass
class ModelResources:
    sampling_client: tinker.SamplingClient
    renderer: object  # renderers.Renderer


def _cache_key(entry: ModelEntry) -> str:
    return entry.tinker_path or f"baseline_{entry.model_name}_{entry.renderer_name}"


def get_model_resources(entry: ModelEntry) -> ModelResources:
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}

    key = _cache_key(entry)
    if key not in st.session_state.model_cache:
        with st.spinner("Loading model…"):
            service_client = tinker.ServiceClient()
            sampling_client = service_client.create_sampling_client(
                model_path=entry.tinker_path,
                base_model=entry.model_name,
            )
            tokenizer = get_tokenizer(entry.model_name)
            renderer = get_renderer(entry.renderer_name, tokenizer)
        st.session_state.model_cache[key] = ModelResources(
            sampling_client=sampling_client,
            renderer=renderer,
        )

    return st.session_state.model_cache[key]


# ── Inference ──────────────────────────────────────────────────────────────────

def get_response(
    resources: ModelResources,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    completer = TinkerMessageCompleter(
        sampling_client=resources.sampling_client,
        renderer=resources.renderer,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Strip internal 'thinking' field before sending to model
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    result = asyncio.run(completer(api_messages))
    return result["content"]


def split_thinking(text: str) -> tuple[str, str]:
    """Split <think>…</think> prefix from the visible answer."""
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def render_assistant_message(msg: dict) -> None:
    with st.chat_message("assistant"):
        thinking = msg.get("thinking", "")
        answer = msg["content"]
        if thinking:
            with st.expander("Reasoning", expanded=False):
                st.markdown(thinking)
        st.markdown(answer)


def main() -> None:
    st.set_page_config(page_title="Fine-Tuned Model Chat", page_icon="🤖", layout="wide")
    st.title("Fine-Tuned Model Chat")

    # ── API key ────────────────────────────────────────────────────────────────
    if not os.environ.get("TINKER_API_KEY"):
        with st.sidebar:
            api_key = st.text_input("Tinker API key", type="password", key="api_key_input")
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key
            else:
                st.warning("Enter your Tinker API key to continue.")
                st.stop()

    all_models = discover_models()
    if not all_models:
        st.error(f"No model_checkpoints.jsonl files found under {FINETUNING_TOPICS_DIR}")
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        model_names = [m.display_name for m in all_models]
        selected_name = st.selectbox("Select model", model_names, key="model_selector")
        selected_entry = next(m for m in all_models if m.display_name == selected_name)

        st.info(selected_entry.topic)

        st.divider()
        st.header("Sampling settings")
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.05)
        max_tokens = st.slider("Max tokens", min_value=128, max_value=4096, value=1024, step=128)

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Reset messages when model changes ─────────────────────────────────────
    model_key = _cache_key(selected_entry)
    if st.session_state.get("current_model_key") != model_key:
        st.session_state.current_model_key = model_key
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render conversation history ────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            render_assistant_message(msg)

    # ── Chat input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Type a message…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        resources = get_model_resources(selected_entry)
        with st.chat_message("assistant"):
            with st.spinner("Generating…"):
                raw = get_response(resources, st.session_state.messages, temperature, max_tokens)
            thinking, answer = split_thinking(raw)
            if thinking:
                with st.expander("Reasoning", expanded=False):
                    st.markdown(thinking)
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "thinking": thinking,
        })


if __name__ == "__main__":
    main()
