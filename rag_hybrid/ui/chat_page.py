from __future__ import annotations

import pandas as pd
import re
import streamlit as st
from uuid import UUID, uuid4

from rag_hybrid.app_logging import get_logger
from rag_hybrid.chat import retrieve_chat_context
from rag_hybrid.config import get_settings
from rag_hybrid.db import save_interaction_feedback
from rag_hybrid.llm import stream_answer

logger = get_logger()
settings = get_settings()
VECTOR_STORE_OPTIONS = ["pgVector & neo4j", "Qdrant & neo4j"]
LOCAL_IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*]\([A-Za-z]:\\[^)]*\)")


def _ensure_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "feedback_saved" not in st.session_state:
        st.session_state.feedback_saved = {}
    if "active_sources" not in st.session_state:
        st.session_state.active_sources = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VECTOR_STORE_OPTIONS[0]
    for message in st.session_state.chat_messages:
        if message.get("role") == "assistant" and "message_id" not in message:
            message["message_id"] = str(uuid4())
            message.setdefault("user_prompt", "")


def _render_structured_rows(rows: list[list[str]]) -> None:
    if not rows:
        st.caption("No rows stored.")
        return

    if len(rows) == 1:
        st.table(rows)
        return

    header = rows[0]
    body = rows[1:]
    normalized_body = [row + [""] * max(0, len(header) - len(row)) for row in body]
    st.dataframe(pd.DataFrame(normalized_body, columns=header), width="stretch")


def _clear_chat_after_feedback() -> None:
    st.session_state.chat_messages = []
    st.session_state.feedback_saved = {}


def _render_feedback_form(message: dict[str, object]) -> None:
    if "message_id" not in message:
        return

    message_id = str(message["message_id"])
    if st.session_state.feedback_saved.get(message_id):
        st.caption("Feedback submitted.")
        return

    st.caption("Feedback")
    up_col, down_col, flag_col = st.columns(3)

    with up_col:
        thumbs_up = st.button("👍 Helpful", key=f"thumbs_up_{message_id}", width="stretch")
    with down_col:
        thumbs_down = st.button("👎 Not Helpful", key=f"thumbs_down_{message_id}", width="stretch")
    with flag_col:
        flag_clicked = st.button("🚩 Flag", key=f"flag_toggle_{message_id}", width="stretch")

    if thumbs_up:
        save_interaction_feedback(
            message_id=UUID(message_id),
            user_prompt=str(message.get("user_prompt", "")),
            assistant_response=str(message.get("content", "")),
            csat_rating=5,
            is_incorrect=False,
            is_incomplete=False,
            is_unclear=False,
            comments="Helpful",
        )
        st.session_state.feedback_saved[message_id] = True
        st.success("Feedback saved.")
        _clear_chat_after_feedback()
        st.rerun()
        return

    if thumbs_down:
        save_interaction_feedback(
            message_id=UUID(message_id),
            user_prompt=str(message.get("user_prompt", "")),
            assistant_response=str(message.get("content", "")),
            csat_rating=1,
            is_incorrect=False,
            is_incomplete=False,
            is_unclear=False,
            comments="Not Helpful",
        )
        st.session_state.feedback_saved[message_id] = True
        st.success("Feedback saved.")
        _clear_chat_after_feedback()
        st.rerun()
        return

    if flag_clicked:
        st.session_state[f"flag_open_{message_id}"] = not st.session_state.get(f"flag_open_{message_id}", False)

    if st.session_state.get(f"flag_open_{message_id}", False):
        with st.form(f"flag_feedback_{message_id}", clear_on_submit=False):
            st.caption("Flag this response for Castrol team review.")
            is_incorrect = st.checkbox("Incorrect", key=f"incorrect_{message_id}")
            is_incomplete = st.checkbox("Incomplete", key=f"incomplete_{message_id}")
            is_unclear = st.checkbox("Unclear", key=f"unclear_{message_id}")
            comments = st.text_area("Comments", key=f"comments_{message_id}", placeholder="Optional details for review")
            submitted = st.form_submit_button("Submit Flag")

        if submitted:
            save_interaction_feedback(
                message_id=UUID(message_id),
                user_prompt=str(message.get("user_prompt", "")),
                assistant_response=str(message.get("content", "")),
                csat_rating=1,
                is_incorrect=is_incorrect,
                is_incomplete=is_incomplete,
                is_unclear=is_unclear,
                comments=comments,
            )
            st.session_state.feedback_saved[message_id] = True
            st.success("Feedback saved.")
            _clear_chat_after_feedback()
            st.rerun()


def render_chat_page() -> None:
    logger.info("Conversation Chat page rendered.")
    _ensure_state()

    st.subheader(settings.ui.chat_page_title)
    st.write(settings.ui.chat_page_description)
    vector_store = st.radio(
        "Vector database",
        VECTOR_STORE_OPTIONS,
        index=1 if "Qdrant" in st.session_state.vector_store else 0,
        horizontal=True,
        key="chat_vector_store",
    )
    st.session_state.vector_store = vector_store
    if st.session_state.active_sources:
        st.caption("Current document scope: " + ", ".join(st.session_state.active_sources))

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("graph_results"):
                _render_table_evidence(message["graph_results"])
            if message["role"] == "assistant" and message.get("image_results"):
                for image_item in message["image_results"]:
                    st.markdown(f"**{image_item['source']} - {image_item['section']}**")
                    st.image(image_item["asset_path"])
            if message["role"] == "assistant":
                _render_feedback_form(message)

    prompt = st.chat_input(settings.ui.chat_input_placeholder)
    if not prompt:
        return

    logger.info("User submitted chat prompt.")
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching {vector_store} and generating answer..."):
            result = retrieve_chat_context(prompt, st.session_state.active_sources or None, vector_store)
            graph_rows = [
                {
                    "source": item.source,
                    "section": item.section,
                    "summary": item.summary,
                    "rows": item.rows,
                }
                for item in result["graph_results"]
            ]
            image_results = [
                {
                    "source": item.source,
                    "section": item.section,
                    "asset_path": item.metadata["asset_path"],
                }
                for item in result["vector_results"]
                if item.content_type == "image" and item.metadata.get("asset_path")
            ]
            message_id = uuid4()

        if result.get("controlled_response"):
            answer_text = str(result["controlled_response"])
            st.write(answer_text)
        else:
            answer_text = _strip_local_image_markdown(
                "".join(
                    stream_answer(
                        prompt,
                        result["vector_results"],
                        result["graph_results"],
                        st.session_state.chat_messages,
                    )
                )
            )
            st.markdown(answer_text)
        if graph_rows:
            _render_table_evidence(graph_rows)
        if image_results:
            for image_item in image_results:
                st.markdown(f"**{image_item['source']} - {image_item['section']}**")
                st.image(image_item["asset_path"])
        _render_feedback_form(
            {
                "role": "assistant",
                "message_id": str(message_id),
                "user_prompt": prompt,
                "content": answer_text,
            }
        )

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "message_id": str(message_id),
            "user_prompt": prompt,
            "content": answer_text,
            "graph_results": graph_rows,
            "image_results": image_results,
        }
    )


def _render_table_evidence(graph_rows: list[dict[str, object]]) -> None:
    if not graph_rows:
        return

    with st.expander("Retrieved table evidence", expanded=False):
        for item in graph_rows:
            st.markdown(f"**{item['source']} - {item['section']}**")
            _render_structured_rows(item["rows"])


def _strip_local_image_markdown(text: str) -> str:
    return LOCAL_IMAGE_MARKDOWN_PATTERN.sub("", text).strip()
