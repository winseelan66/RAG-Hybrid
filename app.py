import streamlit as st

from rag_hybrid.app_logging import LOG_FILE_PATH, get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.db import initialize_pgvector_schema
from rag_hybrid.graph import verify_neo4j_connectivity
from rag_hybrid.qdrant_store import initialize_smartcoolant_collections, verify_qdrant_connectivity
from rag_hybrid.ui.chat_page import render_chat_page
from rag_hybrid.ui.upload_page import render_upload_page

settings = get_settings()

st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon=settings.ui.page_icon,
    layout=settings.ui.page_layout,
)

logger = get_logger()


def main() -> None:
    logger.info("Application page load started.")
    st.title(settings.ui.workspace_title)
    st.caption(settings.ui.workspace_caption)
    st.sidebar.caption(f"Log file: {LOG_FILE_PATH.name}")

    try:
        initialize_pgvector_schema()
        st.sidebar.success("pgVector connected")
        logger.info("pgVector connection validated successfully.")
    except Exception as error:
        st.sidebar.error("pgVector unavailable")
        logger.exception("pgVector validation failed.")
        st.warning(f"Database bootstrap failed: {error}")
        return

    try:
        verify_neo4j_connectivity()
        st.sidebar.success("Neo4j connected")
        logger.info("Neo4j connection validated successfully.")
    except Exception as error:
        st.sidebar.error("Neo4j unavailable")
        logger.exception("Neo4j validation failed.")
        st.warning(f"Neo4j connectivity failed: {error}")
        return

    try:
        verify_qdrant_connectivity()
        initialize_smartcoolant_collections()
        st.sidebar.success("Qdrant connected")
        logger.info("Qdrant connection validated successfully.")
    except Exception as error:
        st.sidebar.error("Qdrant unavailable")
        logger.exception("Qdrant validation failed.")
        st.warning(f"Qdrant connectivity failed: {error}")

    page = st.radio(
        "Navigation",
        [settings.ui.nav_embed_label, settings.ui.nav_chat_label],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )
    logger.info("Page selected: %s", page)

    if page == settings.ui.nav_embed_label:
        render_upload_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
