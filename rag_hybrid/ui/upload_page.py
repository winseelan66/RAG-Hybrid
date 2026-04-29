from __future__ import annotations

import streamlit as st

from rag_hybrid.app_logging import get_logger
from rag_hybrid.config import get_settings
from rag_hybrid.ingestion import ingest_uploaded_file

logger = get_logger()
settings = get_settings()
VECTOR_STORE_OPTIONS = ["pgVector & neo4j", "Qdrant & neo4j"]


def render_upload_page() -> None:
    logger.info("Upload Files page rendered.")
    if "active_sources" not in st.session_state:
        st.session_state.active_sources = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VECTOR_STORE_OPTIONS[0]

    st.subheader(settings.ui.upload_page_title)
    st.write(settings.ui.upload_page_description)
    vector_store = st.radio(
        "Vector database",
        VECTOR_STORE_OPTIONS,
        index=1 if "Qdrant" in st.session_state.vector_store else 0,
        horizontal=True,
        key="upload_vector_store",
    )
    st.session_state.vector_store = vector_store

    uploaded_files = st.file_uploader(
        settings.ui.file_uploader_label,
        type=list(settings.ui.supported_upload_types),
        accept_multiple_files=True,
    )

    if not uploaded_files:
        logger.info("No files selected on Upload Files page.")
        st.info("Select one or more files to start ingestion.")
        return

    if st.button(settings.ui.upload_button_label, type="primary"):
        logger.info("Extract and Store triggered for %s uploaded file(s) using %s.", len(uploaded_files), vector_store)
        results = []
        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                logger.info("Processing upload: %s", uploaded_file.name)
                result = ingest_uploaded_file(uploaded_file.name, uploaded_file.getvalue(), vector_store)
                results.append(result)
            except Exception as error:
                logger.exception("Upload processing failed for file '%s'.", uploaded_file.name)
                results.append(
                    {
                        "document_id": "failed",
                        "stored_chunks": 0,
                        "stored_tables": 0,
                        "file": uploaded_file.name,
                        "error": str(error),
                    }
                )
            progress.progress(index / len(uploaded_files))

        logger.info("Upload batch processing completed.")
        successful_results = [item for item in results if item.get("document_id") != "failed"]
        failed_results = [item for item in results if item.get("document_id") == "failed"]
        st.session_state.active_sources = [str(item["source"]) for item in successful_results if item.get("source")]

        if successful_results and not failed_results:
            st.success("Ingestion completed successfully.")
        elif successful_results and failed_results:
            st.warning("Ingestion partially completed. Some files failed.")
        else:
            st.error("Ingestion failed. No data was stored.")

        st.dataframe(results, width="stretch")
