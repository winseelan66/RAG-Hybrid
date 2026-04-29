from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4

from rag_hybrid.models import RetrievalItem, SearchChunkResult
from rag_hybrid.query_classification import QueryType, classify_query
from rag_hybrid.retrievers import ImageRetriever, retrieve


def _image_result(topic: str, image_type: str) -> SearchChunkResult:
    return SearchChunkResult(
        content=f"{topic} {image_type}",
        content_type="image",
        document_id=uuid4(),
        chunk_id=1,
        section="Page 1",
        source="sample.docx",
        score=0.9,
        metadata={
            "file_name": "sample.docx",
            "page_number": 1,
            "section_title": "Page 1",
            "topic": topic,
            "image_type": image_type,
            "image_caption": f"{image_type} of {topic}",
            "storage_url": "extracted_assets/sample.png",
        },
    )


class MultiContentRetrievalTests(unittest.TestCase):
    def test_show_portugal_map_returns_only_portugal_map(self) -> None:
        with patch(
            "rag_hybrid.retrievers.search_collection",
            return_value=[_image_result("portugal", "map"), _image_result("machine", "dimension_diagram")],
        ):
            results = ImageRetriever().retrieve("show Portugal map")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content_type, "image")
        self.assertEqual(results[0].metadata["topic"], "portugal")
        self.assertEqual(results[0].metadata["image_type"], "map")

    def test_show_machine_dimension_image_returns_only_dimension_diagram(self) -> None:
        with patch(
            "rag_hybrid.retrievers.search_collection",
            return_value=[_image_result("portugal", "map"), _image_result("machine", "dimension_diagram")],
        ):
            results = ImageRetriever().retrieve("show machine dimension image")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content_type, "image")
        self.assertEqual(results[0].metadata["topic"], "machine")
        self.assertEqual(results[0].metadata["image_type"], "dimension_diagram")

    def test_portugal_history_uses_text_retrieval_only(self) -> None:
        self.assertEqual(classify_query("tell me about Portugal history"), QueryType.TEXT_QA)
        with patch("rag_hybrid.retrievers.TextRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.ImageRetriever.retrieve") as image_retrieve, \
             patch("rag_hybrid.retrievers.TableRetriever.retrieve") as table_retrieve:
            result = retrieve("tell me about Portugal history")

        self.assertEqual(result.retrieval_paths, ["smartcoolant_text"])
        image_retrieve.assert_not_called()
        table_retrieve.assert_not_called()

    def test_modbus_register_address_uses_table_retrieval(self) -> None:
        with patch(
            "rag_hybrid.retrievers.TableRetriever.retrieve",
            return_value=[
                RetrievalItem(
                    content_type="table",
                    document_id=str(uuid4()),
                    file_name="manual.pdf",
                    page_number=3,
                    text="Modbus register table",
                    metadata={"table_name": "Registers", "matched_rows": [["Address", "40001"]]},
                )
            ],
        ):
            result = retrieve("what is the Modbus register address?")

        self.assertEqual(result.query_type, QueryType.TABLE_LOOKUP.value)
        self.assertIn("neo4j_tables", result.retrieval_paths)
        self.assertEqual(result.table_results[0].metadata["table_name"], "Registers")

    def test_peacock_age_question_uses_table_retrieval(self) -> None:
        self.assertEqual(classify_query("what peacocks do at every age?"), QueryType.TABLE_LOOKUP)
        with patch("rag_hybrid.retrievers.TableRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.TextRetriever.retrieve") as text_retrieve:
            result = retrieve("what peacocks do at every age?")

        self.assertEqual(result.query_type, QueryType.TABLE_LOOKUP.value)
        self.assertIn("neo4j_tables", result.retrieval_paths)
        text_retrieve.assert_not_called()

    def test_mixed_query_returns_text_and_image_paths(self) -> None:
        with patch("rag_hybrid.retrievers.TextRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.ImageRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.TableRetriever.retrieve", return_value=[]):
            result = retrieve("explain the machine dimension and show the diagram")

        self.assertEqual(result.query_type, QueryType.MIXED_QUERY.value)
        self.assertIn("smartcoolant_text", result.retrieval_paths)
        self.assertIn("smartcoolant_images", result.retrieval_paths)

    def test_troubleshooting_retrieves_all_relevant_paths(self) -> None:
        with patch("rag_hybrid.retrievers.TextRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.ImageRetriever.retrieve", return_value=[]), \
             patch("rag_hybrid.retrievers.TableRetriever.retrieve", return_value=[]):
            result = retrieve("machine is not connecting to network")

        self.assertEqual(result.query_type, QueryType.TROUBLESHOOTING.value)
        self.assertIn("smartcoolant_text", result.retrieval_paths)
        self.assertIn("smartcoolant_images", result.retrieval_paths)
        self.assertIn("neo4j_tables", result.retrieval_paths)


if __name__ == "__main__":
    unittest.main()
