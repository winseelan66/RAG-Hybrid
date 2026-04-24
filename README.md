# RAG-Hybrid
## Vector DB and Neo4j
# 1. Hybrid RAG (Mandatory)
# Combine:
### •	Vector RAG → for semantic search (text) 
### •	Graph RAG (Neo4j) → for relationships (components, specs, dependencies)
Do both searches in parallel, not sequentially
1.	Vector DB 
o	Retrieves: text + table summaries + image captions 
2.	Neo4j 
o	Retrieves: structured data + relationships 
3.	LLM combines both results
