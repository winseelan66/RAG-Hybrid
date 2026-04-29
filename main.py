from rag_hybrid.db import initialize_pgvector_schema


def main() -> None:
    initialize_pgvector_schema()
    print("pgVector schema initialized.")


if __name__ == "__main__":
    main()
