from dataclasses import dataclass
from pathlib import Path
import os


ENV_FILE_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_dotenv(env_file_path: Path = ENV_FILE_PATH) -> None:
    if not env_file_path.exists():
        return

    for raw_line in env_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_int_env(name: str) -> int:
    return int(_get_required_env(name))


def _get_csv_env(name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in _get_required_env(name).split(",") if item.strip())


@dataclass(frozen=True)
class PostgresSettings:
    hosts: tuple[str, ...]
    port: int
    user: str
    password: str
    database: str

    def dsn(self, host: str) -> str:
        return f"host={host} port={self.port} user={self.user} password={self.password} dbname={self.database}"


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class QdrantSettings:
    url: str
    collection: str
    text_collection: str
    image_collection: str


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    system_prompt: str


@dataclass(frozen=True)
class AppUiSettings:
    page_title: str
    page_icon: str
    page_layout: str
    nav_embed_label: str
    nav_chat_label: str
    workspace_title: str
    workspace_caption: str
    upload_page_title: str
    upload_page_description: str
    chat_page_title: str
    chat_page_description: str
    upload_button_label: str
    file_uploader_label: str
    chat_input_placeholder: str
    streamlit_host: str
    streamlit_port: int
    supported_upload_types: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalSettings:
    vector_search_limit: int
    graph_search_limit: int


@dataclass(frozen=True)
class ExtractionSettings:
    embedding_dimension: int
    chunk_size: int


@dataclass(frozen=True)
class LoggingSettings:
    file_name: str


@dataclass(frozen=True)
class AppSettings:
    postgres: PostgresSettings
    neo4j: Neo4jSettings
    qdrant: QdrantSettings
    openai: OpenAISettings
    ui: AppUiSettings
    retrieval: RetrievalSettings
    extraction: ExtractionSettings
    logging: LoggingSettings


def get_settings() -> AppSettings:
    load_dotenv()
    postgres_hosts = _get_csv_env("PGVECTOR_HOSTS")

    return AppSettings(
        postgres=PostgresSettings(
            hosts=postgres_hosts,
            port=_get_int_env("PGVECTOR_PORT"),
            user=_get_required_env("PGVECTOR_USER"),
            password=_get_required_env("PGVECTOR_PASSWORD"),
            database=_get_required_env("PGVECTOR_DB"),
        ),
        neo4j=Neo4jSettings(
            uri=_get_required_env("NEO4J_URI"),
            user=_get_required_env("NEO4J_USER"),
            password=_get_required_env("NEO4J_PASSWORD"),
            database=_get_required_env("NEO4J_DATABASE"),
        ),
        qdrant=QdrantSettings(
            url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
            collection=os.getenv("QDRANT_COLLECTION", "documents_embeddings"),
            text_collection=os.getenv("QDRANT_TEXT_COLLECTION", "smartcoolant_text"),
            image_collection=os.getenv("QDRANT_IMAGE_COLLECTION", "smartcoolant_images"),
        ),
        openai=OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=_get_required_env("OPENAI_MODEL"),
            system_prompt=_get_required_env("OPENAI_SYSTEM_PROMPT"),
        ),
        ui=AppUiSettings(
            page_title=_get_required_env("APP_PAGE_TITLE"),
            page_icon=_get_required_env("APP_PAGE_ICON"),
            page_layout=_get_required_env("APP_PAGE_LAYOUT"),
            nav_embed_label=_get_required_env("APP_NAV_EMBED_LABEL"),
            nav_chat_label=_get_required_env("APP_NAV_CHAT_LABEL"),
            workspace_title=_get_required_env("APP_WORKSPACE_TITLE"),
            workspace_caption=_get_required_env("APP_WORKSPACE_CAPTION"),
            upload_page_title=_get_required_env("APP_UPLOAD_PAGE_TITLE"),
            upload_page_description=_get_required_env("APP_UPLOAD_PAGE_DESCRIPTION"),
            chat_page_title=_get_required_env("APP_CHAT_PAGE_TITLE"),
            chat_page_description=_get_required_env("APP_CHAT_PAGE_DESCRIPTION"),
            upload_button_label=_get_required_env("APP_UPLOAD_BUTTON_LABEL"),
            file_uploader_label=_get_required_env("APP_FILE_UPLOADER_LABEL"),
            chat_input_placeholder=_get_required_env("APP_CHAT_INPUT_PLACEHOLDER"),
            streamlit_host=_get_required_env("APP_STREAMLIT_HOST"),
            streamlit_port=_get_int_env("APP_STREAMLIT_PORT"),
            supported_upload_types=_get_csv_env("APP_SUPPORTED_UPLOAD_TYPES"),
        ),
        retrieval=RetrievalSettings(
            vector_search_limit=_get_int_env("APP_VECTOR_SEARCH_LIMIT"),
            graph_search_limit=_get_int_env("APP_GRAPH_SEARCH_LIMIT"),
        ),
        extraction=ExtractionSettings(
            embedding_dimension=_get_int_env("APP_EMBEDDING_DIMENSION"),
            chunk_size=_get_int_env("APP_CHUNK_SIZE"),
        ),
        logging=LoggingSettings(
            file_name=_get_required_env("APP_LOG_FILE_NAME"),
        ),
    )
