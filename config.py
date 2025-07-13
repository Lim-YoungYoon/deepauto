# config.py
import os
from typing import Set
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ──────────────────────────────
# 공통 초기화
# ──────────────────────────────
load_dotenv()  # .env 환경변수 로드


# ──────────────────────────────
# 파일·API 관련 기본 설정
# ──────────────────────────────
class APIConfig(BaseSettings):
    """REST API 서버 설정 (환경변수 접두어: API_)"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    rate_limit: int = 10
    max_pdf_upload_size: int = 50  # MB

    class Config:
        env_prefix = "API_"


class FileConfig(BaseSettings):
    """문서/임시 파일 경로 설정 (환경변수 접두어: FILE_)"""
    pdf_folder: str = "temp"
    temp_folder: str = "temp"
    allowed_extensions: Set[str] = {"pdf"}

    class Config:
        env_prefix = "FILE_"


# ──────────────────────────────
# LLM 및 RAG 파이프라인 설정
# ──────────────────────────────
class AgentDecisionConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,  # 결정적
        )


class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,  # 창의적이지만 사실적
        )


class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )
        # 마지막 20개의 메시지(Q&A 10쌍)만 대화 역사로 유지
        self.context_limit = 20


class RAGConfig:
    def __init__(self):
        # ── 백엔드 스토리지 ────────────────────────
        self.vector_db_type = "qdrant"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # ── 임베딩 ────────────────────────────────
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # ── LLM 파이프라인 ─────────────────────────
        base_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.llm = ChatOpenAI(model=base_model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)
        self.summarizer_model = ChatOpenAI(model=base_model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
        self.chunker_model = ChatOpenAI(model=base_model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0)
        self.response_generator_model = ChatOpenAI(model=base_model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.3)

        # ── RAG 세부 파라미터 ──────────────────────
        self.collection_name = "rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.top_k = 3
        self.vector_search_type = "similarity"  # 또는 'mmr'
        self.reranker_model = "BAAI/bge-reranker-v2-m3"
        self.reranker_top_k = 3
        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.1
        self.context_limit = 20

        # ── 외부 토큰 ───────────────────────────────
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


# ──────────────────────────────
# 기타 서브시스템 설정
# ──────────────────────────────
class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": True,  # 수정: RAG_AGENT 검증 활성화
            "WEB_SEARCH_AGENT": False,
        }
        self.validation_timeout = 300
        self.default_action = "reject"


class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_pdf_upload = True
        # self.max_chat_history = 50


# ──────────────────────────────
# 통합 Config
# ──────────────────────────────
class Config:
    """
    >>> from config import Config
    >>> cfg = Config()
    >>> cfg.rag.chunk_size
    512
    """

    def __init__(self):
        # 주요 서브 설정
        self.api = APIConfig()
        self.file = FileConfig()
        self.agent_decision = AgentDecisionConfig()
        self.conversation = ConversationConfig()
        self.web_search = WebSearchConfig()
        self.rag = RAGConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()

        # 공통 옵션
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # 마지막 20개 메시지(Q&A 10쌍)

        # 필수 디렉터리 생성
        self._create_directories()

    # ────────── 내부 메서드 ──────────
    def _create_directories(self):
        """uploads, temp, data 경로가 없으면 생성"""
        dirs = [self.file.pdf_folder, self.file.temp_folder, "data"]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
