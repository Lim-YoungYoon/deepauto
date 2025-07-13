import os
import re
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.documents import Document
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams, OptimizersConfigDiff

class VectorStore:
    """
    벡터 스토어 생성, 문서 수집, 관련 문서 검색
    """
    
    def __init__(self, config):
        """설정으로 벡터 스토어를 초기화합니다."""
        self.logger = logging.getLogger(__name__)
        self._setup_config(config)
        self._initialize_client()
    
    def _setup_config(self, config) -> None:
        """설정 매개변수를 설정합니다."""
        self.collection_name = config.rag.collection_name
        self.embedding_dim = config.rag.embedding_dim
        self.distance_metric = config.rag.distance_metric
        self.embedding_model = config.rag.embedding_model
        self.retrieval_top_k = config.rag.top_k
        self.vector_search_type = config.rag.vector_search_type
        self.vectorstore_local_path = config.rag.vector_local_path
        self.docstore_local_path = config.rag.doc_local_path
    
    def _initialize_client(self) -> None:
        """Qdrant 클라이언트를 초기화합니다."""
        self.client = QdrantClient(path=self.vectorstore_local_path)

    def _does_collection_exist(self) -> bool:
        """Qdrant에 컬렉션이 이미 존재하는지 확인합니다."""
        try:
            collection_info = self.client.get_collections()
            collection_names = [collection.name for collection in collection_info.collections]
            return self.collection_name in collection_names
        except Exception as e:
            self.logger.error(f"컬렉션 존재 확인 중 오류: {e}")
            return False

    def _create_collection(self) -> None:
        """밀집 벡터와 희소 벡터가 있는 새 컬렉션을 생성합니다."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=self.embedding_dim, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            self.logger.info(f"새 컬렉션 생성됨: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"컬렉션 생성 중 오류: {e}")
            raise e
            
    def load_vectorstore(self) -> Tuple[QdrantVectorStore, LocalFileStore]:
        """
        새로운 문서를 수집하지 않고 검색 작업을 위해 기존 벡터스토어와 문서스토어를 로드합니다.
        
        Returns:
            (vectorstore, docstore)를 포함하는 튜플
        """
        # 컬렉션이 존재하는지 확인
        if not self._does_collection_exist():
            self.logger.error(f"Collection {self.collection_name} does not exist. Please ingest documents first.")
            raise ValueError(f"Collection {self.collection_name} does not exist")
            
        # 희소 임베딩 설정
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # 벡터 스토어 초기화
        qdrant_vectorstore = self._create_qdrant_vectorstore(sparse_embeddings)
        
        # 문서 저장소
        docstore = LocalFileStore(self.docstore_local_path)
        
        self.logger.info(f"Successfully loaded existing vectorstore and docstore")
        return qdrant_vectorstore, docstore

    def create_vectorstore(
            self,
            document_chunks: List[str],
            document_path: str,
        ) -> None:
        """
        문서 청크에서 벡터 스토어를 생성하거나 기존 스토어에 문서를 업서트합니다.
        
        Args:
            document_chunks: 문서 청크 목록
            document_path: 원본 문서 경로
        """
        # 각 청크에 대한 고유 ID 생성
        doc_ids = [str(uuid4()) for _ in range(len(document_chunks))]
        
        # langchain 문서 생성
        langchain_documents = self._create_langchain_documents(document_chunks, document_path, doc_ids)
        
        # 희소 임베딩 설정
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # 컬렉션이 존재하는지 확인하고, 없으면 생성
        collection_exists = self._does_collection_exist()
        if not collection_exists:
            self._create_collection()
            self.logger.info(f"Created new collection: {self.collection_name}")
        else:
            self.logger.info(f"Collection {self.collection_name} already exists, will upsert documents")
        
        # 벡터 스토어 초기화
        qdrant_vectorstore = self._create_qdrant_vectorstore(sparse_embeddings)
        
        # 부모 문서용 문서 저장소
        docstore = LocalFileStore(self.docstore_local_path)
        
        # 벡터 및 문서 스토어에 문서 수집
        self._ingest_documents(qdrant_vectorstore, docstore, langchain_documents, doc_ids, document_chunks)
    
    def _create_langchain_documents(self, document_chunks: List[str], document_path: str, doc_ids: List[str]) -> List[Document]:
        """청크에서 langchain 문서를 생성합니다."""
        langchain_documents = []
        for id_idx, chunk in enumerate(document_chunks):
            langchain_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": os.path.basename(document_path),
                        "doc_id": doc_ids[id_idx],
                        "source_path": os.path.join("http://localhost:8000/", document_path)
                    }
                )
            )
        
        return langchain_documents
    
    def _create_qdrant_vectorstore(self, sparse_embeddings: FastEmbedSparse) -> QdrantVectorStore:
        """Qdrant 벡터 스토어 인스턴스를 생성합니다."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    
    def _ingest_documents(
        self, 
        qdrant_vectorstore: QdrantVectorStore, 
        docstore: LocalFileStore, 
        langchain_documents: List[Document], 
        doc_ids: List[str], 
        document_chunks: List[str]
    ) -> None:
        """벡터 및 문서 스토어에 문서를 수집합니다."""
        # 벡터 스토어에 문서 추가
        qdrant_vectorstore.add_documents(documents=langchain_documents, ids=doc_ids)
        
        # 저장하기 전에 문자열 청크를 바이트로 인코딩
        encoded_chunks = [chunk.encode('utf-8') for chunk in document_chunks]
        docstore.mset(list(zip(doc_ids, encoded_chunks)))

    def retrieve_relevant_chunks(
            self,
            query: str,
            vectorstore: QdrantVectorStore,
            docstore: LocalFileStore,
        ) -> List[Dict[str, Any]]:
        """
        쿼리를 기반으로 관련 청크를 검색합니다.
        
        Args:
            query: 사용자 쿼리
            vectorstore: 임베딩을 포함하는 벡터 스토어
            docstore: 실제 콘텐츠를 포함하는 문서 스토어
            
        Returns:
            콘텐츠와 점수가 포함된 검색된 문서 딕셔너리 목록
        """
        # 쿼리 정규화
        normalized_query = self._normalize_query(query)
        
        # similarity_search_with_score를 사용하여 문서와 점수 가져오기
        results = vectorstore.similarity_search_with_score(
            query=normalized_query,
            k=self.retrieval_top_k
        )
        
        return self._format_retrieved_documents(results, docstore)
    
    def _normalize_query(self, query: Any) -> str:
        """쿼리를 문자열 형식으로 정규화합니다."""
        if isinstance(query, dict):
            return query.get("text") or query.get("query") or str(query)
        return str(query)
    
    def _format_retrieved_documents(self, results: List[Tuple[Document, float]], docstore: LocalFileStore) -> List[Dict[str, Any]]:
        """출력을 위해 검색된 문서를 포맷합니다."""
        retrieved_docs = []
        
        for chunk, score in results:
            # 문서 스토어에서 전체 문서를 바이트로 가져와서 문자열로 디코딩
            doc_content_bytes = docstore.mget([chunk.metadata['doc_id']])[0]
            doc_content = doc_content_bytes.decode('utf-8')
            
            # reranker가 기대하는 형식으로 문서 딕셔너리 생성
            doc_dict = {
                "id": chunk.metadata['doc_id'],
                "content": doc_content,
                "score": score,  # 실제 유사도 점수 사용
                "source": chunk.metadata['source'],
                "source_path": chunk.metadata['source_path'],
            }
            retrieved_docs.append(doc_dict)
        
        return retrieved_docs