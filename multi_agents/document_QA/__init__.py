import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .doc_parser import PDFParser
from .content_processor import ContentProcessor
from .vectorstore_qdrant import VectorStore
from .reranker import Reranker
from .query_expander import QueryExpander
from .response_generator import ResponseGenerator

class DocumentRAG:
    """
    문서 RAG (Retrieval-Augmented Generation) 시스템으로 문서 처리 및 쿼리를 위한 시스템입니다.
    """
    
    def __init__(self, config):
        """
        RAG 에이전트를 초기화합니다.
        
        Args:
            config: RAG 설정이 포함된 설정 객체
        """
        self._setup_logging()
        self.logger.info("RAG 시스템 초기화 중")
        
        self.config = config
        self.parsed_content_dir = self.config.rag.parsed_content_dir
        
        # 컴포넌트 초기화
        self._initialize_components()
    
    def _setup_logging(self) -> None:
        """RAG 시스템을 위한 로깅을 설정합니다."""
        self.logger = logging.getLogger(f"{self.__module__}")
    
    def _initialize_components(self) -> None:
        """모든 RAG 시스템 컴포넌트를 초기화합니다."""
        self.doc_parser = PDFParser()
        self.content_processor = ContentProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.reranker = Reranker(self.config)
        self.query_expander = QueryExpander(self.config)
        self.response_generator = ResponseGenerator(self.config)
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        디렉터리의 모든 파일을 RAG 시스템에 수집합니다.
        
        Args:
            directory_path: 수집할 파일이 포함된 디렉터리 경로
            
        Returns:
            수집 결과가 포함된 딕셔너리
        """
        start_time = time.time()
        self.logger.info(f"디렉터리에서 파일 수집 중: {directory_path}")
        
        try:
            files = self._get_files_from_directory(directory_path)
            
            if not files:
                return self._create_empty_ingestion_result(start_time)
            
            return self._process_directory_files(files, start_time)
            
        except Exception as e:
            self.logger.error(f"디렉터리 수집 중 오류: {e}")
            return self._create_error_ingestion_result(str(e), start_time)
    
    def _get_files_from_directory(self, directory_path: str) -> List[str]:
        """지정된 디렉터리에서 모든 파일을 가져옵니다."""
        if not os.path.isdir(directory_path):
            raise ValueError(f"디렉터리를 찾을 수 없습니다: {directory_path}")
        
        files = [
            os.path.join(directory_path + '/', f) 
            for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        
        if not files:
            self.logger.warning(f"디렉터리에서 파일을 찾을 수 없습니다: {directory_path}")
        
        return files
    
    def _create_empty_ingestion_result(self, start_time: float) -> Dict[str, Any]:
        """빈 디렉터리 수집을 위한 결과를 생성합니다."""
        return {
            "success": True,
            "documents_ingested": 0,
            "chunks_processed": 0,
            "processing_time": time.time() - start_time
        }
    
    def _create_error_ingestion_result(self, error: str, start_time: float) -> Dict[str, Any]:
        """실패한 수집을 위한 오류 결과를 생성합니다."""
        return {
            "success": False,
            "error": error,
            "processing_time": time.time() - start_time
        }
    
    def _process_directory_files(self, files: List[str], start_time: float) -> Dict[str, Any]:
        """디렉터리의 모든 파일을 처리합니다."""
        total_chunks_processed = 0
        successful_ingestions = 0
        failed_ingestions = 0
        failed_files = []
        
        for file_path in files:
            self.logger.info(f"파일 처리 중 {successful_ingestions + failed_ingestions + 1}/{len(files)}: {file_path}")
            
            try:
                result = self.ingest_file(file_path)
                if result["success"]:
                    successful_ingestions += 1
                    total_chunks_processed += result.get("chunks_processed", 0)
                else:
                    failed_ingestions += 1
                    failed_files.append({"file": file_path, "error": result.get("error", "Unknown error")})
            except Exception as e:
                self.logger.error(f"파일 처리 중 오류 {file_path}: {e}")
                failed_ingestions += 1
                failed_files.append({"file": file_path, "error": str(e)})
        
        return {
            "success": True,
            "documents_ingested": successful_ingestions,
            "failed_documents": failed_ingestions,
            "failed_files": failed_files,
            "chunks_processed": total_chunks_processed,
            "processing_time": time.time() - start_time
        }
    
    def ingest_file(self, document_path: str) -> Dict[str, Any]:
        """
        단일 파일을 RAG 시스템에 수집합니다.
        
        Args:
            document_path: 수집할 파일 경로
            
        Returns:
            수집 결과가 포함된 딕셔너리
        """
        start_time = time.time()
        self.logger.info(f"파일 수집 중: {document_path}")

        try:
            # 1단계: 문서 파싱
            self.logger.info("1. 문서 파싱 및 이미지 추출 중...")
            parsed_document, images = self.doc_parser.parse_document(document_path, self.parsed_content_dir)
            self.logger.info(f"   문서 파싱 완료 및 {len(images)}개 이미지 추출")

            # 2단계: 이미지 요약
            self.logger.info("2. 이미지 요약 중...")
            image_summaries = self.content_processor.summarize_images(images)
            self.logger.info(f"   {len(image_summaries)}개 이미지 요약 생성")

            # 3단계: 이미지 요약과 함께 문서 포맷팅
            self.logger.info("3. 이미지 요약과 함께 문서 포맷팅 중...")
            formatted_document = self.content_processor.format_document_with_images(parsed_document, image_summaries)

            # 4단계: 문서를 의미론적 섹션으로 청킹
            self.logger.info("4. 문서를 의미론적 섹션으로 청킹 중...")
            document_chunks = self.content_processor.chunk_document(formatted_document)
            self.logger.info(f"   문서가 {len(document_chunks)}개 청크로 분할됨")

            # 5단계: 벡터 스토어 및 문서 스토어 생성
            self.logger.info("5. 벡터 스토어 지식 베이스 생성 중...")
            self.vector_store.create_vectorstore(
                document_chunks=document_chunks, 
                document_path=document_path
            )
            
            return {
                "success": True,
                "documents_ingested": 1,
                "chunks_processed": len(document_chunks),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            self.logger.error(f"파일 수집 중 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        RAG 시스템으로 쿼리를 처리합니다.
        
        Args:
            query: 쿼리 문자열
            chat_history: 컨텍스트를 위한 선택적 채팅 히스토리
            
        Returns:
            응답 딕셔너리
        """
        start_time = time.time()
        self.logger.info(f"RAG 에이전트 쿼리 처리 중: {query}")
        
        try:
            # 1단계: 쿼리 확장
            expanded_query = self._expand_query(query)
            
            # 2단계: 문서 검색
            retrieved_documents = self._retrieve_documents(expanded_query)
            if not retrieved_documents:
                return self._create_no_documents_response(start_time)
            
            # 3단계: 문서 재순위화
            reranked_documents, reranked_top_k_picture_paths = self._rerank_documents(expanded_query, retrieved_documents)
            
            # 4단계: 응답 생성
            response = self._generate_final_response(expanded_query, reranked_documents, reranked_top_k_picture_paths, chat_history)
            
            # 타이밍 정보 추가
            response["processing_time"] = time.time() - start_time
            print(response)
            return response
        
        except Exception as e:
            self.logger.error(f"쿼리 처리 중 오류: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_error_response(str(e), start_time)
    
    def _expand_query(self, query: str) -> str:
        """더 나은 검색을 위해 사용자 쿼리를 확장합니다."""
        self.logger.info(f"1. 쿼리 확장 중: '{query}'")
        expansion_result = self.query_expander.expand_query(query)
        expanded_query = expansion_result["expanded_query"]
        self.logger.info(f"   원본: '{query}'")
        self.logger.info(f"   확장: '{expanded_query}'")
        return expanded_query
    
    def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """쿼리에 대한 관련 문서를 검색합니다."""
        self.logger.info(f"2. 쿼리에 대한 관련 문서 검색 중: '{query}'")
        try:
            vectorstore, docstore = self.vector_store.load_vectorstore()
            retrieved_documents = self.vector_store.retrieve_relevant_chunks(
                query=query,
                vectorstore=vectorstore,
                docstore=docstore,
            )
            self.logger.info(f"   Retrieved {len(retrieved_documents)} relevant document chunks")
            return retrieved_documents
        except ValueError as e:
            if "does not exist" in str(e):
                self.logger.warning("No documents have been ingested yet.")
                return []
            else:
                raise e
    
    def _rerank_documents(self, query: str, retrieved_documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Rerank the retrieved documents."""
        self.logger.info(f"3. Reranking the retrieved documents")
        if self.reranker and len(retrieved_documents) > 1:
            reranked_documents, reranked_top_k_picture_paths = self.reranker.rerank(query, retrieved_documents, self.parsed_content_dir)
            self.logger.info(f"   Reranked retrieved documents and chose top {len(reranked_documents)}")
            self.logger.info(f"   Found {len(reranked_top_k_picture_paths)} referenced images")
        else:
            self.logger.info(f"   Could not rerank the retrieved documents, falling back to original scores")
            reranked_documents = retrieved_documents
            reranked_top_k_picture_paths = []
        
        return reranked_documents, reranked_top_k_picture_paths
    
    def _generate_final_response(
        self, 
        query: str, 
        reranked_documents: List[Dict[str, Any]], 
        reranked_top_k_picture_paths: List[str], 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate the final response."""
        self.logger.info("4. Generating response...")
        return self.response_generator.generate_response(
            query=query,
            retrieved_docs=reranked_documents,
            picture_paths=reranked_top_k_picture_paths,
            chat_history=chat_history
        )
    
    def _create_no_documents_response(self, start_time: float) -> Dict[str, Any]:
        """Create response when no documents are available."""
        return {
            "response": "죄송합니다. 아직 문서가 업로드되지 않았습니다. 먼저 PDF 문서를 업로드해주세요.",
            "sources": [],
            "confidence": 0.0,
            "processing_time": time.time() - start_time
        }
    
    def _create_error_response(self, error: str, start_time: float) -> Dict[str, Any]:
        """Create error response."""
        return {
            "response": f"I encountered an error while processing your query: {error}",
            "sources": [],
            "confidence": 0.0,
            "processing_time": time.time() - start_time
        }