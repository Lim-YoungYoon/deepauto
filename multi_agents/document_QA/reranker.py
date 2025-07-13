import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import CrossEncoder

class Reranker:
    """
    교차 인코더 모델을 사용하여 검색된 문서를 재순위화하여 더 정확한 결과를 제공합니다.
    """
    
    def __init__(self, config):
        """
        설정으로 재순위화기를 초기화합니다.
        
        Args:
            config: 재순위화기 설정을 포함하는 설정 객체
        """
        self.logger = logging.getLogger(__name__)
        self._load_model(config)
    
    def _load_model(self, config) -> None:
        """재순위화를 위한 교차 인코더 모델을 로드합니다."""
        try:
            self.model_name = config.rag.reranker_model
            self.logger.info(f"재순위화 모델 로딩 중: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.top_k = config.rag.reranker_top_k
        except Exception as e:
            self.logger.error(f"재순위화 모델 로딩 오류: {e}")
            raise
    
    def rerank(self, query: str, documents: Union[List[Dict[str, Any]], List[str]], parsed_content_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        교차 인코더를 사용하여 쿼리 관련성에 따라 문서를 재순위화합니다.
        
        Args:
            query: 사용자 쿼리
            documents: 문서 목록(딕셔너리) 또는 문자열 목록
            parsed_content_dir: 파싱된 콘텐츠가 포함된 디렉토리
            
        Returns:
            (재순위화된_문서, 그림_참조_경로) 튜플
        """
        try:
            if not documents:
                return [], []
            
            # 문서를 일관된 형식으로 정규화
            normalized_documents = self._normalize_documents(documents)
            
            # 관련성 점수 가져오기
            scores = self._get_relevance_scores(query, normalized_documents)
            
            # 문서에 점수 추가
            scored_documents = self._add_scores_to_documents(normalized_documents, scores)
            
            # 결합 점수로 정렬
            reranked_docs = sorted(scored_documents, key=lambda x: x["combined_score"], reverse=True)
            
            # 필요시 top_k로 제한
            if self.top_k and len(reranked_docs) > self.top_k:
                reranked_docs = reranked_docs[:self.top_k]
            
            # 그림 참조 추출
            picture_reference_paths = self._extract_picture_references(reranked_docs, parsed_content_dir)
            
            return reranked_docs, picture_reference_paths
            
        except Exception as e:
            self.logger.error(f"재순위화 중 오류 발생: {e}")
            # 재순위화 실패 시 원래 순위로 폴백
            self.logger.warning("원래 순위로 폴백합니다")
            return documents, []
    
    def _normalize_documents(self, documents: Union[List[Dict[str, Any]], List[str]]) -> List[Dict[str, Any]]:
        """문서를 일관된 딕셔너리 형식으로 정규화합니다."""
        if not documents:
            return []
        
        # 문서가 문자열 목록인 경우 딕셔너리로 변환
        if isinstance(documents[0], str):
            return self._convert_strings_to_dicts(documents)
        
        # 문서가 딕셔너리 목록인 경우 필수 필드가 존재하는지 확인
        elif isinstance(documents[0], dict):
            return self._ensure_dict_fields(documents)
        
        else:
            raise ValueError(f"지원되지 않는 문서 형식: {type(documents[0])}")
    
    def _convert_strings_to_dicts(self, documents: List[str]) -> List[Dict[str, Any]]:
        """문자열 목록을 딕셔너리 목록으로 변환합니다."""
        docs_list = []
        for i, doc_text in enumerate(documents):
            docs_list.append({
                "id": i,
                "content": doc_text,
                "score": 1.0  # 기본 점수
            })
        return docs_list
    
    def _ensure_dict_fields(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """문서 딕셔너리에 모든 필수 필드가 존재하는지 확인합니다."""
        for i, doc in enumerate(documents):
            # ID가 존재하는지 확인
            if "id" not in doc:
                doc["id"] = i
            # 점수가 존재하는지 확인
            if "score" not in doc:
                doc["score"] = 1.0
            # 콘텐츠가 존재하는지 확인
            if "content" not in doc:
                if "text" in doc:  # 일부 구현에서는 "text"를 사용할 수 있음
                    doc["content"] = doc["text"]
                else:
                    doc["content"] = f"Document {i}"
        
        return documents
    
    def _get_relevance_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """쿼리-문서 쌍에 대한 관련성 점수를 가져옵니다."""
        # 점수 매기기를 위한 쿼리-문서 쌍 생성
        pairs = [(query, doc["content"]) for doc in documents]
        
        # 관련성 점수 가져오기
        return self.model.predict(pairs)
    
    def _add_scores_to_documents(self, documents: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        """문서에 점수를 추가하고 결합 점수를 계산합니다."""
        for i, score in enumerate(scores):
            documents[i]["rerank_score"] = float(score)  # 재순위화에서 나온 새로운 점수 저장
            # 원래 문서에 점수가 없으면 재순위화 점수 사용
            if "score" not in documents[i]:
                documents[i]["score"] = 1.0
            # 원래 점수와 재순위화 점수를 결합(평균)
            documents[i]["combined_score"] = (documents[i]["score"] + float(score)) / 2
        
        return documents
    
    def _extract_picture_references(self, documents: List[Dict[str, Any]], parsed_content_dir: str) -> List[str]:
        """문서에서 그림 참조를 추출합니다."""
        picture_reference_paths = []
        
        for doc in documents:
            matches = re.finditer(r"picture_counter_(\d+)", doc["content"])
            for match in matches:
                counter_value = int(match.group(1))
                # 문서 소스와 카운터를 기반으로 그림 경로 생성
                doc_basename = os.path.splitext(doc['source'])[0]  # 파일 확장자 제거
                picture_path = os.path.join("http://localhost:8000/", parsed_content_dir + "/" + f"{doc_basename}-picture-{counter_value}.png")
                picture_reference_paths.append(picture_path)
        
        return picture_reference_paths