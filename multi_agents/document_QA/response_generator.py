import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    검색된 컨텍스트와 사용자 쿼리를 기반으로 응답을 생성합니다.
    """
    
    # 상수
    DOCUMENT_SECTION_SEPARATOR = "\n\n===DOCUMENT SECTION===\n\n"
    SOURCES_HEADER = "\n\n##### Source documents:"
    IMAGES_HEADER = "\n\n##### Reference images:"
    
    def __init__(self, config):
        """
        응답 생성기를 초기화합니다.
        
        Args:
            config: 설정 객체
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        언어 모델을 위한 프롬프트를 구성합니다.
        
        Args:
            query: 사용자 쿼리
            context: 검색된 문서에서 포맷된 컨텍스트
            chat_history: 선택적 채팅 히스토리
            
        Returns:
            완성된 프롬프트 문자열
        """
        table_instructions = self._get_table_instructions()
        response_format_instructions = self._get_response_format_instructions()
        
        # 프롬프트 구성
        prompt = f"""You are an intelligent document QA system that connects user queries to the right expert, delivering accurate answers based on trusted documents.

        Here are the last few messages from our conversation:
        
        {chat_history}

        The user has asked the following question:
        {query}

        I've retrieved the following information to help answer this question:

        {context}

        {table_instructions}

        {response_format_instructions}

        Based on the provided documents, please answer the user's question thoroughly but concisely. If the documents do not contain the answer, clearly acknowledge the limitations of the available information.

        Do not cite or fabricate any source that is not included in the provided context. Do not make up any source link.

        Document QA Response:"""

        return prompt
    
    def _get_table_instructions(self) -> str:
        """테이블 포맷팅 지침을 가져옵니다."""
        return """
        Some of the retrieved information is presented in table format. When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers, like this:
            | Column1 | Column2 | Column3 |
            |---------|---------|---------|
            | Value1  | Value2  | Value3  |
        2. Re-format the table structure to make it easier to read and understand
        3. If any new component is introduced during re-formatting of the table, mention it explicitly
        4. Clearly interpret the tabular data in your response
        5. Reference the relevant table when presenting specific data points
        6. If appropriate, summarize trends or patterns shown in the tables
        7. If only reference numbers are mentioned and you can fetch the corresponding values like research paper title or authors from the context, replace the reference numbers with the actual values
        """
    
    def _get_response_format_instructions(self) -> str:
        """응답 포맷팅 지침을 가져옵니다."""
        return """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, state: "I don't have enough information to answer this question based on the provided context."
        3. Do not use prior knowledge not contained in the context.
        5. Be concise and accurate.
        6. Provide a well-structured response with heading, sub-headings and tabular structure if required in markdown format based on retrieved knowledge. Keep the headings and sub-headings small sized.
        7. Only provide sections that are meaningful to have in a chatbot reply. For example, do not explicitly mention references.
        8. If values are involved, make sure to respond with perfect values present in context. Do not make up values.
        9. Do not repeat the question in the answer or response."""

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
        """
        검색된 문서를 기반으로 응답을 생성합니다.
        
        Args:
            query: 사용자 쿼리
            retrieved_docs: 검색된 문서 딕셔너리 리스트
            picture_paths: 이미지 경로 리스트
            chat_history: 선택적 채팅 히스토리
            
        Returns:
            응답 텍스트와 소스 정보를 포함한 딕셔너리
        """
        try:
            # 컨텍스트를 위한 문서에서 내용 추출
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # 검색된 문서들을 단일 컨텍스트로 결합
            context = self.DOCUMENT_SECTION_SEPARATOR.join(doc_texts)
            
            # 프롬프트 구성
            prompt = self._build_prompt(query, context, chat_history)
            
            # 응답 생성
            response = self.response_generator_model.invoke(prompt)
            
            # 인용을 위한 소스 추출
            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(retrieved_docs)

            # 소스와 이미지가 포함된 최종 응답 구성
            final_response = self._build_final_response(response.content, sources, picture_paths)
            
            # 최종 응답 포맷팅
            result = {
                "response": final_response,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self._create_error_response()
    
    def _build_final_response(self, response_content: str, sources: List[Dict[str, str]], picture_paths: List[str]) -> str:
        """소스와 이미지가 포함된 최종 응답을 구성합니다."""
        response_with_source = response_content
        
        # 응답에 소스 추가
        if self.include_sources:
            response_with_source += self.SOURCES_HEADER
            for current_source in sources:
                source_path = current_source['path']
                source_title = current_source['title']
                response_with_source += f"\n- [{source_title}]({source_path})"
        
        # 응답에 이미지 경로 추가
        response_with_source += self.IMAGES_HEADER
        for picture_path in picture_paths:
            response_with_source += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
        
        return response_with_source
    
    def _create_error_response(self) -> Dict[str, Any]:
        """오류 응답을 생성합니다."""
        return {
            "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
            "sources": [],
            "confidence": 0.0
        }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        인용을 위해 검색된 문서에서 소스 정보를 추출합니다.
        
        Args:
            documents: 검색된 문서 딕셔너리 리스트
            
        Returns:
            소스 정보 딕셔너리 리스트
        """
        sources = []
        seen_sources = set()  # 중복을 피하기 위해 고유 소스 추적
        
        for doc in documents:
            # source와 source_path 추출
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # 소스 정보가 없으면 건너뛰기
            if not source:
                continue
                
            # 이 소스에 대한 고유 식별자 생성
            source_id = f"{source}|{source_path}"
            
            # 이미 포함된 소스이면 건너뛰기
            if source_id in seen_sources:
                continue
                
            # 소스 리스트에 추가
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # 점수에 따라 소스를 높은 순서대로 정렬
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 정렬에 사용된 점수를 제거하여 최종 소스 리스트 포맷팅
        return self._format_sources(sources)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """정렬에 사용된 점수를 제거하여 소스를 포맷팅합니다."""
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        검색된 문서를 기반으로 신뢰도 점수를 계산합니다.
        
        Args:
            documents: 검색된 문서들
            
        Returns:
            0과 1 사이의 신뢰도 점수
        """
        if not documents:
            return 0.0
            
        # 가능하면 결합 점수(재순위 지정 및 코사인 유사도)를 사용하고, 그렇지 않으면 원래 점수 사용
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:2]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:2]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:2]]
            
        # 상위 2개 문서 점수의 평균 또는 2개 미만인 경우 더 적은 수
        return sum(scores) / len(scores) if scores else 0.0