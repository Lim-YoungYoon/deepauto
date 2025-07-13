import logging
from typing import List, Dict, Any

class QueryExpander:
    """
    사용자 쿼리를 용어로 확장하여 검색을 개선합니다.
    """
    
    def __init__(self, config):
        """쿼리 확장기를 설정으로 초기화합니다."""
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        원본 쿼리를 관련 용어로 확장합니다.
        
        Args:
            original_query: 사용자의 원본 쿼리
            
        Returns:
            원본 및 확장된 쿼리가 포함된 딕셔너리
        """
        self.logger.info(f"쿼리 확장 중: {original_query}")
        
        # 확장 생성
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> Any:
        """LLM을 사용하여 관련 용어로 쿼리를 확장합니다."""
        prompt = f"""
        User Query: {query}

        Expand the query only if you feel like it is required, otherwise keep the user query intact.
        Be specific to the domain mentioned in the user query, do not add concepts from unrelated domains.
        If the user query asks about answering in tabular format, include that in the expanded query and do not answer in tabular format yourself.
        Provide only the expanded query without explanations.
        """
        
        return self.model.invoke(prompt)