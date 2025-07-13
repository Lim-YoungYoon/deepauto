from typing import List, Dict, Any, Optional

from .web_search_agent import WebSearchAgent


class TavilySearch:
    """
    Tavily 웹 검색 결과를 처리하고 적절한 LLM으로 라우팅하여 응답을 생성하는 에이전트입니다.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Tavily 검색 처리 에이전트를 초기화합니다.
        
        Args:
            config: 필요한 설정을 포함하는 구성 딕셔너리
        """
        self.web_search_agent = WebSearchAgent(config)
        self.config = config
    
    def process_web_search_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Tavily 웹 검색 결과를 처리하고 사용자 친화적인 응답을 반환합니다.
        
        Args:
            query: 사용자 쿼리 문자열
            chat_history: 컨텍스트를 위한 선택적 채팅 기록
            
        Returns:
            처리된 응답 문자열
            
        Raises:
            Exception: 처리에 실패한 경우
        """
        if not query or not query.strip():
            return "Error: Empty or invalid query provided."
        
        try:
            return self.web_search_agent.process_web_results(query, chat_history)
        except Exception as e:
            return f"Error processing web search results: {str(e)}"