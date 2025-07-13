import requests
from typing import Dict, Any, Optional

from .tavily_search import TavilySearchTool


class TavilySearchAgent:
    """
    Tavily를 사용하여 웹 소스에서 실시간 정보를 검색하는 에이전트입니다.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Tavily 검색 에이전트를 초기화합니다.
        
        Args:
            config: 필요한 설정이 포함된 설정 딕셔너리
        """
        self.tavily_search_tool = TavilySearchTool()
        self.config = config
    
    def search(self, query: str) -> str:
        """
        Tavily를 사용하여 웹 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리 문자열
            
        Returns:
            포맷된 검색 결과 문자열
            
        Raises:
            Exception: 검색이 실패한 경우
        """
        if not query or not query.strip():
            return "Error: Empty or invalid query provided."
        
        try:
            tavily_results = self.tavily_search_tool.search_tavily(query=query.strip())
            return f"Tavily Results:\n{tavily_results}\n"
        except Exception as e:
            return f"Error performing web search: {str(e)}" 