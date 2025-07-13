import requests
from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults


class TavilySearchTool:
    """
    Tavily API를 사용하여 웹 검색을 수행하는 도구입니다.
    """
    
    def __init__(self, max_results: int = 5) -> None:
        """
        Tavily 검색 도구를 초기화합니다.
        
        Args:
            max_results: 반환할 최대 검색 결과 수
        """
        self.max_results = max_results
        self.tavily_search = TavilySearchResults(max_results=max_results)

    def search_tavily(self, query: str) -> str:
        """
        Tavily API를 사용하여 일반적인 웹 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리 문자열
            
        Returns:
            포맷된 검색 결과 문자열
        """
        if not query or not query.strip():
            return "Error: Empty or invalid query provided."

        try:
            # 쿼리에서 주변 따옴표 제거
            cleaned_query = query.strip('"\'')
            
            search_docs = self.tavily_search.invoke(cleaned_query)
            
            if not search_docs:
                return "No relevant results found."
            
            return self._format_search_results(search_docs)
            
        except Exception as e:
            return f"Error retrieving web search results: {str(e)}"
    
    def _format_search_results(self, search_docs: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 읽기 쉬운 문자열로 포맷합니다.
        
        Args:
            search_docs: 검색 결과 딕셔너리 리스트
            
        Returns:
            포맷된 검색 결과 문자열
        """
        formatted_results = []
        
        for result in search_docs:
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            content = result.get("content", "No content")
            score = result.get("score", "No score")
            
            formatted_result = (
                f"title: {title} - "
                f"url: {url} - "
                f"content: {content} - "
                f"score: {score}"
            )
            formatted_results.append(formatted_result)
        
        final_result = "\n".join(formatted_results)
        return final_result