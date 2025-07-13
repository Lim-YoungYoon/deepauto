import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from .tavily_agent import TavilySearchAgent

load_dotenv()


class WebSearchAgent:
    """
    웹 검색 결과를 처리하고 적절한 LLM으로 라우팅하여 응답을 생성합니다.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        웹 검색 에이전트를 초기화합니다.
        
        Args:
            config: LLM 및 기타 설정을 포함하는 구성 딕셔너리
        """
        self.tavily_search_agent = TavilySearchAgent(config)
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        웹 검색을 위한 프롬프트를 구성합니다.
        
        Args:
            query: 사용자 쿼리
            chat_history: 채팅 기록
            
        Returns:
            완성된 프롬프트 문자열
        """
        # 제공된 경우 채팅 기록 추가
        # print("Chat History:", chat_history)
            
        # 프롬프트 구성
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def _build_llm_prompt(self, query: str, web_results: str) -> str:
        """
        웹 검색 결과를 처리하기 위한 LLM 프롬프트를 구성합니다.
        
        Args:
            query: 원본 사용자 쿼리
            web_results: 웹 검색 결과
            
        Returns:
            완성된 LLM 프롬프트 문자열
        """
        return (
            "You are an AI assistant specialized in information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        웹 검색 결과를 가져와서 LLM으로 처리하고 사용자 친화적인 응답을 반환합니다.
        
        Args:
            query: 사용자 쿼리 문자열
            chat_history: 컨텍스트를 위한 선택적 채팅 기록
            
        Returns:
            LLM에서 처리된 응답
            
        Raises:
            Exception: 처리에 실패한 경우
        """
        if not query or not query.strip():
            return "Error: Empty or invalid query provided."
        
        try:
            # 웹 검색 쿼리 프롬프트 구성
            web_search_query_prompt = self._build_prompt_for_web_search(
                query=query, 
                chat_history=chat_history
            )
            
            # 최적화된 검색 쿼리 생성
            web_search_query = self.llm.invoke(web_search_query_prompt)
            
            # 웹 검색 결과 검색
            web_results = self.tavily_search_agent.search(web_search_query.content)
            
            # 결과 처리용 LLM 프롬프트 구성
            llm_prompt = self._build_llm_prompt(query, web_results)
            
            # LLM으로 결과 처리
            response = self.llm.invoke(llm_prompt)
            
            return response
            
        except Exception as e:
            return f"Error processing web search results: {str(e)}"
