"""
멀티 에이전트 문서 QA 챗봇을 위한 에이전트 결정 시스템

이 모듈은 LangGraph를 사용하여 다양한 에이전트의 오케스트레이션을 처리합니다.
사용자 쿼리를 내용과 컨텍스트에 따라 적절한 에이전트로 동적으로 라우팅합니다.
"""

import json
import time
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os
import getpass
from dotenv import load_dotenv
from multi_agents.document_QA import DocumentRAG
from multi_agents.tavily import TavilySearch
from multi_agents.tavily.web_search_agent import WebSearchAgent
from multi_agents.guardrail.local_guardrails import LocalGuardrails
from langgraph.checkpoint.memory import MemorySaver
from visualization.visualization_adapter import get_visualization_adapter
import cv2
import numpy as np
from config import Config

load_dotenv()

# 설정 로드
config = Config()

# 메모리 초기화
memory = MemorySaver()

# 시각화 어댑터 초기화
visualization_adapter = get_visualization_adapter()

# 스레드 지정
thread_config = {"configurable": {"thread_id": "1"}}


class AgentConfig:
    """에이전트 결정 시스템의 설정."""
    
    # 결정 모델
    DECISION_MODEL = "gpt-4.1"  # 또는 선호하는 모델
    
    # 이미지 분석을 위한 비전 모델
    VISION_MODEL = "gpt-4.1"
    
    # 응답에 대한 신뢰도 임계값
    CONFIDENCE_THRESHOLD = 0.85
    
    # 결정 에이전트를 위한 시스템 지시사항
    DECISION_SYSTEM_PROMPT = """You are an intelligent document QA routing system that routes user queries to the appropriate specialized agent.
    Your job is to analyze the user's request and determine which agent is best suited to handle it based on the query content, presence of uploaded PDFs, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT – For general chat, greetings, and questions that are not related to documents.
    2. RAG_AGENT – For answering questions based on uploaded PDF files or existing document knowledge base. This agent handles both new PDF uploads and queries against existing documents.
    3. WEB_SEARCH_AGENT – For questions that require up-to-date information beyond the contents of uploaded documents, such as current events or recent developments.

    Make your decision based on these guidelines:
    - If the user has uploaded a PDF file, always route the query to the RAG_AGENT (which will process the PDF and answer questions).
    - If no PDF is uploaded:
        - Use the RAG_AGENT for questions about existing documents in the knowledge base.
        - Use the WEB_SEARCH_AGENT for questions about recent events, new developments, or time-sensitive topics.
        - Use the CONVERSATION_AGENT for greetings, small talk, or general non-document-related queries.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """


class AgentState(MessagesState):
    """워크플로우에서 유지되는 상태."""
    agent_name: Optional[str]  # 현재 활성 에이전트
    current_input: Optional[Union[str, Dict]]  # 처리할 입력
    has_pdf: bool  # 현재 입력에 PDF가 포함되어 있는지 여부
    pdf_path: Optional[str]  # PDF가 있는 경우 경로
    output: Optional[str]  # 사용자에게 최종 출력
    needs_human_validation: bool  # 인간 검증이 필요한지 여부
    retrieval_confidence: float  # 검색에 대한 신뢰도 (RAG 에이전트용)
    bypass_routing: bool  # 가드레일을 위한 에이전트 라우팅 우회 플래그
    insufficient_info: bool  # RAG 응답에 불충분한 정보가 있음을 나타내는 플래그


class AgentDecision(TypedDict):
    """결정 에이전트의 출력 구조."""
    agent: str
    reasoning: str
    confidence: float


class InputProcessor:
    """입력 분석 및 전처리를 처리합니다."""
    
    def __init__(self, guardrails):
        self.guardrails = guardrails
    
    def extract_text_from_input(self, current_input: Union[str, Dict]) -> str:
        """다양한 입력 형식에서 텍스트 내용을 추출합니다."""
        if isinstance(current_input, str):
            return current_input
        elif isinstance(current_input, dict):
            return current_input.get("text", "")
        return ""
    
    def check_pdf_presence(self, current_input: Union[str, Dict]) -> tuple[bool, Optional[str]]:
        """입력에 PDF가 있는지 확인하고 경로를 추출합니다."""
        if isinstance(current_input, dict) and "pdf" in current_input:
            return True, current_input.get("pdf", None)
        return False, None
    
    def apply_input_guardrails(self, input_text: str) -> tuple[bool, Optional[str]]:
        """입력 가드레일을 적용하고 입력이 허용되는지 반환합니다."""
        if not input_text:
            return True, None
        
        is_allowed, message = self.guardrails.check_input(input_text)
        return is_allowed, message if not is_allowed else None


class ConversationContextBuilder:
    """메시지 히스토리에서 대화 컨텍스트를 구축합니다."""
    
    @staticmethod
    def build_recent_context(messages: List[BaseMessage], limit: int = None) -> str:
        """최근 대화 메시지에서 컨텍스트를 구축합니다."""
        if limit:
            messages = messages[-limit:]
        
        context_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)


class AgentExecutor:
    """다양한 에이전트 유형의 실행을 처리합니다."""
    
    def __init__(self, config):
        self.config = config
        self.rag_agent = DocumentRAG(config)
        self.web_search_agent = WebSearchAgent(config)
    
    def execute_conversation_agent(self, state: AgentState) -> AgentState:
        """일반 대화를 처리합니다."""
        print(f"Selected agent: CONVERSATION_AGENT")
        
        # 시각화를 위한 에이전트 활동 기록
        start_time = time.time()
        
        messages = state["messages"]
        current_input = state["current_input"]
        
        # 입력 텍스트 추출
        input_text = InputProcessor(None).extract_text_from_input(current_input)
        
        # 대화 컨텍스트 구축
        recent_context = ConversationContextBuilder.build_recent_context(messages)
        
        # 대화 프롬프트 생성
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Document QA Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and document-related queries. You must respond naturally and clearly, helping users understand document content when available and guiding them appropriately when not.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining a professional and helpful tone.
        - Answer **questions based on uploaded PDF documents**.
        - Route **questions about recent or time-sensitive topics** to web search if needed.
        - Handle **follow-up questions** using prior conversation context.

        ### Guidelines for Responding:

        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly and engaging manner.
        - Keep responses **concise and conversational**, unless further explanation is needed.

        2. **Document-Based Questions:**
        - If the user has uploaded a PDF and is asking about its content, answer clearly using information from the document.
        - If no PDF is uploaded and the question requires specific document content, ask the user to upload the relevant file.

        3. **Follow-Up & Clarifications:**
        - Maintain awareness of previous messages to improve continuity.
        - If the user's query is unclear, ask **clarifying questions** before answering.

        4. **Handling Uncertainty or Limits:**
        - If a question cannot be answered due to lack of document data or current knowledge, offer to search online or explain the limitation.
        - For questions about current events or newly published information, suggest using the web search tool.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when appropriate.
        - If information is drawn from external sources, briefly mention it (e.g., "According to recent reports...").
        - Avoid making unsupported claims—be transparent about the scope of your knowledge.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"  
        **You:** "I'm doing great and ready to help! What can I assist you with today?"

        **User:** "I uploaded a PDF. Can you explain section 2?"  
        **You:** "Sure! Based on the uploaded document, section 2 talks about how deep learning models are used to classify document topics. Would you like a summary?"

        Conversational LLM Response:"""

        response = self.config.conversation.llm.invoke(conversation_prompt)
        
        # 시각화를 위한 대화 에이전트 활동 기록
        processing_time = time.time() - start_time
        response_text = response.content if hasattr(response, 'content') else str(response)
        visualization_adapter.record_conversation_activity(
            query=input_text,
            response=response_text,
            processing_time=processing_time
        )
        
        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def execute_rag_agent(self, state: AgentState) -> AgentState:
        """PDF 처리 및 신뢰도 기반 라우팅을 사용하여 RAG로 문서 지식 쿼리를 처리합니다."""
        print(f"Selected agent: RAG_AGENT")
        
        # 시각화를 위한 에이전트 활동 기록
        start_time = time.time()
        
        messages = state["messages"]
        query = state["current_input"]
        pdf_path = state.get("pdf_path")
        rag_context_limit = self.config.rag.context_limit

        # 최근 컨텍스트 구축
        recent_context = ConversationContextBuilder.build_recent_context(
            messages, rag_context_limit
        )

        # 업로드된 PDF가 있으면 처리
        if pdf_path:
            print(f"Processing uploaded PDF: {pdf_path}")
            self.rag_agent.ingest_file(pdf_path)
        
        # RAG로 쿼리 처리
        response = self.rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # 응답 내용 확인
        response_content = response["response"]
        response_text = self._extract_response_text(response_content)
        
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        # 불충분한 정보 확인
        insufficient_info = self._check_insufficient_info(response_text)
        print(f"Insufficient info flag set to: {insufficient_info}")

        # 신뢰도와 정보 충분성에 따른 출력 결정
        if (retrieval_confidence >= self.config.rag.min_retrieval_confidence and 
            not insufficient_info):
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")

        # 시각화를 위한 RAG 에이전트 활동 기록
        processing_time = time.time() - start_time
        visualization_adapter.record_rag_agent_activity(
            query=query,
            response=response_text,
            retrieval_confidence=retrieval_confidence,
            processing_time=processing_time
        )

        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }
    
    def execute_web_search_agent(self, state: AgentState) -> AgentState:
        """웹 검색 결과를 처리하고, LLM으로 처리하여 정제된 응답을 생성합니다."""
        print(f"Selected agent: WEB_SEARCH_AGENT")
        print("[WEB_SEARCH_AGENT] Processing Web Search Results...")
        
        # 시각화를 위한 에이전트 활동 기록
        start_time = time.time()
        
        messages = state["messages"]
        web_search_context_limit = self.config.web_search.context_limit

        recent_context = ConversationContextBuilder.build_recent_context(
            messages, web_search_context_limit
        )

        processed_response = self.web_search_agent.process_web_results(
            query=state["current_input"], 
            chat_history=recent_context
        )
        
        # 시각화를 위한 웹 검색 에이전트 활동 기록
        processing_time = time.time() - start_time
        response_text = processed_response.content if hasattr(processed_response, 'content') else str(processed_response)
        visualization_adapter.record_web_search_activity(
            query=state["current_input"],
            search_results=[response_text],  # 시각화를 위해 단순화
            processing_time=processing_time
        )
        
        # 관련된 에이전트 결정
        if state['agent_name'] is not None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_AGENT"
        else:
            involved_agents = "WEB_SEARCH_AGENT"

        return {
            **state,
            "output": processed_response,
            "agent_name": involved_agents
        }
    
    def _extract_response_text(self, response_content: Any) -> str:
        """응답 객체에서 텍스트 내용을 추출합니다."""
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            return response_content.content
        return response_content
    
    def _check_insufficient_info(self, response_text: str) -> bool:
        """응답이 불충분한 정보를 나타내는지 확인합니다."""
        if not isinstance(response_text, str):
            return False
        
        insufficient_indicators = [
            "I don't have enough information to answer this question based on the provided context",
            "I don't have enough information",
            "don't have enough information",
            "not enough information",
            "insufficient information",
            "cannot answer",
            "unable to answer"
        ]
        
        response_lower = response_text.lower()
        return any(indicator.lower() in response_lower for indicator in insufficient_indicators)


class GuardrailsProcessor:
    """입력 및 출력 가드레일 처리를 처리합니다."""
    
    def __init__(self, guardrails):
        self.guardrails = guardrails
    
    # def process_validation_input(self, state: AgentState) -> AgentState:
    #     """인간 검증 입력을 처리합니다."""
    #     output = state["output"]
    #     current_input = state["current_input"]
    #     
    #     if not output or not isinstance(output, (str, AIMessage)):
    #         return state

    #     output_text = output if isinstance(output, str) else output.content
        
    #     # 인간 검증 메시지 처리
    #     if "Human Validation Required" in output_text:
    #         validation_input = InputProcessor(None).extract_text_from_input(current_input)
    #             
    #         if validation_input.lower().startswith(('yes', 'no')):
    #             validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
    #                 
    #         if validation_input.lower().startswith('no'):
    #             fallback_message = AIMessage(
    #                 content="The previous document analysis requires further review. A healthcare professional has flagged potential inaccuracies."
    #             )
    #             return {
    #                 **state,
    #                 "messages": [validation_response, fallback_message],
    #                 "output": fallback_message
    #             }
    #             
    #         return {
    #                 **state,
    #                 "messages": validation_response
    #             }
    #     
    #     return state
    
    def apply_output_sanitization(self, state: AgentState) -> AgentState:
        """가드레일을 사용하여 출력 정제를 적용합니다."""
        output = state["output"]
        current_input = state["current_input"]

        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        input_text = InputProcessor(None).extract_text_from_input(current_input)
        
        # 출력 정제 적용
        sanitized_output = self.guardrails.check_output(output_text, input_text)
        print(sanitized_output)
        
        # 정제된 메시지 생성
        sanitized_message = (AIMessage(content=sanitized_output) 
                           if isinstance(output, AIMessage) else sanitized_output)
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }


def create_agent_graph():
    """에이전트 오케스트레이션을 위한 LangGraph를 생성하고 구성합니다."""
    
    # 컴포넌트 초기화
    guardrails = LocalGuardrails(config.rag.llm)
    input_processor = InputProcessor(guardrails)
    agent_executor = AgentExecutor(config)
    guardrails_processor = GuardrailsProcessor(guardrails)
    
    # 결정 모델과 파서 초기화
    decision_model = config.agent_decision.llm
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # 결정 프롬프트와 체인 생성
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    decision_chain = decision_prompt | decision_model | json_parser
    
    # 그래프 상태 변환 정의
    def analyze_input(state: AgentState) -> AgentState:
        """PDF를 감지하고 입력 유형을 결정하기 위해 입력을 분석합니다."""
        current_input = state["current_input"]
        
        # 텍스트 추출 및 가드레일 확인
        input_text = input_processor.extract_text_from_input(current_input)
        if input_text:
            is_allowed, message = input_processor.apply_input_guardrails(input_text)
            if not is_allowed:
                print(f"Selected agent: INPUT GUARDRAILS, Message: {message}")
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_pdf": False,
                    "pdf_path": None,
                    "bypass_routing": True
                }
        
        # PDF 존재 확인
        has_pdf, pdf_path = input_processor.check_pdf_presence(current_input)
        if has_pdf:
            print("ANALYZED PDF PATH: ", pdf_path)
        
        return {
            **state,
            "has_pdf": has_pdf,
            "pdf_path": pdf_path,
            "bypass_routing": False
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """가드레일로 인해 일반 라우팅을 우회해야 하는지 확인합니다."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """쿼리를 처리할 에이전트를 결정합니다."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_pdf = state["has_pdf"]
        pdf_path = state["pdf_path"]
        
        # 결정 모델을 위한 입력 준비
        input_text = input_processor.extract_text_from_input(current_input)
        
        # 최근 대화 히스토리에서 컨텍스트 생성
        recent_context = ConversationContextBuilder.build_recent_context(messages[-6:])
        
        # 결정 입력을 위해 모든 것을 결합
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has PDF: {has_pdf}
        PDF path: {pdf_path if has_pdf else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # PDF가 있으면 RAG_AGENT로 라우팅
        if has_pdf:
            updated_state = {**state, "agent_name": "RAG_AGENT"}
            return {"agent_state": updated_state, "next": "RAG_AGENT"}
        
        # 다른 경우에는 LLM 결정 사용
        decision = decision_chain.invoke({"input": decision_input})
        print(f"Decision: {decision['agent']}")
        updated_state = {**state, "agent_name": decision["agent"]}
        
        # 시각화를 위한 결정 에이전트 활동 기록
        visualization_adapter.record_decision_agent_activity(
            query=input_text,
            selected_agent=decision["agent"],
            confidence=decision.get("confidence", 0.0),
            reasoning=decision.get("reasoning", "")
        )
        
        # if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
        #     return {"agent_state": updated_state, "next": "needs_validation"}
        return {"agent_state": updated_state, "next": decision["agent"]}

    def confidence_based_routing(state: AgentState) -> str:
        """RAG 신뢰도 점수와 응답 내용에 따라 라우팅합니다."""
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_AGENT"
        return "apply_guardrails"  # 검증 대신 가드레일로 직접 이동
    
    # def handle_human_validation(state: AgentState) -> Dict:
    #     """필요한 경우 인간 검증을 준비합니다."""
    #     if state.get("needs_human_validation", False):
    #         return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
    #     return {"agent_state": state, "next": END}
    
    # def perform_human_validation(state: AgentState) -> AgentState:
    #     """인간 검증 프로세스를 처리합니다."""
    #     print(f"Selected agent: HUMAN_VALIDATION")

    #     validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

    #     validation_message = AIMessage(content=validation_prompt)

    #     return {
    #         **state,
    #         "output": validation_message,
    #         "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
    #     }

    def apply_output_guardrails(state: AgentState) -> AgentState:
        """생성된 응답에 출력 가드레일을 적용합니다."""
        # 먼저 검증 입력 처리 (주석 처리됨)
        # state = guardrails_processor.process_validation_input(state)
        
        # 그 다음 출력 정제 적용
        return guardrails_processor.apply_output_sanitization(state)
    
    # 워크플로우 그래프 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", agent_executor.execute_conversation_agent)
    workflow.add_node("RAG_AGENT", agent_executor.execute_rag_agent)
    workflow.add_node("WEB_SEARCH_AGENT", agent_executor.execute_web_search_agent)
    # workflow.add_node("check_validation", handle_human_validation)
    # workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # 엣지 정의
    workflow.set_entry_point("analyze_input")
    
    # 가드레일 우회를 위한 조건부 라우팅 추가
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # 결정 라우터를 에이전트에 연결
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_AGENT": "WEB_SEARCH_AGENT",
            # "needs_validation": "RAG_AGENT"  # 검증 라우팅 주석 처리됨
        }
    )
    
    # 에이전트 출력을 가드레일로 직접 연결 (검증 우회)
    workflow.add_edge("CONVERSATION_AGENT", "apply_guardrails")
    workflow.add_edge("WEB_SEARCH_AGENT", "apply_guardrails")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)

    # workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    # workflow.add_conditional_edges(
    #     "check_validation",
    #     lambda x: x["next"],
    #     {
    #         "human_validation": "human_validation",
    #         END: "apply_guardrails"
    #     }
    # )
    
    # 그래프 컴파일
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """기본값으로 에이전트 상태를 초기화합니다."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_pdf": False,
        "pdf_path": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> str:
    """
    에이전트 결정 시스템을 통해 사용자 쿼리를 처리합니다.
    
    Args:
        query: 사용자 입력 (텍스트 문자열 또는 텍스트와 이미지가 포함된 딕셔너리)
        conversation_history: 이전 메시지의 선택적 목록, 상태가 대화 히스토리를 저장하므로 더 이상 필요하지 않음
        
    Returns:
        적절한 에이전트의 응답
    """
    # 아직 시작되지 않았다면 시각화 모니터링 시작
    if not visualization_adapter.is_monitoring:
        visualization_adapter.start_monitoring()
    
    # 그래프 초기화
    graph = create_agent_graph()
    
    # 상태 초기화
    state = init_agent_state()
    
    # 현재 쿼리 추가
    state["current_input"] = query

    # 이미지 업로드 케이스 처리
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded a document"
    
    state["messages"] = [HumanMessage(content=query)]

    # 그래프 실행
    result = graph.invoke(state, thread_config)

    # 히스토리를 합리적인 크기로 유지
    if len(result["messages"]) > config.max_conversation_history:
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # 콘솔에서 대화 히스토리 시각화
    for m in result["messages"]:
        m.pretty_print()
    
    print(result["messages"])
    
    return result