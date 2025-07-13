import logging
import uuid
from typing import Dict, Any, Optional
from fastapi import Response, Cookie

from models.request_models import QueryRequest, ChatResponse, ErrorResponse
from multi_agents.decision import process_query

logger = logging.getLogger(__name__)


class ChatService:
    """채팅 관련 비즈니스 로직을 처리하는 서비스 클래스"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_chat_query(
        self, 
        request: QueryRequest, 
        response: Response, 
        session_id: Optional[str] = Cookie(None)
    ) -> ChatResponse:
        """채팅 쿼리를 처리합니다."""
        # 세션 ID 생성
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"채팅 쿼리 처리 중: {request.query[:100]}...")
            
            # 에이전트 시스템을 통해 쿼리 처리
            response_data = process_query(request.query)
            response_text = response_data['messages'][-1].content
            
            # 세션 쿠키 설정
            response.set_cookie(key="session_id", value=session_id)
            
            result = ChatResponse(
                status="success",
                response=response_text,
                agent=response_data["agent_name"]
            )
            
            self.logger.info(f"채팅 쿼리 처리 완료 - 에이전트: {response_data['agent_name']}")
            return result
            
        except ValueError as e:
            self.logger.warning(f"채팅 엔드포인트에서 잘못된 요청: {str(e)}")
            raise ValueError(f"잘못된 요청: {str(e)}")
        except Exception as e:
            self.logger.error(f"채팅 엔드포인트에서 예상치 못한 오류: {str(e)}", exc_info=True)
            raise Exception("요청 처리 중 내부 서버 오류가 발생했습니다")
    
    def process_pdf_query(
        self,
        file_path: str,
        text: str,
        response: Response,
        session_id: Optional[str] = Cookie(None)
    ) -> ChatResponse:
        """문서 쿼리를 처리합니다."""
        # 세션 ID 생성
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            self.logger.info(f"문서 쿼리 처리 중: {text[:100] if text else 'Document only'}...")
            
            # 에이전트 시스템을 통해 쿼리 처리
            query = {"text": text, "pdf": file_path}
            response_data = process_query(query)
            response_text = response_data['messages'][-1].content
            
            # 세션 쿠키 설정
            response.set_cookie(key="session_id", value=session_id)
            
            result = ChatResponse(
                status="success",
                response=response_text,
                agent=response_data["agent_name"]
            )
            
            self.logger.info(f"문서 처리 완료 - 에이전트: {response_data['agent_name']}")
            return result
            
        except ValueError as e:
            self.logger.warning(f"PDF 엔드포인트에서 잘못된 요청: {str(e)}")
            raise ValueError(f"잘못된 요청: {str(e)}")
        except Exception as e:
            self.logger.error(f"문서 엔드포인트에서 예상치 못한 오류: {str(e)}", exc_info=True)
            raise Exception("문서 처리 중 내부 서버 오류가 발생했습니다")
    
    # def process_validation(
    #     self,
    #     validation_result: str,
    #     comments: Optional[str],
    #     response: Response,
    #     session_id: Optional[str] = Cookie(None)
    # ) -> Dict[str, Any]:
    #     """문서 QA 검증 요청을 처리합니다."""
    #     # 세션 ID 생성
    #     if not session_id:
    #         session_id = str(uuid.uuid4())
    #     
    #     try:
    #         # 세션 쿠키 설정
    #         response.set_cookie(key="session_id", value=session_id)
    #         
    #         # 검증 입력으로 에이전트 결정 시스템 재실행
    #         validation_query = f"Validation result: {validation_result}"
    #         if comments:
    #             validation_query += f" Comments: {comments}"
    #         
    #         response_data = process_query(validation_query)
    #         
    #         if validation_result.lower() == 'yes':
    #             return {
    #                 "status": "validated",
    #                 "message": "**Output confirmed by human validator:**",
    #                 "response": response_data['messages'][-1].content
    #             }
    #         else:
    #             return {
    #                 "status": "rejected",
    #                 "comments": comments,
    #                 "message": "**Output requires further review:**",
    #                 "response": response_data['messages'][-1].content
    #             }
    #             
    #     except Exception as e:
    #         self.logger.error(f"문서 QA 검증 처리 중 오류: {str(e)}", exc_info=True)
    #         raise Exception(f"문서 QA 검증 처리 중 오류가 발생했습니다: {str(e)}") 