from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """채팅 쿼리 요청 모델"""
    query: str = Field(..., description="사용자 질문")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="대화 히스토리")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    status: str = Field(..., description="응답 상태")
    response: str = Field(..., description="에이전트 응답")
    agent: str = Field(..., description="처리한 에이전트 이름")


class ValidationRequest(BaseModel):
    """검증 요청 모델"""
    validation_result: str = Field(..., description="검증 결과 (yes/no)")
    comments: Optional[str] = Field(None, description="검증 코멘트")


class ValidationResponse(BaseModel):
    """검증 응답 모델"""
    status: str = Field(..., description="검증 상태")
    message: str = Field(..., description="검증 메시지")
    response: str = Field(..., description="에이전트 응답")
    comments: Optional[str] = Field(None, description="검증 코멘트")


class ErrorResponse(BaseModel):
    """에러 응답 모델"""
    status: str = Field(default="error", description="에러 상태")
    agent: str = Field(default="System", description="에러 발생 에이전트")
    response: str = Field(..., description="에러 메시지") 