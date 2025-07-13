import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from models.request_models import ErrorResponse

logger = logging.getLogger(__name__)


async def request_entity_too_large_handler(request: Request, exc: Exception):
    """문서 크기 초과 예외 처리"""
    return JSONResponse(
        status_code=413,
        content=ErrorResponse(
            response="문서가 너무 큽니다. 최대 허용 크기를 초과했습니다."
        ).dict()
    )


async def validation_exception_handler(request: Request, exc: Exception):
    """검증 예외 처리"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            response=f"입력 데이터 검증 실패: {str(exc)}"
        ).dict()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    logger.error(f"예상치 못한 오류 발생: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            response="내부 서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        ).dict()
    ) 