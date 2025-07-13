import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import Config
from api.routes import router
from visualization.visualization_routes import visualization_router
from api.middleware import (
    request_entity_too_large_handler,
    validation_exception_handler,
    general_exception_handler
)

# 설정 로드
config = Config()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="DeepAuto Document QA System",
    version="1.0.0",
    description="자율 문서 QA를 위한 멀티 에이전트 시스템"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# 라우터 등록
app.include_router(router)
app.include_router(visualization_router)

# 예외 핸들러 등록
@app.exception_handler(413)
async def handle_request_entity_too_large(request: Request, exc: Exception):
    return await request_entity_too_large_handler(request, exc)

@app.exception_handler(422)
async def handle_validation_exception(request: Request, exc: Exception):
    return await validation_exception_handler(request, exc)

@app.exception_handler(Exception)
async def handle_general_exception(request: Request, exc: Exception):
    return await general_exception_handler(request, exc)

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    logger.info("Multi-Agent Document QA System 서버가 시작되었습니다.")
    logger.info(f"서버 주소: http://{config.api.host}:{config.api.port}")

# 애플리케이션 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    logger.info("Multi-Agent Document QA System 서버가 종료되었습니다.")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=config.api.host, 
        port=config.api.port,
        log_level="info"
    )

