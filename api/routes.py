import os
import shutil
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from config import Config
from models.request_models import QueryRequest, ChatResponse, ErrorResponse
from services.chat_service import ChatService
from utils.file_utils import FileUtils

# 로거 설정
logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter()

# 설정 및 서비스 초기화
config = Config()
chat_service = ChatService()
file_utils = FileUtils(
    allowed_extensions=config.file.allowed_extensions,
    max_size_mb=config.api.max_pdf_upload_size
)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 HTML 페이지 제공"""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/health")
def health_check():
    """Docker 헬스 체크용 엔드포인트"""
    return {"status": "healthy"}


@router.post("/chat")
def chat(
    request: QueryRequest, 
    response: Response, 
    session_id: Optional[str] = Cookie(None)
):
    """멀티 에이전트 문서 QA 시스템을 통한 사용자 텍스트 쿼리 처리"""
    try:
        return chat_service.process_chat_query(request, response, session_id)
    except ValueError as e:
        logger.warning(f"채팅 엔드포인트에서 잘못된 요청: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"채팅 엔드포인트에서 예상치 못한 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="요청 처리 중 내부 서버 오류가 발생했습니다")


@router.post("/upload")
async def upload_pdf(
    response: Response,
    pdf: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """문서 업로드 및 텍스트 입력 처리"""
    # 파일 타입 검증
    if not file_utils.is_allowed_file(pdf.filename):
        logger.warning(f"지원하지 않는 파일 타입 시도: {pdf.filename}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                response="지원하지 않는 문서 형식입니다. 허용된 형식: PDF"
            ).dict()
        )
    
    # 파일 저장 경로 생성
    filename = file_utils.generate_secure_filename(pdf.filename)
    file_path = os.path.join(config.file.pdf_folder, filename)
    
    try:
        # 파일 저장
        if not await file_utils.save_uploaded_file(pdf, file_path):
                    return JSONResponse(
            status_code=413,
            content=ErrorResponse(
                response=f"문서가 너무 큽니다. 최대 허용 크기: {config.api.max_pdf_upload_size}MB"
            ).dict()
        )
        
        # 쿼리 처리
        return chat_service.process_pdf_query(file_path, text, response, session_id)
        
    except ValueError as e:
        logger.warning(f"업로드 엔드포인트에서 잘못된 요청: {str(e)}")
        file_utils.cleanup_temp_file(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"업로드 엔드포인트에서 예상치 못한 오류: {str(e)}", exc_info=True)
        file_utils.cleanup_temp_file(file_path)
        raise HTTPException(status_code=500, detail="문서 처리 중 내부 서버 오류가 발생했습니다")
    finally:
        # 임시 파일 정리
        file_utils.cleanup_temp_file(file_path)


# @router.post("/validate")
# def validate_document_output(
#     response: Response,
#     validation_result: str = Form(...), 
#     comments: Optional[str] = Form(None),
#     session_id: Optional[str] = Cookie(None)
# ):
#     """문서 QA 출력에 대한 인간 검증 처리"""
#     try:
#         return chat_service.process_validation(validation_result, comments, response, session_id)
#     except Exception as e:
#         logger.error(f"문서 QA 검증 엔드포인트에서 오류: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear_data")
async def clear_data():
    """데이터 디렉토리 정리 - data와 uploads 폴더의 모든 하위 폴더와 파일 삭제"""
    logger.info("clear_data 엔드포인트 호출됨")
    
    try:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, '..', 'data')
        temp_dir = os.path.join(base_dir, '..', 'temp')
        
        logger.info(f"data_dir 경로: {data_dir}")
        logger.info(f"temp_dir 경로: {temp_dir}")
        logger.info(f"data_dir 존재: {os.path.exists(data_dir)}")
        logger.info(f"temp_dir 존재: {os.path.exists(temp_dir)}")
        
        deleted_items = []
        
        # data 폴더 정리
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        deleted_items.append(f"data/{item}")
                        logger.info(f"파일 삭제됨: data/{item}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        deleted_items.append(f"data/{item}")
                        logger.info(f"디렉토리 삭제됨: data/{item}")
                except Exception as e:
                    logger.error(f"data 항목 삭제 실패 {item}: {str(e)}")
                    continue
        else:
            logger.info("data 디렉토리가 존재하지 않습니다.")
        
        # temp 폴더 정리
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        deleted_items.append(f"temp/{item}")
                        logger.info(f"파일 삭제됨: temp/{item}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        deleted_items.append(f"temp/{item}")
                        logger.info(f"디렉토리 삭제됨: temp/{item}")
                except Exception as e:
                    logger.error(f"temp 항목 삭제 실패 {item}: {str(e)}")
                    continue
        else:
            logger.info("temp 디렉토리가 존재하지 않습니다.")
        
        logger.info("모든 데이터 정리가 완료되었습니다.")
        return JSONResponse(content={
            "success": True, 
            "message": f"총 {len(deleted_items)}개의 항목이 성공적으로 삭제되었습니다.",
            "deleted_items": deleted_items
        })
        
    except Exception as e:
        logger.error(f"데이터 정리 중 오류: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": f"데이터 정리 중 오류가 발생했습니다: {str(e)}"
            }
        ) 