import os
import uuid
import time
import logging
from typing import Optional
from werkzeug.utils import secure_filename
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class FileUtils:
    """문서 처리 관련 유틸리티 클래스"""
    
    def __init__(self, allowed_extensions: set, max_size_mb: int):
        self.allowed_extensions = allowed_extensions
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def is_allowed_file(self, filename: str) -> bool:
        """문서 확장자가 허용되는지 확인"""
        if not filename or '.' not in filename:
            return False
        return filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def validate_file_size(self, file_content: bytes) -> bool:
        """문서 크기가 허용 범위 내인지 확인"""
        return len(file_content) <= self.max_size_bytes
    
    def generate_secure_filename(self, original_filename: str) -> str:
        """안전한 문서명 생성"""
        return secure_filename(f"{uuid.uuid4()}_{original_filename}")
    
    def cleanup_temp_file(self, file_path: str, max_retries: int = 3) -> bool:
        """임시 문서를 안전하게 삭제"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"임시 문서 삭제 성공: {file_path}")
                    return True
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"임시 문서 삭제 실패 ({max_retries}회 시도): {file_path}, 오류: {str(e)}")
                    return False
                else:
                    time.sleep(0.1)
        return True
    
    async def save_uploaded_file(self, file: UploadFile, save_path: str) -> bool:
        """업로드된 문서를 저장"""
        try:
            file_content = await file.read()
            
            # 문서 크기 검증
            if not self.validate_file_size(file_content):
                logger.warning(f"문서가 너무 큽니다: {file.filename}, 크기: {len(file_content)} bytes")
                return False
            
            # 문서 저장
            with open(save_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"문서 업로드 성공: {file.filename}")
            return True
            
        except Exception as e:
            logger.error(f"문서 저장 중 오류: {str(e)}")
            return False 