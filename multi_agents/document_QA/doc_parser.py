import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    RapidOcrOptions, 
    smolvlm_picture_description
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

class PDFParser:
    """
    docling을 사용하여 문서 파싱을 처리합니다.
    """
    
    def __init__(self):
        """PDF 파서를 초기화합니다."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("문서 파서가 초기화되었습니다!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
            image_resolution_scale: float = 2.0,
            do_ocr: bool = True,
            do_tables: bool = True,
            do_formulas: bool = True,
            do_picture_desc: bool = False
        ) -> Tuple[Any, List[str]]:
        """
        문서를 파싱하고 구조화된 내용과 이미지를 추출합니다.
        
        Args:
            document_path: 파싱할 문서 경로
            output_dir: 추출된 이미지를 저장할 디렉터리
            image_resolution_scale: 추출된 이미지의 해상도 스케일
            do_ocr: OCR 처리 활성화
            do_tables: 테이블 구조 추출 활성화
            do_formulas: 수식 보강 활성화
            do_picture_desc: 이미지 설명 생성 활성화
            
        Returns:
            (파싱된_문서, 이미지_경로_리스트)를 포함한 튜플
        """
        try:
            # 입력 검증
            self._validate_inputs(document_path, output_dir)
            
            # 출력 디렉터리 생성
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # 변환 구성 및 실행
            conversion_res = self._convert_document(
                document_path, 
                image_resolution_scale, 
                do_ocr, 
                do_tables, 
                do_formulas, 
                do_picture_desc
            )
            
            # 문서 파일명 추출
            doc_filename = conversion_res.input.file.stem
            
            # 이미지 저장
            self._save_page_images(conversion_res, output_dir_path, doc_filename)
            image_paths = self._save_element_images(conversion_res, output_dir_path, doc_filename)
            
            # 요약을 위한 이미지 추출
            images = self._extract_summarization_images(conversion_res)
            
            self.logger.info(f"문서 변환 성공. 페이지: {len(conversion_res.document.pages)}")
            
            return conversion_res.document, images
            
        except Exception as e:
            self.logger.error(f"문서 파싱 중 오류 {document_path}: {e}")
            raise
    
    def _validate_inputs(self, document_path: str, output_dir: str) -> None:
        """입력 매개변수를 검증합니다."""
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"문서를 찾을 수 없습니다: {document_path}")
        
        if not document_path.lower().endswith('.pdf'):
            raise ValueError(f"지원하지 않는 파일 형식입니다. PDF가 필요하지만, 받은 파일: {document_path}")
    
    def _convert_document(
        self, 
        document_path: str, 
        image_resolution_scale: float, 
        do_ocr: bool, 
        do_tables: bool, 
        do_formulas: bool, 
        do_picture_desc: bool
    ) -> Any:
        """docling을 사용하여 문서를 변환합니다."""
        # 파이프라인 옵션 구성
        pipeline_options = self._create_pipeline_options(
            image_resolution_scale, do_ocr, do_tables, do_formulas, do_picture_desc
        )
        
        # 문서 변환기 초기화
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        # 문서 변환
        return converter.convert(document_path)
    
    def _create_pipeline_options(
        self, 
        image_resolution_scale: float, 
        do_ocr: bool, 
        do_tables: bool, 
        do_formulas: bool, 
        do_picture_desc: bool
    ) -> PdfPipelineOptions:
        """문서 변환을 위한 파이프라인 옵션을 생성합니다."""
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc
        )
        
        # 테이블 구조 모드 설정
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        return pipeline_options
    
    def _save_page_images(self, conversion_res: Any, output_dir_path: Path, doc_filename: str) -> None:
        """변환된 문서에서 페이지 이미지를 저장합니다."""
        for page_no, page in conversion_res.document.pages.items():
            try:
                page_image_filename = output_dir_path / f"{doc_filename}-{page_no}.png"
                with page_image_filename.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")
            except Exception as e:
                self.logger.warning(f"페이지 {page_no}의 페이지 이미지 저장 중 오류: {e}")
    
    def _save_element_images(self, conversion_res: Any, output_dir_path: Path, doc_filename: str) -> List[str]:
        """그림과 테이블의 이미지를 저장합니다."""
        table_counter = 0
        picture_counter = 0
        image_paths = []
        
        for element, _level in conversion_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = output_dir_path / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                    
            if isinstance(element, PictureItem):
                picture_path = f"{doc_filename}-picture-{picture_counter}.png"
                element_image_filename = output_dir_path / picture_path
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                
                # 이미지 리스트에 경로 추가
                image_paths.append(str(element_image_filename))
                picture_counter += 1
        
        return image_paths
    
    def _extract_summarization_images(self, conversion_res: Any) -> List[str]:
        """요약을 위한 이미지를 추출합니다."""
        images = []
        for picture in conversion_res.document.pictures:
            ref = picture.get_ref().cref
            image = picture.image
            if image:
                images.append(str(image.uri))
        
        return images