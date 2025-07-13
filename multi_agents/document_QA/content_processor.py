import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

class ContentProcessor:
    """
    파싱된 내용을 처리합니다 - 이미지를 요약하고, LLM 기반 의미론적 청크를 생성합니다
    """
    
    # 상수
    IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
    PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
    SPLIT_PATTERN = "\n#"
    CHUNK_START_TAG = "<|start_chunk_{}|>"
    CHUNK_END_TAG = "<|end_chunk_{}|>"
    MIN_CHUNK_WORDS = 256
    MAX_CHUNK_WORDS = 512
    
    def __init__(self, config):
        """
        내용 프로세서를 초기화합니다.
        
        Args:
            config: 모델 설정이 포함된 설정 객체
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        제공된 모델을 사용하여 이미지를 요약합니다 (오류 처리 포함).
        
        Args:
            images: 이미지 경로 리스트
            
        Returns:
            실패한 이미지에 대한 플레이스홀더가 포함된 이미지 요약 리스트
        """
        prompt_template = """Describe the image in detail while keeping it concise and to the point.
For context, the image may be part of a document used in a research paper, report, or technical documentation, particularly in fields such as social science, engineering, or scientific analysis.
Be specific about elements like tables, figures, plots (e.g., bar plots, line graphs), formulas, or diagrams if they are present in the image.
Only summarize what is explicitly shown in the image, without adding any interpretation or assumptions beyond the visible content.
If the image is not informative for document understanding or QA (e.g., contains only UI buttons, logos, or decorative elements), return 'non-informative'.
"""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()
        
        return self._process_images_with_chain(images, summary_chain)
    
    def _process_images_with_chain(self, images: List[str], summary_chain) -> List[str]:
        """오류 처리를 포함하여 요약 체인을 사용하여 이미지를 처리합니다."""
        results = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                self.logger.warning(f"이미지 처리 중 오류: {str(e)}")
                results.append("no image summary")
        
        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        파싱된 문서를 포맷하여 이미지 플레이스홀더를 이미지 요약으로 교체합니다.
        
        Args:
            parsed_document: doc_parser에서 파싱된 문서
            image_summaries: 이미지 요약 리스트
            
        Returns:
            이미지 요약이 포함된 포맷된 문서 텍스트
        """
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=self.PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=self.IMAGE_PLACEHOLDER
        )
        
        return self._replace_occurrences(
            formatted_parsed_document, 
            self.IMAGE_PLACEHOLDER, 
            image_summaries
        )
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        대상 플레이스홀더의 발생을 해당 교체물로 교체합니다.
        
        Args:
            text: 플레이스홀더가 포함된 텍스트
            target: 교체할 플레이스홀더
            replacements: 각 발생에 대한 교체물 리스트
            
        Returns:
            교체가 적용된 텍스트
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    result = result.replace(target, '', 1)
            else:
                # 더 이상 발생이 발견되지 않으면 루프를 중단합니다
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        문서를 의미론적 청크로 분할합니다.
        
        Args:
            formatted_document: 포맷된 문서 텍스트
            
        Returns:
            문서 청크 리스트
        """
        # 섹션 경계로 분할
        chunks = formatted_document.split(self.SPLIT_PATTERN)
        
        chunked_text = self._create_chunked_text(chunks)
        
        # LLM 기반 의미론적 청킹
        chunking_response = self._get_chunking_suggestions(chunked_text)
        
        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _create_chunked_text(self, chunks: List[str]) -> str:
        """마커가 있는 청크 텍스트를 생성합니다."""
        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # 청크에 #을 다시 추가합니다
            chunked_text += f"{self.CHUNK_START_TAG.format(i)}\n{chunk}\n{self.CHUNK_END_TAG.format(i)}\n"
        
        return chunked_text
    
    def _get_chunking_suggestions(self, chunked_text: str) -> str:
        """LLM에서 청킹 제안을 받습니다."""
        CHUNKING_PROMPT = """
        You are an assistant specialized in splitting text into semantically consistent sections. 
        
        Following is the document text:
        <document>
        {document_text}
        </document>
        
        <instructions>
        Instructions:
            1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
            2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
            3. Each chunk must be between 256 and 512 words.
            4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
            5. The chunks must be listed in ascending order.
            6. Provide your response in the form: 'split_after: 3, 5'.
        </instructions>
        
        Respond only with the IDs of the chunks where you believe a split should occur.
        YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
        """.strip()
        
        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        return self.chunker_model.invoke(formatted_chunking_prompt).content
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        LLM 제안 분할점에 따라 텍스트를 분할합니다.
        
        Args:
            chunked_text: 청크 마커가 있는 텍스트
            llm_response: 분할 제안이 있는 LLM 응답
            
        Returns:
            문서 청크 리스트
        """
        split_after = self._extract_split_points(llm_response)
        
        # 분할이 제안되지 않았다면 전체 텍스트를 하나의 섹션으로 반환합니다
        if not split_after:
            return [chunked_text]

        return self._create_sections_from_chunks(chunked_text, split_after)
    
    def _extract_split_points(self, llm_response: str) -> List[int]:
        """LLM 응답에서 분할점을 추출합니다."""
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 
        
        return split_after
    
    def _create_sections_from_chunks(self, chunked_text: str, split_after: List[int]) -> List[str]:
        """분할점을 기반으로 청크에서 섹션을 생성합니다."""
        # 텍스트에서 모든 청크 마커를 찾습니다
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # 분할점에 따라 청크를 그룹화합니다
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # 마지막 섹션이 비어있지 않다면 추가합니다
        if current_section:
            sections.append("".join(current_section).strip())

        return sections