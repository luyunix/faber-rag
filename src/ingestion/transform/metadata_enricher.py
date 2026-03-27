"""元数据增强转换：基于规则 + 可选 LLM 增强。"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# 默认最大并行工作线程数用于 LLM 调用
DEFAULT_MAX_WORKERS = 5


class MetadataEnricher(BaseTransform):
    """用标题、摘要和标签丰富 chunk 元数据。
    
    处理流程：
        1. 基于规则的增强：从内容中提取基本元数据
        2. （可选）LLM 增强：生成语义丰富的元数据
        3. LLM 失败时：优雅降级到基于规则的元数据
    
    输出元数据：
        - title: chunk 的简短标题/主题
        - summary: 内容的简洁摘要
        - tags: 相关关键词/主题列表
        - enriched_by: "rule" 或 "llm"
    
    配置（通过 settings.yaml）：
        - ingestion.metadata_enricher.use_llm: bool - 启用 LLM 增强
        - ingestion.metadata_enricher.prompt_path: str - 自定义提示文件路径
    
    设计原则：
        - 优雅降级：LLM 错误不会阻止 ingestion
        - 原子处理：每个 chunk 独立处理
        - 可观测性：在元数据中记录 enriched_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """初始化 MetadataEnricher。
        
        参数：
            settings: 应用配置
            llm: 可选的 LLM 实例（用于测试；如果为 None 则自动创建）
            prompt_path: 可选的自定义提示文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/metadata_enrichment.txt"))
        
        # 确定是否应该使用 LLM
        enricher_config = {}
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            ingestion_config = settings.ingestion
            # 检查 ingestion 是否有 metadata_enricher 属性（dataclass）或 dict
            if hasattr(ingestion_config, 'metadata_enricher') and ingestion_config.metadata_enricher:
                enricher_config = ingestion_config.metadata_enricher
            elif isinstance(ingestion_config, dict):
                enricher_config = ingestion_config.get('metadata_enricher', {})
        
        self.use_llm = enricher_config.get('use_llm', False) if enricher_config else False
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for metadata enrichment")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """通过增强元数据转换 chunk。
        
        参数：
            chunks: 要增强的 chunk 列表
            trace: 可选的跟踪上下文
            
        返回：
            已增强的 chunk 列表（与输入长度相同）
        """
        if not chunks:
            return []
        
        # 进程 chunks in parallel if LLM is enabled
        if self.use_llm and self.llm:
            return self._transform_parallel(chunks, trace)
        else:
            return self._transform_sequential(chunks, trace)
    
    def _enrich_single_chunk(
        self, 
        chunk: Chunk, 
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """Enrich a single chunk. Thread-safe.
        
        Args:
            chunk: Chunk to enrich
            trace: Optional trace context
            
        Returns:
            Tuple of (enriched_chunk, enriched_by, error_message)
        """
        try:
            # 步骤 1: Rule-based enrichment
            rule_metadata = self._rule_based_enrich(chunk.text)
            
            # 步骤 2: LLM enhancement
            if self.use_llm and self.llm:
                llm_metadata = self._llm_enrich(chunk.text, trace)
                
                if llm_metadata:
                    enriched_metadata = llm_metadata
                    enriched_by = "llm"
                else:
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                    enriched_metadata['enrich_fallback_reason'] = "llm_failed"
            else:
                enriched_metadata = rule_metadata
                enriched_by = "rule"
            
            final_metadata = {
                **(chunk.metadata or {}),
                **enriched_metadata,
                'enriched_by': enriched_by
            }
            
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=final_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, enriched_by, None)
            
        except Exception as e:
            logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
            text_preview = ""
            if chunk.text:
                text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
            minimal_metadata = {
                **(chunk.metadata or {}),
                'title': 'Untitled',
                'summary': text_preview,
                'tags': [],
                'enriched_by': 'error',
                'enrich_error': str(e)
            }
            enriched_chunk = Chunk(
                id=chunk.id,
                text=chunk.text or "",
                metadata=minimal_metadata,
                source_ref=chunk.source_ref
            )
            return (enriched_chunk, "error", str(e))
    
    def _transform_parallel(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks in parallel using ThreadPoolExecutor."""
        max_workers = min(DEFAULT_MAX_WORKERS, len(chunks))
        enriched_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0
        
        logger.debug(f"Processing {len(chunks)} chunks in parallel (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._enrich_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_chunk, enriched_by, error = future.result()
                    enriched_chunks[idx] = enriched_chunk
                    
                    if enriched_by == "llm":
                        llm_enhanced_count += 1
                    elif enriched_by == "rule" and error is None:
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel enrichment: {e}")
                    enriched_chunks[idx] = chunks[idx]
        
        success_count = sum(1 for c in enriched_chunks if c is not None)
        
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": True,
                "max_workers": max_workers
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _transform_sequential(
        self, 
        chunks: List[Chunk], 
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks sequentially (fallback when LLM disabled)."""
        enriched_chunks = []
        success_count = 0
        llm_enhanced_count = 0
        fallback_count = 0
        
        for chunk in chunks:
            try:
                # 步骤 1: Rule-based enrichment (always performed)
                rule_metadata = self._rule_based_enrich(chunk.text)
                
                # 步骤 2: 可选 LLM enhancement
                if self.use_llm and self.llm:
                    llm_metadata = self._llm_enrich(chunk.text, trace)
                    
                    if llm_metadata:
                        # LLM success
                        enriched_metadata = llm_metadata
                        enriched_by = "llm"
                        llm_enhanced_count += 1
                    else:
                        # LLM failed, fallback to rule-based
                        enriched_metadata = rule_metadata
                        enriched_by = "rule"
                        fallback_count += 1
                        enriched_metadata['enrich_fallback_reason'] = "llm_failed"
                else:
                    # LLM disabled, use rule-based
                    enriched_metadata = rule_metadata
                    enriched_by = "rule"
                
                # Merge enriched metadata with existing metadata
                final_metadata = {
                    **(chunk.metadata or {}),
                    **enriched_metadata,
                    'enriched_by': enriched_by
                }
                
                # 创建 enriched chunk
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=final_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
                success_count += 1
                
            except Exception as e:
                # Atomic failure: log and preserve original with minimal metadata
                logger.error(f"Failed to enrich chunk {chunk.id}: {e}")
                # Handle 空 text case
                text_preview = ""
                if chunk.text:
                    text_preview = chunk.text[:100] + '...' if len(chunk.text) > 100 else chunk.text
                minimal_metadata = {
                    **(chunk.metadata or {}),
                    'title': 'Untitled',
                    'summary': text_preview,
                    'tags': [],
                    'enriched_by': 'error',
                    'enrich_error': str(e)
                }
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text or "",  # Ensure text is not None
                    metadata=minimal_metadata,
                    source_ref=chunk.source_ref
                )
                enriched_chunks.append(enriched_chunk)
        
        # 记录 trace
        if trace:
            trace.record_stage("metadata_enricher", {
                "total_chunks": len(chunks),
                "success_count": success_count,
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "parallel": False
            })
        
        logger.info(
            f"Enriched {success_count}/{len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, Fallback: {fallback_count})"
        )
        
        return enriched_chunks
    
    def _rule_based_enrich(self, text: str) -> Dict[str, Any]:
        """使用基于规则的启发式方法提取元数据。
        
        参数：
            text: Chunk 文本内容
            
        返回：
            包含 title、summary、tags 的字典
            
        异常：
            TypeError: 如果 text 为 None
        """
        if text is None:
            raise TypeError("Chunk text cannot be None")
        
        # Extract title from first heading or first line
        title = self._extract_title(text)
        
        # 从第一行生成 summary
        summary = self._extract_summary(text)
        
        # Extract tags from common patterns
        tags = self._extract_tags(text)
        
        return {
            'title': title,
            'summary': summary,
            'tags': tags
        }
    
    def _extract_title(self, text: str) -> str:
        """使用启发式方法从文本中提取标题。
        
        优先级：
            1. Markdown 标题（# Title）
            2. 第一行（如果足够短）
            3. 第一句
            4. 前 N 个字符
        """
        if not text:
            return "Untitled"
        
        # 检查 markdown 标题
        heading_match = re.match(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # 使用第一行（如果它很短且看起来像标题）
        first_line = text.split('\n')[0].strip()
        if first_line and len(first_line) <= 100 and not first_line.endswith(('.', ',', ';')):
            return first_line
        
        # 使用第一句（不带尾随标点）
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0]:
            title = sentences[0].strip()
            # 移除尾随标点（如果存在）
            title = re.sub(r'[.!?]+$', '', title)
            if len(title) <= 150:
                return title
            return title[:147] + "..."
        
        # 回退：前 100 个字符
        return text[:100].strip() + ("..." if len(text) > 100 else "")
    
    def _extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """使用前 N 句话从文本中提取摘要。
        
        参数：
            text: 源文本
            max_sentences: 要包含的最大句子数量
            
        返回：
            摘要文本
        """
        if not text:
            return ""
        
        # 分成句子
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 取前 N 句话
        summary_sentences = sentences[:max_sentences]
        summary = ' '.join(summary_sentences).strip()
        
        # 限制长度
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def _extract_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """使用关键词提取启发式方法提取标签。
        
        参数：
            text: 源文本
            max_tags: 要提取的最大标签数量
            
        返回：
            标签字符串列表
        """
        if not text:
            return []
        
        tags = set()
        
        # 提取大写单词（潜在专有名词）
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        tags.update(capitalized[:5])
        
        # 提取代码标识符（camelCase、snake_case）
        identifiers = re.findall(r'\b[a-z]+(?:[A-Z][a-z]*)+\b|\b[a-z]+_[a-z_]+\b', text)
        tags.update(identifiers[:5])
        
        # 提取 markdown 粗体/斜体术语（潜在关键词）
        markdown_keywords = re.findall(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_', text)
        for match in markdown_keywords[:5]:
            for group in match:
                if group:
                    tags.add(group.strip())
        
        # 转换为列表并限制数量
        tag_list = sorted(list(tags))[:max_tags]
        
        return tag_list
    
    def _llm_enrich(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[Dict[str, Any]]:
        """使用 LLM 增强元数据。
        
        参数：
            text: Chunk 文本内容
            trace: 可选的跟踪上下文
            
        返回：
            包含 title、summary、tags 的字典，失败时返回 None
        """
        if not self.llm:
            return None
        
        try:
            # 加载 prompt 模板
            prompt = self._load_prompt()
            
            # 构建带文本的 prompt
            formatted_prompt = prompt.replace("{chunk_text}", text[:2000])  # Limit text length
            
            # 调用 LLM
            messages = [Message(role="user", content=formatted_prompt)]
            response = self.llm.chat(messages)
            
            if not response:
                logger.warning("LLM returned empty response for metadata enrichment")
                return None
            
            # 从响应中提取文本（处理字符串和 ChatResponse 对象）
            response_text = response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            elif not isinstance(response, str):
                response_text = str(response)
            
            # 解析 LLM 响应
            metadata = self._parse_llm_response(response_text)
            
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": True,
                    "response_length": len(response_text)
                })
            
            return metadata
            
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")
            if trace:
                trace.record_stage("llm_enrich", {
                    "success": False,
                    "error": str(e)
                })
            return None
    
    def _load_prompt(self) -> str:
        """从文件加载提示模板。
        
        返回：
            提示模板字符串
            
        异常：
            FileNotFoundError: 如果提示文件不存在
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        prompt_path = Path(self._prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self._prompt_path}")
        
        self._prompt_template = prompt_path.read_text(encoding='utf-8')
        logger.info(f"Loaded metadata enrichment prompt from {self._prompt_path}")
        
        return self._prompt_template
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """将 LLM 响应解析为结构化元数据。
        
        期望格式：
            Title: <title>
            Summary: <summary>
            Tags: <tag1>, <tag2>, <tag3>
        
        参数：
            response: LLM 响应文本
            
        返回：
            包含 title、summary、tags 的字典
        """
        metadata = {
            'title': '',
            'summary': '',
            'tags': []
        }
        
        # 提取标题
        title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # 提取摘要
        summary_match = re.search(r'Summary:\s*(.+?)(?:\n(?:Tags:|$))', response, re.IGNORECASE | re.DOTALL)
        if summary_match:
            metadata['summary'] = summary_match.group(1).strip()
        
        # 提取标签
        tags_match = re.search(r'Tags:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # 按逗号分割并清理
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            metadata['tags'] = tags
        
        # 验证：确保非空值
        if not metadata['title']:
            metadata['title'] = 'Untitled'
        if not metadata['summary']:
            metadata['summary'] = response[:500]  # Fallback to raw response
        
        return metadata
