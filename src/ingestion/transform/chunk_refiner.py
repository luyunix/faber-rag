"""Chunk 优化转换：基于规则的清理 + 可选 LLM 增强。"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.settings import Settings, resolve_path
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.base_llm import BaseLLM, Message
from src.observability.logger import get_logger

logger = get_logger(__name__)

# 默认 max parallel workers for LLM calls
DEFAULT_MAX_WORKERS = 5


class ChunkRefiner(BaseTransform):
    """通过基于规则的清理和可选的 LLM 增强来优化 chunk。
    
    处理流程：
        1. 基于规则的优化：移除噪声（空白、页眉/页脚、HTML）
        2. （可选）LLM 优化：智能内容改进
        3. LLM 失败时：优雅降级到基于规则的结果
    
    配置（通过 settings.yaml）：
        - ingestion.chunk_refiner.use_llm: bool - 启用 LLM 增强
        - ingestion.chunk_refiner.prompt_path: str - 自定义提示文件路径
    
    设计原则：
        - 优雅降级：LLM 错误不会阻止 ingestion
        - 原子处理：每个 chunk 独立处理
        - 可观测性：在元数据中记录 refined_by
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseLLM] = None,
        prompt_path: Optional[str] = None
    ):
        """初始化 ChunkRefiner。
        
        参数：
            settings: 应用配置
            llm: 可选的 LLM 实例（用于测试；如果为 None 则自动创建）
            prompt_path: 可选的自定义提示文件路径
        """
        self.settings = settings
        self._llm = llm
        self._prompt_template: Optional[str] = None
        self._prompt_path = prompt_path or str(resolve_path("config/prompts/chunk_refinement.txt"))
        self._batch_prompt_path = str(resolve_path("config/prompts/chunk_refinement_batch.txt"))

        # 读取 chunk_refiner 配置（支持 dataclass 和 dict 两种格式）
        refiner_config = None
        if hasattr(settings, 'ingestion') and settings.ingestion is not None:
            refiner_config = getattr(settings.ingestion, 'chunk_refiner', None)

        if refiner_config is not None:
            self.use_llm = getattr(refiner_config, 'use_llm', False)
            self.batch_mode = getattr(refiner_config, 'batch_mode', True)
            self.batch_size = getattr(refiner_config, 'batch_size', 10)
        else:
            self.use_llm = False
            self.batch_mode = True
            self.batch_size = 10
        
    @property
    def llm(self) -> Optional[BaseLLM]:
        """延迟加载 LLM 实例。"""
        if self.use_llm and self._llm is None:
            try:
                self._llm = LLMFactory.create(self.settings)
                logger.info("LLM initialized for chunk refinement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}. Falling back to rule-based only.")
                self.use_llm = False
        return self._llm
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """通过优化 pipeline 转换 chunk。

        参数：
            chunks: 要优化的 chunk 列表
            trace: 可选的跟踪上下文

        返回：
            已优化的 chunk 列表（与输入长度相同）
        """
        if not chunks:
            return []

        # 步骤 1: 所有 chunks 先做规则精化
        rule_refined: List[Chunk] = []
        for chunk in chunks:
            rule_text = self._rule_based_refine(chunk.text)
            rule_refined.append(Chunk(
                id=chunk.id,
                text=rule_text,
                metadata={**(chunk.metadata or {}), 'refined_by': 'rule'},
                source_ref=chunk.source_ref
            ))

        # 步骤 2: 如果启用 LLM，根据 batch_mode 选择处理方式
        if self.use_llm and self.llm:
            if self.batch_mode:
                return self._transform_batch(rule_refined, trace)
            else:
                return self._transform_parallel(rule_refined, trace)

        return rule_refined

    def _transform_parallel(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """逐 chunk 并行调用 LLM（batch_mode=false）。"""
        import concurrent.futures

        max_workers = min(5, len(chunks))
        refined_chunks = [None] * len(chunks)
        llm_enhanced_count = 0
        fallback_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._refine_single_chunk, chunk, trace): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    refined_chunk, refined_by, _ = future.result()
                    refined_chunks[idx] = refined_chunk
                    if refined_by == "llm":
                        llm_enhanced_count += 1
                    elif refined_by == "rule":
                        fallback_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error in parallel refinement: {e}")
                    refined_chunks[idx] = chunks[idx]

        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "batch_mode": False,
                "parallel": True,
                "max_workers": max_workers,
            })

        logger.info(
            f"Refined {len(chunks)} chunks (batch_mode=false) "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )

        return refined_chunks

    def _refine_single_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> Tuple[Chunk, str, Optional[str]]:
        """优化单个 chunk。"""
        try:
            rule_refined_text = self._rule_based_refine(chunk.text)

            if self.use_llm and self.llm:
                llm_refined_text = self._llm_refine(rule_refined_text, trace)
                if llm_refined_text:
                    refined_text = llm_refined_text
                    refined_by = "llm"
                else:
                    refined_text = rule_refined_text
                    refined_by = "rule"
            else:
                refined_text = rule_refined_text
                refined_by = "rule"

            refined_chunk = Chunk(
                id=chunk.id,
                text=refined_text,
                metadata={
                    **(chunk.metadata or {}),
                    'refined_by': refined_by
                },
                source_ref=chunk.source_ref
            )
            return (refined_chunk, refined_by, None)

        except Exception as e:
            logger.error(f"Failed to refine chunk {chunk.id}: {e}")
            return (chunk, "error", str(e))

    def _transform_batch(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """使用批量 LLM 调用精化 chunks。"""
        llm_enhanced_count = 0
        fallback_count = 0
        results = {chunk.id: chunk for chunk in chunks}

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            try:
                llm_results = self._llm_refine_batch(batch, trace)
                for chunk_id, refined_text in llm_results.items():
                    if chunk_id in results:
                        results[chunk_id].text = refined_text
                        results[chunk_id].metadata['refined_by'] = 'llm'
                        llm_enhanced_count += 1
            except Exception as e:
                logger.warning(f"Batch LLM refinement failed for chunks {i}-{i + len(batch)}: {e}")
                fallback_count += len(batch)

        # 未被 LLM 处理的保持 rule-based
        for chunk in results.values():
            if chunk.metadata.get('refined_by') != 'llm':
                chunk.metadata['refined_by'] = 'rule'
                if chunk.id not in getattr(self, '_last_llm_results', {}):
                    fallback_count += 1

        if trace:
            trace.record_stage("chunk_refiner", {
                "total_chunks": len(chunks),
                "llm_enhanced_count": llm_enhanced_count,
                "fallback_count": fallback_count,
                "use_llm": self.use_llm,
                "batch_size": self.batch_size,
            })

        logger.info(
            f"Refined {len(chunks)} chunks "
            f"(LLM: {llm_enhanced_count}, fallback: {fallback_count})"
        )

        return list(results.values())

    def _llm_refine_batch(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> Dict[str, str]:
        """一次 LLM 调用精化一批 chunks。

        返回：
            {chunk_id: refined_text} 字典
        """
        prompt_template = self._load_prompt_batch()
        if not prompt_template:
            logger.warning("Batch prompt template not found, skipping LLM refinement")
            return {}

        # 构建 chunks 文本块
        chunks_text = ""
        for chunk in chunks:
            chunks_text += f"--- CHUNK_START:{chunk.id} ---\n{chunk.text}\n--- CHUNK_END:{chunk.id} ---\n\n"

        prompt = (
            prompt_template
            .replace("{chunk_count}", str(len(chunks)))
            .replace("{chunks}", chunks_text)
        )

        messages = [Message(role="user", content=prompt)]
        response = self.llm.chat(messages, trace=trace)

        response_text = response.content if hasattr(response, "content") else str(response)
        results = self._parse_batch_response(response_text)

        # 记录哪些 chunk 被成功处理（用于 fallback 计数）
        self._last_llm_results = set(results.keys())

        return results

    def _parse_batch_response(self, response: str) -> Dict[str, str]:
        """从批量 LLM 响应中解析每个 chunk 的精化结果。

        返回：
            {chunk_id: refined_text} 字典
        """
        results: Dict[str, str] = {}
        pattern = r"---\s*CHUNK_START:\s*([^\s]+)\s*---\n(.*?)\n---\s*CHUNK_END:\s*\1\s*---"
        for match in re.finditer(pattern, response, re.DOTALL):
            chunk_id = match.group(1).strip()
            content = match.group(2).strip()
            results[chunk_id] = content
        return results

    def _load_prompt_batch(self) -> Optional[str]:
        """加载批量精化 prompt 模板。"""
        try:
            prompt_path = Path(self._batch_prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Batch prompt file not found: {self._batch_prompt_path}")
                return None
            return prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to load batch prompt template: {e}")
            return None
    
    def _rule_based_refine(self, text: str) -> str:
        """应用基于规则的文本清理。
        
        清理操作：
            1. 移除页眉/页脚（分隔线 + 元数据）
            2. 移除 HTML 注释
            3. 移除 HTML 标签（保留内容）
            4. 规范化过多的空白
            5. 保留代码块和 Markdown 格式
        
        参数：
            text: 原始 chunk 文本
            
        返回：
            清理后的文本
        """
        if not text:
            return text
        
        # 如果只有空白则提前返回
        if not text.strip():
            return ""
        
        # 保留代码块（稍后提取并恢复）
        code_blocks = []
        code_block_pattern = r'```[\s\S]*?```'
        
        def extract_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        text = re.sub(code_block_pattern, extract_code_block, text)
        
        # 1. 移除带有页码/页脚的分隔线
        # 模式：────────────────
        # 后跟：Page XX, Footer text 等
        text = re.sub(
            r'─{10,}.*?(?:Page \d+|Footer|Section \d+|©|Confidential).*?─{10,}',
            '',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(r'─{10,}', '', text)  # 移除剩余的分隔线
        
        # 2. 移除 HTML 注释
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # 3. 移除 HTML 标签（但保留内容）
        text = re.sub(r'<[^>]+>', '', text)
        
        # 4. 规范化空白
        # - 将多个空格合并为单个空格
        text = re.sub(r' {2,}', ' ', text)
        
        # - 将 3 个或更多连续换行符合并为 2 个（保留段落断行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 5. 移除每行的前导/尾随空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 6. 恢复代码块
        for i, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", code_block)
        
        # 最终清理
        text = text.strip()
        
        return text
    
    def _llm_refine(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """应用基于 LLM 的智能优化。
        
        参数：
            text: 基于规则优化后的文本
            trace: 可选的跟踪上下文
            
        返回：
            LLM 优化的文本，失败时返回 None
        """
        if not text or not text.strip():
            return text
        
        try:
            # 加载 prompt 模板
            prompt_template = self._load_prompt()
            if not prompt_template:
                logger.warning("Prompt template not found, skipping LLM refinement")
                return None
            
            # 填充 prompt
            if '{text}' not in prompt_template:
                logger.error("Prompt template missing {text} placeholder")
                return None
            
            prompt = prompt_template.replace('{text}', text)
            
            # 使用消息对象调用 LLM
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace)
            
            # 从 ChatResponse 中提取文本
            if isinstance(response, str):
                refined_text = response
            else:
                # response 是 ChatResponse 对象
                refined_text = response.content
            
            if refined_text and refined_text.strip():
                return refined_text.strip()
            else:
                logger.warning("LLM returned empty result")
                return None
                
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    def _load_prompt(self) -> Optional[str]:
        """从文件加载提示模板。
        
        返回：
            提示模板字符串，如果文件不存在则返回 None
        """
        if self._prompt_template is not None:
            return self._prompt_template
        
        try:
            prompt_path = Path(self._prompt_path)
            if not prompt_path.exists():
                logger.warning(f"Prompt file not found: {self._prompt_path}")
                return None
            
            self._prompt_template = prompt_path.read_text(encoding='utf-8')
            logger.debug(f"Loaded prompt template from {self._prompt_path}")
            return self._prompt_template
            
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            return None
