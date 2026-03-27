"""用于为 chunk 添加图像描述的 Image Captioner 转换。

性能优化：
1. 仅处理 chunk 文本中实际引用的图像（通过 [IMAGE: id] 占位符）
2. 使用描述缓存避免对相同图像的重复 Vision API 调用
3. 完全跳过没有图像引用的 chunk
4. 对唯一图像进行并行处理，线程安全的缓存
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# 用于查找图像占位符的正则表达式：[IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# 默认最大并行工作线程数用于 Vision API 调用
DEFAULT_MAX_WORKERS = 3  # 比文本 LLM 低，因为成本/延迟更高


class ImageCaptioner(BaseTransform):
    """使用 Vision LLM 为 chunk 中引用的图像生成描述。
    
    此转换识别包含图像引用的 chunk，使用 Vision LLM
    生成描述性标题，并用这些标题丰富 chunk 文本/元数据，
    以改善视觉内容的检索。
    
    主要特性：
    - 仅处理 chunk 文本中实际引用的图像（而非元数据中的所有图像）
    - 缓存描述以避免重复的 Vision API 调用
    - 线程安全的描述缓存，支持未来的并行化
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        # 描述缓存：image_id -> caption 字符串（带锁的线程安全）
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        
        # 检查 vision LLM 是否在 settings 中启用
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
            try:
                self.llm = llm or LLMFactory.create_vision_llm(settings)
            except Exception as e:
                logger.error(f"Failed to initialize Vision LLM: {e}")
                # We don't raise here to allow pipeline to continue without captioning
                # effectively falling back to no-op for this transform
        else:
            logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """从配置加载图像描述提示。"""
        # 假设标准相对路径。在生产环境中，逻辑可能更健壮。
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """提取 chunk 文本中实际引用的图像 ID。
        
        参数：
            text: Chunk 文本内容
            
        返回：
            在 [IMAGE: id] 占位符中找到的图像 ID 列表
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """获取图像的描述，如果可用则使用缓存。线程安全。
        
        参数：
            img_id: 图像标识符
            img_path: 图像文件路径
            trace: 可选的跟踪上下文
            
        返回：
            描述字符串，失败时返回 None
        """
        # 首先检查缓存（线程安全读取）
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]
        
        # Validate path
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content
            
            # 缓存结果（线程安全写入）
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """处理 chunk 并为引用的图像添加描述。
        
        仅处理通过 [IMAGE: id] 占位符实际引用在 chunk 文本中的图像。
        使用缓存避免重复的 API 调用。
        对唯一图像进行并行处理。
        """
        if not self.llm:
            return chunks
        
        # 从所有 chunk 的 metadata 构建 image lookup
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta
        
        logger.info(f"Found {len(image_lookup)} unique images in document")
        
        # Clear cache for new document processing
        with self._cache_lock:
            self._caption_cache.clear()
        
        # 首次遍历：收集所有需要生成描述的唯一图像 ID
        images_to_caption: Dict[str, str] = {}  # img_id -> img_path
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")
        
        # 并行生成所有唯一图像的描述
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)
        
        # 第二次遍历：将描述应用到 chunk
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # 从缓存获取描述（已由并行处理填充）
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """并行生成多个图像的描述。
        
        参数：
            images_to_caption: img_id -> img_path 字典
            trace: 可选的跟踪上下文
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
