"""多模态组装器，用于组合文本和图片内容。

本模块通过以下步骤组装多模态 MCP 响应：
- 检测 chunk 元数据中的图片引用
- 读取图片文件并编码为 base64
- 构建符合 MCP 规范的 ImageContent 块

设计原则：
- 懒加载：仅当显式请求时才加载图片
- 错误弹性：缺失的图片不会破坏文本响应
- 格式检测：从文件内容自动推断 MIME 类型
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.types import RetrievalResult


logger = logging.getLogger(__name__)


# 支持的图片格式及其 MIME 类型
MIME_TYPE_MAP: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

# 用于格式检测的魔数（当扩展名不可靠时作为后备）
MAGIC_BYTES: Dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP 以 RIFF 开头
    b"BM": "image/bmp",
}

# 文本中的图片占位符模式：[IMAGE: image_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class ImageReference:
    """块内图片的引用。

    属性：
        image_id: 图片的唯一标识符
        file_path: 图片文件的文件系统路径
        page: 源文档中的可选页码
        text_offset: 占位符在块文本中的字符偏移量
        text_length: 占位符字符串的长度
        caption: ImageCaptioner 生成的可选说明
    """
    image_id: str
    file_path: Optional[str] = None
    page: Optional[int] = None
    text_offset: Optional[int] = None
    text_length: Optional[int] = None
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "caption": self.caption,
        }


@dataclass
class ImageContent:
    """已加载准备好用于 MCP 响应的图片内容。

    属性：
        image_id: 图片的唯一标识符
        data: Base64 编码的图片数据
        mime_type: MIME 类型（例如 "image/png"）
        caption: 图片的可选说明
    """
    image_id: str
    data: str  # base64 编码
    mime_type: str
    caption: Optional[str] = None

    def to_mcp_content(self) -> types.ImageContent:
        """转换为 MCP ImageContent 块。

        返回：
            用于协议响应的 MCP ImageContent 对象。
        """
        return types.ImageContent(
            type="image",
            data=self.data,
            mimeType=self.mime_type,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "data": self.data,
            "mime_type": self.mime_type,
            "caption": self.caption,
        }


class MultimodalAssembler:
    """从检索结果组装多模态内容。

    本类从块元数据中提取图片引用，
    加载图片文件，并构建结合文本和图片的符合 MCP 规范的内容块。

    示例：
        >>> assembler = MultimodalAssembler()
        >>> results = [RetrievalResult(chunk_id="doc1_001", ...)]
        >>> content_blocks = assembler.assemble(results)
        >>> # 返回 TextContent 和 ImageContent 块列表

    参数：
        image_storage: 用于路径查找的可选 ImageStorage 实例。
            为 None 时，直接使用元数据中的文件路径。
        max_images_per_result: 每个结果包含的最大图片数量（默认：5）。
        include_captions: 是否将图片说明作为文本包含（默认：True）。
    """

    def __init__(
        self,
        image_storage: Optional[Any] = None,
        max_images_per_result: int = 5,
        include_captions: bool = True,
    ) -> None:
        """初始化 MultimodalAssembler。

        参数：
            image_storage: 用于解析图片路径的可选 ImageStorage。
            max_images_per_result: 每个检索结果包含的最大图片数量。
            include_captions: 是否添加说明文本块。
        """
        self._image_storage = image_storage
        self.max_images_per_result = max_images_per_result
        self.include_captions = include_captions

    def extract_image_refs(
        self,
        result: RetrievalResult,
    ) -> List[ImageReference]:
        """从检索结果中提取图片引用。

        查找位置：
        1. metadata.images 列表（结构化图片信息）
        2. 文本中的 [IMAGE: id] 占位符（后备）

        参数：
            result: 包含块数据的 RetrievalResult。

        返回：
            在结果中找到的 ImageReference 对象列表。
        """
        refs: List[ImageReference] = []
        metadata = result.metadata or {}

        # 主要来源：元数据中的结构化图片列表
        images_list = metadata.get("images", [])
        if isinstance(images_list, list):
            for img_info in images_list[:self.max_images_per_result]:
                if isinstance(img_info, dict) and "id" in img_info:
                    ref = ImageReference(
                        image_id=img_info["id"],
                        file_path=img_info.get("path"),
                        page=img_info.get("page"),
                        text_offset=img_info.get("text_offset"),
                        text_length=img_info.get("text_length"),
                    )
                    refs.append(ref)

        # 添加说明（如果可用）
        captions = metadata.get("image_captions", {})
        if isinstance(captions, dict):
            for ref in refs:
                if ref.image_id in captions:
                    ref.caption = captions[ref.image_id]

        # 后备：如果无结构化引用则从文本解析占位符
        if not refs and result.text:
            placeholders = IMAGE_PLACEHOLDER_PATTERN.findall(result.text)
            for image_id in placeholders[:self.max_images_per_result]:
                image_id = image_id.strip()
                ref = ImageReference(
                    image_id=image_id,
                    caption=captions.get(image_id) if isinstance(captions, dict) else None,
                )
                refs.append(ref)

        return refs

    def resolve_image_path(
        self,
        ref: ImageReference,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """解析图片引用的文件系统路径。

        参数：
            ref: 要解析的 ImageReference。
            collection: 用于路径构建的可选集合名称。

        返回：
            如果找到则为绝对文件路径，否则为 None。
        """
        # 如果可用则使用显式路径
        if ref.file_path:
            path = Path(ref.file_path)
            if path.exists():
                return str(path.resolve())

        # 尝试 ImageStorage 查找        if self._image_storage is not None:
            try:
                path = self._image_storage.get_image_path(ref.image_id)
                if path and Path(path).exists():
                    return path
            except Exception as e:
                logger.warning(f"ImageStorage 查找 {ref.image_id} 失败: {e}")

        # 基于约定的路径：data/images/{collection}/{image_id}.png
        if collection:
            from src.core.settings import resolve_path
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = resolve_path(f"data/images/{collection}/{ref.image_id}{ext}")
                if candidate.exists():
                    return str(candidate.resolve())

        return None

    def load_image(
        self,
        file_path: str,
    ) -> Optional[ImageContent]:
        """加载并编码图片文件。

        参数：
            file_path: 图片文件的路径。

        返回：
            包含 base64 数据和 MIME 类型的 ImageContent，失败则为 None。
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"图片文件未找到: {file_path}")
                return None

            # 读取文件内容
            data = path.read_bytes()
            if not data:
                logger.warning(f"图片文件为空: {file_path}")
                return None

            # 检测 MIME 类型            mime_type = self._detect_mime_type(path, data)

            # 编码为 base64            base64_data = base64.b64encode(data).decode("utf-8")

            return ImageContent(
                image_id=path.stem,
                data=base64_data,
                mime_type=mime_type,
            )

        except Exception as e:
            logger.error(f"加载图片 {file_path} 失败: {e}")
            return None

    def _detect_mime_type(
        self,
        path: Path,
        data: bytes,
    ) -> str:
        """从文件扩展名或魔数检测 MIME 类型。

        参数：
            path: 用于扩展名检测的文件路径。
            data: 用于魔数字节检测的文件内容。

        返回：
            MIME 类型字符串（未知时默认为 "image/png"）。
        """
        # 首先尝试扩展名
        suffix = path.suffix.lower()
        if suffix in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[suffix]

        # 后备到魔数字节
        for magic, mime_type in MAGIC_BYTES.items():
            if data.startswith(magic):
                return mime_type

        # 默认为 PNG        logger.debug(f"未知的图片格式 {path}，默认为 image/png")
        return "image/png"

    def assemble_for_result(
        self,
        result: RetrievalResult,
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为单个结果组装多模态内容块。

        参数：
            result: 要处理的 RetrievalResult。
            collection: 用于路径解析的可选集合名称。

        返回：
            MCP 内容块列表（TextContent 和 ImageContent）。
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = []

        # 提取图片引用
        refs = self.extract_image_refs(result)

        # 加载并添加图片
        for ref in refs:
            # 解析路径
            file_path = self.resolve_image_path(ref, collection)
            if not file_path:
                logger.debug(f"无法解析图片路径: {ref.image_id}")
                continue

            # 加载图片
            image_content = self.load_image(file_path)
            if image_content is None:
                continue

            # 使用引用信息更新
            image_content.image_id = ref.image_id
            image_content.caption = ref.caption

            # 添加图片块
            blocks.append(image_content.to_mcp_content())

            # 如果启用则添加说明作为文本
            if self.include_captions and ref.caption:
                caption_text = f"**图片说明 ({ref.image_id}):** {ref.caption}"
                blocks.append(types.TextContent(type="text", text=caption_text))

        return blocks

    def assemble(
        self,
        results: List[RetrievalResult],
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为多个结果组装多模态内容块。

        参数：
            results: 要处理的 RetrievalResult 列表。
            collection: 用于路径解析的可选集合名称。

        返回：
            来自所有结果的所有 MCP 内容块列表。
        """
        all_blocks: List[Union[types.TextContent, types.ImageContent]] = []
        seen_image_ids: set = set()

        for result in results:
            blocks = self.assemble_for_result(result, collection)

            # 跨结果去重图片
            for block in blocks:
                if isinstance(block, types.ImageContent):
                    # 检查是否已见过此图片
                    # ImageContent 没有 image_id，所以我们哈希数据                    data_hash = hash(block.data[:100])  # 使用前缀以提高效率
                    if data_hash in seen_image_ids:
                        continue
                    seen_image_ids.add(data_hash)

                all_blocks.append(block)

        return all_blocks

    def has_images(self, result: RetrievalResult) -> bool:
        """检查结果是否包含图片引用。

        参数：
            result: 要检查的 RetrievalResult。

        返回：
            如果结果有图片引用则返回 True，否则返回 False。
        """
        refs = self.extract_image_refs(result)
        return len(refs) > 0

    def count_images(self, results: List[RetrievalResult]) -> int:
        """统计所有结果中的图片总数。

        参数：
            results: 要统计的 RetrievalResult 列表。

        返回：
            找到的图片引用总数。
        """
        total = 0
        for result in results:
            refs = self.extract_image_refs(result)
            total += len(refs)
        return total
