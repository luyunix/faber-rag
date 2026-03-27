"""文档分块模块 - 用于业务层的文本切分适配器。

本模块作为 libs.splitter（纯文本切分）与 Ingestion Pipeline（业务对象转换）
之间的适配层。它将 Document 对象转换为 Chunk 对象，并进行适当的 ID 生成、
元数据继承和可追溯性处理。

核心增值服务（相对于 libs.splitter）：
1. 分块 ID 生成：为每个分块生成确定性和唯一的 ID
2. 元数据继承：将 Document 元数据传播到所有分块
3. chunk_index：记录在文档中的顺序位置
4. source_ref：建立父子可追溯关系
5. 类型转换：str → Chunk 对象（core.types 契约）

设计原则：
- 适配器模式：桥接文本切分工具与业务对象
- 配置驱动：使用 SplitterFactory 进行基于配置的策略选择
- 确定性：同一 Document 重复切分产生相同的 Chunk ID
- 类型安全：强制执行 core.types.Chunk 契约
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """将文档转换为带业务级增强的分块。

    此类包装文本切分器（来自 libs）并添加业务逻辑：
    - 生成稳定的分块 ID
    - 继承和扩展元数据
    - 维护文档可追溯性

    属性：
        _splitter: 来自 libs 层的基础文本切分器
        _settings: 分块行为的配置设置

    示例：
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="长文档内容...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"生成了 {len(chunks)} 个分块")
        >>> print(f"第一个分块 ID: {chunks[0].id}")
        >>> print(f"第一个分块索引: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """使用配置初始化 DocumentChunker。

        参数：
            settings: 包含切分器配置的设置对象。
                     切分器配置应位于 settings.splitter.*

        异常：
            ValueError: 如果切分器配置无效或提供商未知
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
    
    def split_document(self, document: Document) -> List[Chunk]:
        """将文档分块并进行完整的业务增强。

        这是协调转换的主要入口点：
        1. 使用底层切分器获取文本片段
        2. 为每个分块生成确定性 ID
        3. 继承和扩展文档的元数据
        4. 创建符合 core.types 契约的 Chunk 对象

        参数：
            document: 要切分为分块的源文档

        返回：
            Chunk 对象列表，具有以下特点：
            - 唯一、确定性的 ID
            - 继承的元数据 + chunk_index + source_ref
            - 正确的类型契约 (core.types.Chunk)

        异常：
            ValueError: 如果文档没有文本或结构无效

        示例：
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="第一节内容。\\n\\n第二节内容。",
            ...     metadata={"source_path": "file.pdf", "title": "报告"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"文档 {document.id} 没有可切分的文本内容")

        # 步骤 1：使用底层切分器获取文本片段
        text_fragments = self._splitter.split_text(document.text)

        if not text_fragments:
            raise ValueError(
                f"切分器未返回文档 {document.id} 的分块。"
                f"文本长度：{len(document.text)}"
            )

        # 步骤 2：将文本片段转换为带增强的 分块 对象
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成唯一且确定性的分块 ID。

        ID 格式：{doc_id}_{index:04d}_{content_hash}
        - doc_id：父文档标识符
        - index：顺序位置（零填充至 4 位数字）
        - content_hash：文本 SHA256 哈希的前 8 个字符

        这确保了：
        - 唯一性：doc_id + index + content_hash 的组合
        - 确定性：相同输入总是产生相同 ID
        - 可读性：人类可读的格式结构

        参数：
            doc_id：父文档 ID
            index：分块的顺序位置（从 0 开始）
            text：分块文本内容

        返回：
            唯一的分块 ID 字符串

        示例：
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # 计算内容哈希以确保唯一性
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

        # 格式：{doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """从文档继承元数据并添加分块特定字段。

        这会创建一个新的元数据字典，包含：
        - 来自 document.metadata 的所有字段（复制，非引用）
        - chunk_index：顺序位置（从 0 开始）
        - source_ref：父文档 ID 的引用
        - image_refs：此分块中引用的图片 ID 列表（从占位符中提取）

        注意：文档级别的 'images' 字段被故意从分块元数据中排除，
        因为它会是冗余的。取而代之的是，基于分块文本中找到的
        [IMAGE: xxx] 占位符来填充分块特定的 'image_refs'。

        参数：
            document：要继承其元数据的源文档
            chunk_index：此分块的顺序位置
            chunk_text：此分块的文本内容（用于提取 image_refs）

        返回：
            包含继承和分块特定字段的元数据字典

        示例：
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="内容",
            ...     metadata={"source_path": "file.pdf", "title": "报告"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "参见 [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re

        # 复制所有文档元数据（对于基本类型浅拷贝就足够了）
        chunk_metadata = document.metadata.copy()

        # 获取文档级别的图片列表用于查找
        doc_images = document.metadata.get("images", [])

        # 移除文档级别的 'images' 字段——我们将在下面添加分块特定的图片
        chunk_metadata.pop("images", None)

        # 添加分块特定字段
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id

        # 通过查找 [IMAGE: xxx] 占位符来从分块文本中提取 image_refs
        image_refs = []
        if chunk_text:
            # 匹配 [IMAGE: image_id] 占位符的模式
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]

        chunk_metadata["image_refs"] = image_refs

        # 构建分块特定的 'images' 列表，包含被引用图片的完整元数据
        # ImageCaptioner 需要这些来访问 Vision API 调用的图片路径
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])

        if chunk_images:
            chunk_metadata["images"] = chunk_images

        # 尝试从第一个引用的图片确定页码
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")

        return chunk_metadata
