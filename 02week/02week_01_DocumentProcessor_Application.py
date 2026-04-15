#!/usr/bin/env python3
"""
RAG文档处理器 - 支持多格式文档加载和智能分块
适配本地Ollama + qwen2.5:14b-custom
修复LangChain新版本兼容性问题
"""

import hashlib
import os
from typing import List, Optional
from pathlib import Path

# LangChain导入 - 修复新版本路径问题
try:
    # LangChain 0.3+ 新版本
    from langchain_core.documents import Document
    print("✅ 使用 langchain_core.documents")
except ImportError:
    try:
        # LangChain 0.2 版本
        from langchain.schema import Document
        print("✅ 使用 langchain.schema")
    except ImportError:
        # 旧版本
        from langchain.docstore.document import Document
        print("✅ 使用 langchain.docstore.document")

# 文档加载器
try:
    from langchain_community.document_loaders import (
        PyMuPDFLoader,
        TextLoader,
        CSVLoader,
    )
    print("✅ 使用 langchain_community.document_loaders")
except ImportError:
    try:
        from langchain.document_loaders import (
            PyMuPDFLoader,
            TextLoader,
            CSVLoader,
        )
        print("✅ 使用 langchain.document_loaders")
    except ImportError:
        print("⚠️ 请安装: pip install langchain-community pymupdf")

# 文本分块器
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )
    print("✅ 使用 langchain_text_splitters")
except ImportError:
    try:
        from langchain.text_splitter import (
            RecursiveCharacterTextSplitter,
            MarkdownHeaderTextSplitter,
        )
        print("✅ 使用 langchain.text_splitter")
    except ImportError:
        print("⚠️ 请安装: pip install langchain-text-splitters")

# ===================== 配置 =====================
CHUNK_SIZE = 1000          # 每个块的最大字符数
CHUNK_OVERLAP = 200        # 块之间的重叠字符数

SUPPORTED_FORMATS = {
    '.pdf': 'pdf',
    '.txt': 'txt',
    '.csv': 'csv',
    '.md': 'md',
    '.markdown': 'md'
}


class DocumentProcessor:
    """企业级文档处理器 - 支持多格式、智能分块、元数据管理"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 基础分块器（用于普通文本）
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # Markdown标题分块器
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )
    
    def load_document(self, file_path: str, file_type: str = None) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            file_type: 文件类型（可选，默认从扩展名识别）
        
        Returns:
            List[Document]: 文档列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 识别文件类型
        if file_type is None:
            ext = Path(file_path).suffix.lower()
            if ext not in SUPPORTED_FORMATS:
                raise ValueError(f"不支持的文件类型: {ext}，支持: {list(SUPPORTED_FORMATS.keys())}")
            file_type = SUPPORTED_FORMATS[ext]
        
        # 选择加载器
        loaders = {
            'pdf': PyMuPDFLoader(file_path),
            'txt': TextLoader(file_path, encoding='utf-8'),
            'csv': CSVLoader(file_path, encoding='utf-8'),
            'md': TextLoader(file_path, encoding='utf-8'),
        }
        
        print(f"📄 正在加载文档: {file_path} (类型: {file_type})")
        
        try:
            docs = loaders[file_type].load()
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            raise
        
        # 添加元数据
        for idx, doc in enumerate(docs):
            doc.metadata['source'] = os.path.basename(file_path)
            doc.metadata['source_path'] = file_path
            doc.metadata['file_type'] = file_type
            doc.metadata['page'] = doc.metadata.get('page', idx)
            # 生成文档唯一ID
            unique_str = f"{file_path}_{doc.metadata.get('page', idx)}"
            doc.metadata['doc_id'] = hashlib.md5(unique_str.encode()).hexdigest()[:16]
        
        print(f"  ✅ 加载完成，共 {len(docs)} 页/段")
        return docs
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        批量加载目录下的所有文档
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归加载子目录
        
        Returns:
            List[Document]: 所有文档的列表
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"目录不存在: {directory_path}")
        
        all_docs = []
        pattern = "**/*" if recursive else "*"
        
        for ext in SUPPORTED_FORMATS.keys():
            for file_path in Path(directory_path).glob(f"{pattern}{ext}"):
                try:
                    docs = self.load_document(str(file_path))
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"  ⚠️ 加载失败 {file_path}: {e}")
        
        print(f"\n📊 批量加载完成，共加载 {len(all_docs)} 个文档片段")
        return all_docs
    
    def smart_chunk(self, documents: List[Document]) -> List[Document]:
        """
        智能分块
        
        Args:
            documents: 原始文档列表
        
        Returns:
            List[Document]: 分块后的文档列表
        """
        all_chunks = []
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'txt')
            
            # Markdown文档使用标题结构分块
            if file_type == 'md':
                try:
                    md_chunks = self.md_splitter.split_text(doc.page_content)
                    for chunk in md_chunks:
                        # 继承原始元数据
                        chunk.metadata.update(doc.metadata)
                        chunk.metadata['chunk_method'] = 'markdown_header'
                    all_chunks.extend(md_chunks)
                except Exception as e:
                    print(f"  ⚠️ Markdown分块失败，使用基础分块: {e}")
                    chunks = self.base_splitter.split_documents([doc])
                    for chunk in chunks:
                        chunk.metadata['chunk_method'] = 'recursive_fallback'
                    all_chunks.extend(chunks)
            else:
                # 普通文档使用递归分块
                chunks = self.base_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata['chunk_method'] = 'recursive'
                all_chunks.extend(chunks)
        
        # 为每个块添加索引和哈希
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata['chunk_index'] = idx
            chunk.metadata['total_chunks'] = len(all_chunks)
            # 生成内容哈希用于去重
            chunk.metadata['chunk_hash'] = hashlib.md5(
                chunk.page_content.encode('utf-8')
            ).hexdigest()[:16]
        
        print(f"✂️  分块完成: {len(documents)} 个文档 → {len(all_chunks)} 个块")
        if all_chunks:
            avg_size = sum(len(c.page_content) for c in all_chunks) // len(all_chunks)
            print(f"   平均块大小: {avg_size} 字符")
        
        return all_chunks
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        一站式处理：加载 + 分块
        
        Args:
            file_path: 文件路径
        
        Returns:
            List[Document]: 处理后的文档块
        """
        print(f"\n{'='*50}")
        print(f"开始处理文档: {file_path}")
        print(f"{'='*50}")
        
        docs = self.load_document(file_path)
        chunks = self.smart_chunk(docs)
        
        print(f"\n✅ 处理完成！")
        print(f"   原始文档: {len(docs)} 页/段")
        print(f"   分块数量: {len(chunks)}")
        if chunks:
            print(f"   总字符数: {sum(len(c.page_content) for c in chunks)}")
        
        return chunks
    
    def get_statistics(self, chunks: List[Document]) -> dict:
        """获取分块统计信息"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(c.page_content) for c in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) // len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_chars': sum(chunk_sizes),
            'file_types': list(set(c.metadata.get('file_type', 'unknown') for c in chunks)),
            'chunk_methods': list(set(c.metadata.get('chunk_method', 'unknown') for c in chunks)),
        }
    
    def save_chunks_to_file(self, chunks: List[Document], output_path: str):
        """保存分块结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"{'='*60}\n")
                f.write(f"块索引: {chunk.metadata.get('chunk_index', 'N/A')}\n")
                f.write(f"来源: {chunk.metadata.get('source', 'N/A')}\n")
                f.write(f"分块方法: {chunk.metadata.get('chunk_method', 'N/A')}\n")
                f.write(f"文档ID: {chunk.metadata.get('doc_id', 'N/A')}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk.page_content)
                f.write(f"\n\n")
        
        print(f"💾 分块结果已保存到: {output_path}")


def create_test_file():
    """创建测试文件"""
    test_content = """# 人工智能安全概述

人工智能安全是保障AI系统可靠运行的关键领域。

## 主要安全威胁

### 1. 提示词注入攻击
攻击者通过精心构造的输入，诱导模型执行非预期操作。

### 2. 数据投毒
在训练数据中注入恶意样本，影响模型行为。

### 3. 模型窃取
通过API查询窃取模型参数或功能。

## 防御策略

- 输入验证与过滤
- 输出内容审核
- 访问权限控制
- 全链路审计日志
"""
    
    with open("test_document.md", "w", encoding="utf-8") as f:
        f.write(test_content)
    print("✅ 创建测试文件: test_document.md")


def main():
    """主函数 - 演示完整使用流程"""
    print("""
╔══════════════════════════════════════════════════════════╗
║     RAG文档处理器 - 完整演示程序                         ║
║     支持格式: PDF, TXT, CSV, MD                          ║
║     适配: Ollama + qwen2.5:14b-custom                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 初始化处理器
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # 1. 创建测试文件
    create_test_file()
    
    # 2. 处理单个文档
    print("\n" + "="*60)
    print("示例1: 处理单个Markdown文档")
    print("="*60)
    
    chunks = processor.process_document("test_document.md")
    
    # 3. 查看统计信息
    if chunks:
        stats = processor.get_statistics(chunks)
        print(f"\n📊 统计信息:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    # 4. 显示分块结果
    if chunks:
        print(f"\n📝 分块预览:")
        for chunk in chunks[:3]:  # 只显示前3个块
            print(f"\n  --- 块 {chunk.metadata.get('chunk_index')} ---")
            print(f"  来源: {chunk.metadata.get('source')}")
            print(f"  方法: {chunk.metadata.get('chunk_method')}")
            content_preview = chunk.page_content[:100].replace('\n', ' ')
            print(f"  内容预览: {content_preview}...")
    
    # 5. 保存结果
    if chunks:
        processor.save_chunks_to_file(chunks, "chunks_output.txt")
    
    # 6. 清理测试文件
    print("\n" + "="*60)
    print("清理测试文件...")
    if os.path.exists("test_document.md"):
        os.remove("test_document.md")
        print("   test_document.md 已删除")
    if os.path.exists("chunks_output.txt"):
        print("   chunks_output.txt 已保存，未删除")
    
    print("\n✅ 演示完成！")
    print("\n💡 下一步:")
    print("   1. 将分块结果存入向量数据库（Chroma/Qdrant）")
    print("   2. 使用Ollama + qwen2.5:14b-custom进行检索增强生成")
    print("   3. 实现完整的RAG问答系统")


if __name__ == "__main__":
    main()