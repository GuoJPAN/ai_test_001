#!/usr/bin/env python3
"""
RAG文档处理器 - 极简Demo版
适合初学者理解文档分块的核心概念
原始文档
    ↓
【步骤1】读取文件 → 获取完整文本内容
    ↓
【步骤2】设置参数 → 确定块大小和重叠大小
    ↓
【步骤3】滑动窗口 → 按固定步长切割文本
    ↓
【步骤4】创建块对象 → 为每个块添加元数据
    ↓
【步骤5】输出结果 → 返回分块后的文档列表
"""

import hashlib
import os
from pathlib import Path


# ===================== 简化版Document类 =====================
class SimpleDocument:
    """简化版文档类，不依赖LangChain"""
    def __init__(self, content, metadata=None):
        self.page_content = content      # 文档内容
        self.metadata = metadata or {}   # 元数据（来源、类型等）


# ===================== 简化版文本分块器 =====================
class SimpleTextSplitter:
    """简化版文本分块器 - 核心逻辑易懂"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        初始化分块器
        
        参数:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        """
        将文本切分成多个块
        
        核心逻辑：滑动窗口
        """
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap  # 每次移动的距离
        
        while start < len(text):
            # 计算结束位置
            end = min(start + self.chunk_size, len(text))
            
            # 提取当前块
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 移动到下一个块的开始位置
            start += step
            
            # 如果剩下的文本太少，就结束
            if len(text) - start < self.chunk_size // 2:
                break
        
        return chunks
    
    def split_documents(self, documents):
        """
        将文档列表中的每个文档进行分块
        """
        all_chunks = []
        
        for doc in documents:
            # 对每个文档的内容进行分块
            text_chunks = self.split_text(doc.page_content)
            
            # 为每个块创建新的Document对象
            for i, chunk_text in enumerate(text_chunks):
                # 复制原文档的元数据
                new_metadata = doc.metadata.copy()
                # 添加块特有的元数据
                new_metadata['chunk_index'] = i
                new_metadata['chunk_total'] = len(text_chunks)
                
                # 创建新的文档块
                chunk_doc = SimpleDocument(chunk_text, new_metadata)
                all_chunks.append(chunk_doc)
        
        return all_chunks


# ===================== 简化版文档处理器 =====================
class SimpleDocumentProcessor:
    """简化版文档处理器 - 只支持TXT和MD文件"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """初始化处理器"""
        self.splitter = SimpleTextSplitter(chunk_size, chunk_overlap)
        print(f"📝 文档处理器已初始化")
        print(f"   块大小: {chunk_size} 字符")
        print(f"   重叠大小: {chunk_overlap} 字符")
    
    def load_document(self, file_path):
        """
        加载单个文档（简化版，只支持TXT和MD）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 识别文件类型
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in ['.txt', '.md']:
            print(f"⚠️ 简化版只支持 .txt 和 .md 文件")
            print(f"   当前文件: {file_path}")
            return []
        
        print(f"\n📄 正在加载: {file_path}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建文档对象
        metadata = {
            'source': os.path.basename(file_path),  # 文件名
            'file_type': file_ext[1:],               # 文件类型（txt/md）
            'file_size': len(content),               # 文件大小
        }
        
        doc = SimpleDocument(content, metadata)
        
        print(f"   ✅ 加载完成，共 {len(content)} 字符")
        return [doc]
    
    def process_document(self, file_path):
        """
        一站式处理：加载 + 分块
        """
        print(f"\n{'='*50}")
        print(f"开始处理: {file_path}")
        print(f"{'='*50}")
        
        # 1. 加载文档
        docs = self.load_document(file_path)
        
        if not docs:
            return []
        
        # 2. 分块
        chunks = self.splitter.split_documents(docs)
        
        # 3. 显示统计信息
        print(f"\n✅ 处理完成！")
        print(f"   原始文档: 1 个")
        print(f"   分块数量: {len(chunks)} 个")
        
        if chunks:
            total_chars = sum(len(c.page_content) for c in chunks)
            avg_size = total_chars // len(chunks)
            print(f"   总字符数: {total_chars}")
            print(f"   平均块大小: {avg_size} 字符")
        
        return chunks
    
    def display_chunks(self, chunks, max_display=3):
        """显示分块结果"""
        if not chunks:
            print("没有分块数据")
            return
        
        print(f"\n📝 分块预览（显示前{max_display}个）:")
        print("-" * 50)
        
        for i, chunk in enumerate(chunks[:max_display]):
            print(f"\n【块 {i+1}】")
            print(f"  来源: {chunk.metadata.get('source', '未知')}")
            print(f"  块索引: {chunk.metadata.get('chunk_index', 0)}/{chunk.metadata.get('chunk_total', 0)}")
            print(f"  内容长度: {len(chunk.page_content)} 字符")
            print(f"  内容预览: {chunk.page_content[:80]}...")
            print("-" * 30)
    
    def save_chunks(self, chunks, output_path):
        """保存分块结果到文件"""
        if not chunks:
            print("没有数据可保存")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(f"{'='*60}\n")
                f.write(f"来源文件: {chunk.metadata.get('source', 'N/A')}\n")
                f.write(f"块编号: {chunk.metadata.get('chunk_index', 0)}/{chunk.metadata.get('chunk_total', 0)}\n")
                f.write(f"{'='*60}\n")
                f.write(chunk.page_content)
                f.write(f"\n\n")
        
        print(f"💾 分块结果已保存到: {output_path}")


# ===================== 辅助函数 =====================
def create_sample_file():
    """创建示例文件"""
    content = """# 人工智能安全入门

## 什么是人工智能安全？
人工智能安全是指保护AI系统免受攻击、确保AI行为符合预期的一门技术。

## 主要安全威胁

### 1. 提示词注入
攻击者通过特殊构造的输入来操纵AI的行为。
例如：让AI忽略原本的安全规则。

### 2. 数据投毒
攻击者在训练数据中混入恶意样本。
例如：让AI学会输出错误的信息。

### 3. 模型窃取
攻击者通过大量查询来复制AI的功能。
例如：用10万次查询训练出一个类似的模型。

## 如何防御？

1. 输入过滤：检查用户输入是否包含恶意内容
2. 输出审核：检查AI的输出是否安全
3. 权限控制：限制AI能执行的操作
4. 审计日志：记录所有操作以便追溯

## 总结
AI安全是一个快速发展的领域，需要持续学习和实践。
"""
    
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ 已创建示例文件: sample.txt")


def explain_chunking():
    """解释分块的概念"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                    什么是文本分块？                          ║
╚═══════════════════════════════════════════════════════════════╝

【为什么要分块？】
1. AI模型有输入长度限制（比如只能处理2000字）
2. 从长文本中找信息就像大海捞针，效率低
3. 分块后可以只检索相关的小块，又快又准

【分块的核心参数】
1. chunk_size（块大小）：每个小块的最大长度
2. chunk_overlap（重叠）：块之间重叠的部分

【例子】
原文: "ABCDEFGHIJ"（10个字符）
chunk_size=5, chunk_overlap=2

块1: ABCDE
块2:   CDEFG  （重叠了"CDE"）
块3:     FGHIJ （重叠了"FG"）

重叠的好处：避免信息被切断在边界上！

    """)


def demo_different_chunk_sizes():
    """演示不同块大小的效果"""
    print("\n" + "="*60)
    print("演示：不同块大小的效果")
    print("="*60)
    
    # 创建测试文本
    text = "人工智能安全。提示词注入攻击。数据投毒攻击。模型窃取攻击。防御策略。"
    
    for size in [10, 20, 30]:
        splitter = SimpleTextSplitter(chunk_size=size, chunk_overlap=5)
        chunks = splitter.split_text(text)
        
        print(f"\n块大小 = {size} 字符:")
        print(f"  原文: {text}")
        print(f"  分块数: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"    块{i+1}: {chunk}")


# ===================== 主函数 =====================
def main():
    """主函数 - 演示完整流程"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║            RAG文档处理器 - 极简Demo版                        ║
║            适合初学者理解文档分块的核心概念                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. 先解释什么是分块
    explain_chunking()
    
    # 2. 演示不同块大小的效果
    demo_different_chunk_sizes()
    
    # 3. 创建示例文件
    print("\n" + "="*60)
    print("准备示例文件")
    print("="*60)
    create_sample_file()
    
    # 4. 初始化处理器
    print("\n" + "="*60)
    print("初始化文档处理器")
    print("="*60)
    processor = SimpleDocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # 5. 处理文档
    chunks = processor.process_document("sample.txt")
    
    # 6. 显示分块结果
    processor.display_chunks(chunks, max_display=5)
    
    # 7. 保存结果
    processor.save_chunks(chunks, "demo_chunks_output.txt")
    
    # 8. 清理
    print("\n" + "="*60)
    print("清理文件")
    print("="*60)
    if os.path.exists("sample.txt"):
        os.remove("sample.txt")
        print("🗑️  已删除: sample.txt")
    if os.path.exists("demo_chunks_output.txt"):
        print("📁 已保存: demo_chunks_output.txt")
    
    print("\n" + "="*60)
    print("✅ Demo运行完成！")
    print("="*60)
    print("\n💡 核心要点回顾:")
    print("   1. 分块就是把长文本切成小段")
    print("   2. chunk_size控制每段长度")
    print("   3. chunk_overlap保持上下文连贯")
    print("   4. 元数据记录每个块的来源信息")
    print("\n📚 下一步学习:")
    print("   1. 尝试修改chunk_size和chunk_overlap")
    print("   2. 用自己的txt文件测试")
    print("   3. 学习如何将分块存入向量数据库")


if __name__ == "__main__":
    main()