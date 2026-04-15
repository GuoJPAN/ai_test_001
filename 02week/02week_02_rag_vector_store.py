# rag_vector_store.py
"""
RAG向量存储管理器 - 基于LangChain + Chroma + Ollama

功能说明：
1. 管理文档的向量化存储（使用Ollama的嵌入模型）
2. 支持多个集合（collection）隔离不同知识库
3. 支持批量添加文档，提高索引效率

依赖安装：
pip install langchain langchain-community langchain-chroma langchain-ollama chromadb

前置条件：
1. 已安装并启动Ollama服务（https://ollama.com/）
2. 已拉取嵌入模型：ollama pull nomic-embed-text
"""

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Optional
from langchain_core.documents import Document


class VectorStoreManager:
    """企业级向量存储管理器
    
    核心职责：
    1. 管理多个向量集合（类似数据库中的表）
    2. 提供文档的向量化存储和检索能力
    3. 支持批量写入，提升性能
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        初始化向量存储管理器
        
        参数:
            persist_directory: 向量数据库持久化目录路径，默认为当前目录下的chroma_db文件夹
                              Chroma会将向量数据持久化到磁盘，重启后数据不丢失
        """
        self.persist_directory = persist_directory
        
        # 初始化嵌入模型（用于将文本转换为向量）
        # 使用轻量级的 all-minilm 模型（~30MB），下载快、性能好
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",      # 轻量级嵌入模型
            base_url="http://localhost:11434",  # Ollama服务地址
        )
        
        # 存储不同集合的Chroma实例
        # 键：集合名称（如"knowledge_base"），值：Chroma向量数据库对象
        self._stores: Dict[str, Chroma] = {}
    
    def get_or_create_store(self, collection_name: str) -> Chroma:
        """
        获取或创建一个向量集合
        
        参数:
            collection_name: 集合名称（类似数据库表名）
            
        返回:
            Chroma向量数据库对象，可用于添加文档或执行相似度搜索
        """
        if collection_name not in self._stores:
            # 如果集合不存在，则创建新的Chroma实例
            # Chroma会自动创建对应的磁盘目录来持久化数据
            self._stores[collection_name] = Chroma(
                collection_name=collection_name,          # 集合名称
                embedding_function=self.embeddings,       # 嵌入函数（用于向量化）
                persist_directory=f"{self.persist_directory}/{collection_name}",  # 数据存储目录
            )
            print(f"[INFO] 创建新集合: {collection_name}")
        return self._stores[collection_name]
    
    def add_documents_batch(self, collection_name: str, chunks: List[Document], batch_size: int = 50):
        """
        批量添加文档到指定集合
        
        参数:
            collection_name: 目标集合名称
            chunks: 文档分块列表（LangChain Document对象列表）
            batch_size: 每批处理的文档数量，默认50
        """
        # 获取或创建集合
        store = self.get_or_create_store(collection_name)
        
        total = len(chunks)
        print(f"[INFO] 开始索引 {total} 个文档块到集合 '{collection_name}'")
        
        # 分批处理，避免一次性加载过多数据导致内存溢出
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            store.add_documents(batch)  # 将这批文档添加到向量数据库
            print(f"[INFO] 已索引 {min(i + batch_size, total)}/{total}")
        
        # 注意：新版本的 Chroma 会自动持久化，不需要手动调用 persist()
        # 数据会自动保存到 persist_directory 目录
        print(f"[INFO] 索引完成！数据已自动保存至 {self.persist_directory}/{collection_name}")
    
    def similarity_search(self, collection_name: str, query: str, k: int = 5) -> List[Document]:
        """
        在指定集合中执行相似度搜索
        
        参数:
            collection_name: 集合名称
            query: 查询文本
            k: 返回最相似的前k个文档，默认5
            
        返回:
            最相似的k个文档列表（每个文档包含page_content和metadata）
        """
        store = self.get_or_create_store(collection_name)
        return store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, collection_name: str, query: str, k: int = 5) -> List[tuple]:
        """
        在指定集合中执行相似度搜索，并返回相似度分数
        
        参数:
            collection_name: 集合名称
            query: 查询文本
            k: 返回最相似的前k个文档，默认5
            
        返回:
            (文档, 相似度分数) 元组列表，分数越低表示越相似
        """
        store = self.get_or_create_store(collection_name)
        return store.similarity_search_with_relevance_scores(query, k=k)
    
    def delete_collection(self, collection_name: str):
        """
        删除整个集合（慎用！会永久删除该集合中的所有向量数据）
        
        参数:
            collection_name: 要删除的集合名称
        """
        if collection_name in self._stores:
            # 从Chroma客户端删除集合
            self._stores[collection_name].delete_collection()
            # 从内存中移除引用
            del self._stores[collection_name]
            print(f"[WARN] 已删除集合: {collection_name}")
        else:
            print(f"[WARN] 集合 '{collection_name}' 不存在")
    
    def list_collections(self) -> List[str]:
        """列出所有已创建的集合名称"""
        return list(self._stores.keys())


# ===================== 使用示例 =====================
if __name__ == "__main__":
    """
    运行方式：
    python rag_vector_store.py
    
    注意：运行前请确保Ollama服务已启动且嵌入模型已下载
    启动Ollama: ollama serve
    下载模型: ollama pull all-minilm:l6-v2
    """
    
    print("=" * 50)
    print("RAG向量存储管理器 - 使用示例")
    print("=" * 50)
    
    # 检查Ollama服务是否可用
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            print("[ERROR] Ollama服务未正常运行，请先启动: ollama serve")
            exit(1)
        print("[OK] Ollama服务连接成功")
    except requests.exceptions.ConnectionError:
        print("[ERROR] 无法连接到Ollama服务，请确保:")
        print("  1. Ollama已安装: https://ollama.com/")
        print("  2. 服务已启动: ollama serve")
        print("  3. 模型已下载: ollama pull all-minilm:l6-v2")
        exit(1)
    
    # 1. 创建管理器实例
    manager = VectorStoreManager(persist_directory="./my_rag_db")
    
    # 2. 准备示例文档（模拟从PDF或网页加载的内容）
    sample_chunks = [
        Document(
            page_content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            metadata={"source": "ai_intro.txt", "page": 1, "category": "AI基础"}
        ),
        Document(
            page_content="机器学习是AI的一个子领域，它使系统能够从数据中学习并改进，而无需显式编程。",
            metadata={"source": "ml_intro.txt", "page": 1, "category": "机器学习"}
        ),
        Document(
            page_content="深度学习使用多层神经网络，在图像识别、自然语言处理等领域取得了突破性进展。",
            metadata={"source": "dl_intro.txt", "page": 1, "category": "深度学习"}
        ),
        Document(
            page_content="RAG（检索增强生成）结合了信息检索和语言生成，让LLM能够访问外部知识库，减少幻觉问题。",
            metadata={"source": "rag_concept.txt", "page": 1, "category": "RAG技术"}
        ),
        Document(
            page_content="LangChain是一个用于开发LLM应用的框架，提供了链式调用、代理、记忆等核心组件。",
            metadata={"source": "langchain_intro.txt", "page": 1, "category": "开发框架"}
        ),
    ]
    
    # 3. 批量添加文档到"ai_knowledge"集合
    print("\n[步骤1] 添加文档到向量数据库...")
    manager.add_documents_batch(
        collection_name="ai_knowledge",
        chunks=sample_chunks,
        batch_size=2  # 小批量演示
    )
    
    # 4. 执行相似度搜索
    print("\n[步骤2] 执行相似度搜索...")
    queries = [
        "什么是RAG？",
        "机器学习是什么？",
        "LangChain有什么用？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = manager.similarity_search(
            collection_name="ai_knowledge",
            query=query,
            k=2  # 返回最相似的2个文档
        )
        
        print("搜索结果:")
        for i, doc in enumerate(results, 1):
            # 截断过长的内容
            content = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            print(f"  {i}. {content}")
            print(f"     来源: {doc.metadata.get('source', '未知')}")
    
    # 5. 带分数的相似度搜索
    print("\n[步骤3] 带分数的相似度搜索...")
    query = "深度学习神经网络"
    results_with_score = manager.similarity_search_with_score(
        collection_name="ai_knowledge",
        query=query,
        k=2
    )
    
    print(f"查询: {query}")
    for i, (doc, score) in enumerate(results_with_score, 1):
        print(f"  {i}. 相似度分数: {score:.4f}")
        print(f"     内容: {doc.page_content[:60]}...")
    
    # 6. 查看集合信息
    print("\n[步骤4] 查看集合状态...")
    print(f"当前管理的集合: {manager.list_collections()}")
    print(f"集合 'ai_knowledge' 数据目录: {manager.persist_directory}/ai_knowledge")
    
    print("\n✅ 示例运行完成！")
    print("提示：如需重新运行，建议先删除 ./my_rag_db 目录避免数据重复")