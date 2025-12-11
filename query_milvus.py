"""
文档查询模块（Milvus版） - 将向量数据库替换为 Milvus

该模块基于 `query.py` 的逻辑实现两阶段检索（向量搜索 -> Mongo 完整文档回填），
但使用 LangChain 社区版的 Milvus 向量存储作为向量数据库。

配置通过环境变量读取：`MILVUS_URI`, `MILVUS_USER`, `MILVUS_PASSWORD`, `MILVUS_COLLECTION`。
"""
from langchain_core.documents.base import Document
import os
from typing import List
from langchain_community.vectorstores import Milvus
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from mongo import get_mongo_doc_store


class DocumentQuery:
    """文档查询类（Milvus）"""

    def __init__(self,
                 collection_name: str = None,
                 mongo_db_name: str = "transformers",
                 mongo_collection_name: str = "documents",
                 embedding_model_path: str = None,
                 deepseek_api_key: str = None,
                 milvus_uri: str = None,
                 milvus_user: str = None,
                 milvus_password: str = None):
        """
        初始化查询系统，使用 Milvus 作为向量存储
        """
        # Embedding model
        if embedding_model_path is None:
            embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-base-en-v15")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path
        )


        # Milvus connection configuration (env overrides constructor args)
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "transformers")
        milvus_uri = milvus_uri or os.getenv("MILVUS_URI", "https://in03-d209eefb9b23d4d.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
        milvus_user = milvus_user or os.getenv("MILVUS_USER", "db_d209eefb9b23d4d")
        milvus_password = milvus_password or  os.getenv("MILVUS_PASSWORD", "!QAZ2wsx")
        connection_args = {}
        if milvus_uri:
            connection_args["uri"] = milvus_uri
        if milvus_user:
            connection_args["user"] = milvus_user
        if milvus_password:
            connection_args["password"] = milvus_password

        # 初始化 Milvus 向量数据库（使用 LangChain 社区适配器）
        print(f"Initializing Milvus vectorstore for collection '{self.collection_name}' with connection_args keys: {list(connection_args.keys())}")
        try:
            # The Milvus vectorstore exposes the same high-level API as other LangChain vectorstores
            self.vectorstore = Milvus(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                connection_args=connection_args if connection_args else None,
            )
        except TypeError:
            # Fallback: Some langchain-community versions expect different ctor args;
            # attempt to create via from_documents with no-op documents to attach to existing collection.
            try:
                print("Milvus() constructor failed; attempting Milvus.from_documents fallback...")
                self.vectorstore = Milvus.from_documents(
                    documents=[],
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args=connection_args if connection_args else None,
                    drop_old=False,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Milvus vectorstore: {e}")

        # 文档存储（Mongo）
        self.doc_store = get_mongo_doc_store()

        # 检索器
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.doc_store,
            id_key="doc_id",
            search_type="similarity",
            search_kwargs={
                "k": 8,
            }
        )

        # DeepSeek LLM 初始化
        if deepseek_api_key is None:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        if not deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Please set the DEEPSEEK_API_KEY environment variable.")

        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        self.llm_deepseek = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0.3,
            request_timeout=60,
            max_retries=1,
            callbacks=[],
        )

        # 中文 Prompt 模板
        # Prompt template (English text, instructs assistant to answer in Chinese)
        self.prompt = ChatPromptTemplate.from_template("""
            You are a professional document analysis assistant. Please answer the user's question based on the provided context.

            Please follow these requirements:
            1. Answer in Chinese.
            2. Be accurate, complete, and well-structured.
            3. If the question involves specific technical concepts, provide detailed explanations.
            4. If the context is insufficient or the information needed is not present in the provided context, explicitly say "I cannot answer based on the provided context." Do not attempt to hallucinate or invent facts.
            5. Do not mention images or charts; answer based on the textual content only.

            Context:
            {context}

            Question: {question}

            Please provide a detailed answer or explicitly state you cannot answer.
            """)

    def query(self, query_text: str, top_k: int = None) -> List[Document]:
        """
        两阶段查询：向量检索 -> Mongo 回填完整文档
        """
        try:
            print(f"🔍 开始两阶段查询 (Milvus): '{query_text}'")
            print("📋 阶段1: 向量数据库搜索（Milvus）...")
            k = top_k or 8
            vector_results = self.retriever.vectorstore.similarity_search(query_text, k=k)
            print(f"✅ 向量搜索完成，找到 {len(vector_results)} 个向量结果")

            # 提取doc_id
            doc_ids = []
            vector_info = []
            for i, doc in enumerate(vector_results):
                doc_id = doc.metadata.get('doc_id')
                if doc_id:
                    doc_ids.append(doc_id)
                    vector_info.append({
                        'index': i,
                        'doc_id': doc_id,
                        'content_preview': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    })
                    print(f"  📋 提取doc_id: {doc_id}")
                else:
                    print(f"  ⚠️  向量结果 {i} 缺少doc_id")

            if not doc_ids:
                print("❌ 未找到任何doc_id")
                return []

            # 阶段2：从Mongo获取完整文档
            print(f"📋 阶段2: 从MongoDB获取完整文档... 查询 {len(doc_ids)} 个doc_id")
            full_docs = self.doc_store.mget(doc_ids)
            print(f"✅ MongoDB查询完成，获得 {len(full_docs)} 个完整文档")

            # 合并
            docs = []
            for i, (vector_doc, full_doc) in enumerate(zip(vector_results, full_docs)):
                doc_id = vector_info[i]['doc_id']
                if full_doc:
                    if isinstance(full_doc, dict):
                        metadata = full_doc.get('metadata', {}) or {}
                        doc_type = full_doc.get('doc_type', 'unknown')
                        if doc_type != 'unknown':
                            metadata['doc_type'] = doc_type
                        raw_doc_data = full_doc.get('raw_doc_data', {})
                        if raw_doc_data:
                            metadata['doc_data'] = raw_doc_data
                        if 'image_base64' in full_doc:
                            metadata['image_base64'] = full_doc['image_base64']
                            metadata['content_type'] = full_doc.get('content_type', 'image/png')
                            metadata['original_filename'] = full_doc.get('original_filename', 'image.png')

                        doc = Document(
                            page_content=full_doc.get('page_content', ''),
                            metadata=metadata
                        )
                        doc.metadata['doc_id'] = doc_id
                        print(f"  ✅ MongoDB完整文档 {doc_id}: {len(doc.page_content)} 字符")
                    elif isinstance(full_doc, Document):
                        doc = full_doc
                        if 'doc_id' not in doc.metadata:
                            doc.metadata['doc_id'] = doc_id
                        print(f"  ✅ MongoDB Document对象 {doc_id}: {len(doc.page_content)} 字符")
                    else:
                        doc = vector_doc
                        print(f"  ⚠️  使用向量搜索结果 {doc_id}: {len(doc.page_content)} 字符")
                else:
                    doc = vector_doc
                    print(f"  ⚠️  MongoDB未找到，使用向量搜索结果 {doc_id}: {len(doc.page_content)} 字符")

                # 图片信息
                if hasattr(doc, 'metadata') and doc.metadata:
                    if 'image_base64' in doc.metadata:
                        print(f"  🖼️  {doc_id} 包含图片数据")

                docs.append(doc)

            print(f"✅ 两阶段查询完成，返回 {len(docs)} 个完整文档")
            return docs

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
            return []
