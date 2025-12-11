"""
文档查询模块 - RAG系统with DeepSeek增强回答

该模块提供RAG检索功能，使用deepseek进行增强回答
参考six_sister_query.py的实现方式
"""

import os

# 在导入任何其他模块之前设置环境变量以避免ONNX Runtime警告
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

from langchain_core.documents.base import Document
from typing import List
from langchain_community.vectorstores import Chroma
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
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers import MultiQueryRetriever  # 用于多查询检索器，可以把一个问题改写成多个问题
#from langchain_classic.retrievers import EnsembleRetriever  # 用于BM25关键词检索器
#from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import AmazonKendraRetriever
#from langchain_community.retrievers import EnsembleRetriever

class DocumentQuery:
    """文档查询类，提供RAG检索功能with DeepSeek增强回答"""
    
    def __init__(self, 
                 vectorstore_path: str = "./chroma_db",
                 collection_name: str = "transformers",
                 mongo_db_name: str = "transformers",
                 mongo_collection_name: str = "documents",
                 embedding_model_path: str = None,
                 deepseek_api_key: str = None):
        """
        初始化查询系统
        
        Args:
            vectorstore_path: Chroma向量数据库路径
            collection_name: Chroma集合名称
            mongo_db_name: MongoDB数据库名称
            mongo_collection_name: MongoDB集合名称
            embedding_model_path: 嵌入模型路径
            deepseek_api_key: DeepSeek API密钥
        """
        self.vectorstore_path = vectorstore_path
        self.collection_name = collection_name
        
        # 初始化嵌入模型（使用双语嵌入模型）
        if embedding_model_path is None:
            embedding_model_path = r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-base-en-v15"
        reranker_model_path = r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-reranker-base"
  
 
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path
        )

        print(f"Loading Reranker model from: {reranker_model_path}")
        self.reranker = HuggingFaceCrossEncoder(model_name=reranker_model_path)

        
        
        # 初始化向量数据库
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=vectorstore_path
        )
        
        # 初始化文档存储 (使用默认配置：数据库"transformers"，集合"documents")
        self.doc_store = get_mongo_doc_store()
        
        # 创建检索器（添加更多查询参数优化）
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.doc_store,
            id_key="doc_id",
            search_type="similarity",
            # 可以添加以下参数来优化查询：
             search_kwargs={
                 "k": 8,  # 返回文档数量（默认4）
             }
        )
 
        
        # 初始化DeepSeek LLM
        # Prefer explicit parameter, otherwise read from environment variable
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

        # Prompt template (English text, instructs assistant to answer in Chinese)
        self.prompt = ChatPromptTemplate.from_template("""
            You are a professional document analysis assistant. Please answer the user's question based on the provided context.

            Please follow these requirements:
            1. Answer in English.
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
        查询文档（基础检索）- 修复版本
        
        真正的两阶段查询：
        1. 向量数据库搜索相似向量
        2. 根据doc_id从MongoDB获取完整文档
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量限制
            
        Returns:
            相关文档列表
        """
        
        try:
            print(f"🔍 开始两阶段查询: '{query_text}'")
            
            # 第一阶段：向量数据库搜索
            print("📋 阶段1: 向量数据库搜索...")
            if top_k:
                vector_results = self.retriever.vectorstore.similarity_search(query_text, k=top_k)
            else:
                vector_results = self.retriever.vectorstore.similarity_search(query_text, k=8)
            
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
            
            # 第二阶段：从docstore获取完整文档
            print(f"📋 阶段2: 从MongoDB获取完整文档...")
            print(f"  🔍 查询 {len(doc_ids)} 个doc_id: {doc_ids}")
            
            full_docs = self.doc_store.mget(doc_ids)
            print(f"✅ MongoDB查询完成，获得 {len(full_docs)} 个完整文档")
            
            # 合并vector搜索结果和完整文档
            docs = []
            for i, (vector_doc, full_doc) in enumerate(zip(vector_results, full_docs)):
                doc_id = vector_info[i]['doc_id']
                
                if full_doc:
                    # 优先使用docstore中的完整文档
                    if isinstance(full_doc, dict):
                        # 直接使用MongoDB返回的完整结构
                        metadata = full_doc.get('metadata', {}) or {}
                        
                        # 确保包含doc_type信息（来自MongoDB的_deserialize_document方法）
                        doc_type = full_doc.get('doc_type', 'unknown')
                        if doc_type != 'unknown':
                            metadata['doc_type'] = doc_type
                        
                        # 如果MongoDB有返回完整的doc_data结构，直接使用它
                        raw_doc_data = full_doc.get('raw_doc_data', {})
                        if raw_doc_data:
                            metadata['doc_data'] = raw_doc_data
                        
                        # 提取图片数据
                        if 'image_base64' in full_doc:
                            metadata['image_base64'] = full_doc['image_base64']
                            metadata['content_type'] = full_doc.get('content_type', 'image/png')
                            metadata['original_filename'] = full_doc.get('original_filename', 'image.png')
                        
                        # 创建Document对象
                        doc = Document(
                            page_content=full_doc.get('page_content', ''),
                            metadata=metadata
                        )
                        # 确保包含doc_id
                        doc.metadata['doc_id'] = doc_id
                        
                        print(f"  ✅ MongoDB完整文档 {doc_id}: {len(doc.page_content)} 字符")
                        print(f"      doc_type: {doc_type}")
                        print(f"      包含图片: {'是' if 'image_base64' in metadata else '否'}")
                    elif isinstance(full_doc, Document):
                        # 如果已经是Document对象
                        doc = full_doc
                        if 'doc_id' not in doc.metadata:
                            doc.metadata['doc_id'] = doc_id
                        print(f"  ✅ MongoDB Document对象 {doc_id}: {len(doc.page_content)} 字符")
                    else:
                        # 兜底：使用vector搜索结果
                        doc = vector_doc
                        print(f"  ⚠️  使用向量搜索结果 {doc_id}: {len(doc.page_content)} 字符")
                else:
                    # 兜底：使用vector搜索结果
                    doc = vector_doc
                    print(f"  ⚠️  MongoDB未找到，使用向量搜索结果 {doc_id}: {len(doc.page_content)} 字符")
                
                # 提取图片数据
                if hasattr(doc, 'metadata') and doc.metadata:
                    image_data = {}
                    if 'image_base64' in doc.metadata:
                        image_data['image_base64'] = doc.metadata['image_base64']
                        image_data['content_type'] = doc.metadata.get('content_type', 'image/png')
                        image_data['original_filename'] = doc.metadata.get('original_filename', f'image_{i+1}.png')
                    
                    if image_data:
                        print(f"  🖼️  {doc_id} 包含图片数据: {list(image_data.keys())}")
                
                # 提取表格HTML数据（如果存在）
                if hasattr(doc, 'metadata') and doc.metadata:
                    # 检查是否包含表格的HTML表示
                    if 'text_as_html' in doc.metadata:
                        # 将表格HTML数据添加到metadata中，供前端使用
                        doc.metadata['table_html'] = doc.metadata['text_as_html']
                        print(f"  📊  {doc_id} 包含表格数据")
                    elif hasattr(doc, 'text_as_html'):
                        # 如果text_as_html是文档对象的属性
                        doc.metadata['table_html'] = doc.text_as_html
                        print(f"  📊  {doc_id} 包含表格数据")

                docs.append(doc)
            
            print(f"✅ 两阶段查询完成，返回 {len(docs)} 个完整文档")
            return docs
            
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
            return []


    def generate_answer(self, query_text: str, docs: list[Document], answer_top_k: int = 3) -> tuple[str, list[dict]]:
        """
        Generate an enhanced answer using DeepSeek (or configured llm).
        This is a new method to avoid changing existing `query()` behaviour.

        Returns: (answer_text, sources)
        - answer_text: the generated answer string, or a safe refusal string when evidence insufficient
        - sources: list of dicts with keys `id`, `page`, `title`, `source` (mirrors existing response shape)
        """
        # basic gating: if no docs, refuse
        if not docs:
            return (
                "I cannot answer based on the provided documents.",
                [],
            )

        # pick top-k documents (preserve order if already ranked)
        selected = docs[:answer_top_k]

        # Build context snippets
        ctx_parts: list[str] = []
        sources: list[dict] = []
        for d in selected:
            content = (d.page_content or "").strip()
            # truncate to avoid huge prompts
            snippet = content[:1500]
            ctx_parts.append(snippet)
            sources.append({
                "id": d.metadata.get("id") or d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "title": d.metadata.get("title"),
                "source": d.metadata.get("source"),
            })

        context_text = "\n\n---\n\n".join(ctx_parts)

        # Construct prompt: use existing prompt template if present, else a simple template
        if hasattr(self, "prompt") and isinstance(self.prompt, ChatPromptTemplate):
            # try to use the template's formatting; append the context
            try:
                prompt_text = (
                    self.prompt.format_messages({"query": query_text})[0].content
                    + "\n\nContext:\n"
                    + context_text
                    + "\n\nIf the provided documents do not contain enough information to answer, reply exactly: 'I cannot answer based on the provided documents.'"
                )
            except Exception:
                prompt_text = (
                    f"Question: {query_text}\n\nContext:\n{context_text}\n\nIf the provided documents do not contain enough information to answer, reply exactly: 'I cannot answer based on the provided documents.'"
                )
        else:
            prompt_text = (
                f"Answer the question based only on the following extracted document snippets. If you cannot answer from these snippets, reply exactly: 'I cannot answer based on the provided documents.'\n\nQuestion: {query_text}\n\nContext:\n{context_text}"
            )

        # Attempt to call the configured DeepSeek LLM (self.llm_deepseek) if available
        answer_text = None
        try:
            if hasattr(self, "llm_deepseek") and self.llm_deepseek is not None:
                # ChatOpenAI typically supports __call__ or predict
                try:
                    # prefer simple call
                    answer_text = self.llm_deepseek.predict(prompt_text)
                except Exception:
                    try:
                        resp = self.llm_deepseek.generate([prompt_text])
                        # attempt to extract text
                        if resp and hasattr(resp, "generations") and len(resp.generations) > 0:
                            answer_text = resp.generations[0][0].text
                    except Exception:
                        answer_text = None
            else:
                if hasattr(self, "llm") and self.llm is not None:
                    try:
                        answer_text = self.llm.predict(prompt_text)
                    except Exception:
                        answer_text = None
        except Exception:
            answer_text = None

        # fallback: if still no answer_text, produce a safe refusal
        if not answer_text:
            return (
                "I cannot answer based on the provided documents.",
                sources,
            )

        # Return answer and provenance
        return (answer_text, sources)