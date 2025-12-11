"""文档查询模块"""

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
    """文档查询类"""
    
    def __init__(self, 
                 vectorstore_path: str = "./chroma_db",
                 collection_name: str = "transformers",
                 embedding_model_path: str = None,
                 deepseek_api_key: str = None):
        """初始化查询系统"""
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
        
        # 初始化文档存储
        self.doc_store = get_mongo_doc_store()
        
        # 创建检索器
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.doc_store,
            id_key="doc_id",
            search_type="similarity",
            search_kwargs={"k": 8}
        )
 
        
        # 初始化DeepSeek LLM
        if deepseek_api_key is None:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

        if not deepseek_api_key:
            raise RuntimeError("请设置DEEPSEEK_API_KEY环境变量")

        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        self.llm_deepseek = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=deepseek_api_key,
            temperature=0.3,
            request_timeout=60,
            max_retries=1
        )

        # 提示模板
        self.prompt = ChatPromptTemplate.from_template("""
            你是专业文档分析助手，请基于提供的上下文回答用户问题。

            要求：
            1. 使用英文回答
            2. 准确、完整、结构清晰
            3. 涉及技术概念时提供详细解释
            4. 信息不足时明确说明无法回答
            5. 仅基于文本内容回答，不提及图片或图表

            上下文：
            {context}

            问题：{question}

            请提供详细回答或明确说明无法回答。
            """)
 
    
    def query(self, query_text: str, top_k: int = None) -> List[Document]:
        """查询文档"""
        
        try:
            print(f"开始查询: '{query_text}'")
            
            # 设定最终需要的数量
            final_k = top_k if top_k else 8
            
            # 向量数据库粗排
            candidate_k = final_k * 3 
            print(f"向量粗排 (召回 {candidate_k} 个候选)...")
            
            vector_results = self.retriever.vectorstore.similarity_search(query_text, k=candidate_k)
            print(f"向量召回完成，找到 {len(vector_results)} 个候选")
            
            if not vector_results:
                return []

            # Rerank 重排序
            print("Rerank 重排序...")
            
            # 准备 (Query, Document) 对
            passages = [doc.page_content for doc in vector_results]
            model_inputs = [[query_text, passage] for passage in passages]
            
            # 计算相关性分数
            scores = self.reranker.score(model_inputs)
            
            # 将分数绑定到文档并排序
            results_with_scores = list(zip(vector_results, scores))
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 截取 Top-K
            sorted_vector_results = [doc for doc, score in results_with_scores[:final_k]]
            print(f"Rerank完成，保留前 {len(sorted_vector_results)} 个文档")

            vector_results = sorted_vector_results 
            
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
                else:
                    print(f"向量结果 {i} 缺少doc_id")
            
            if not doc_ids:
                print("未找到任何doc_id")
                return []
            
            # 从docstore获取完整文档
            print("从MongoDB获取完整文档...")
            full_docs = self.doc_store.mget(doc_ids)
            print(f"MongoDB查询完成，获得 {len(full_docs)} 个完整文档")
            
            # 合并vector搜索结果和完整文档
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
                    elif isinstance(full_doc, Document):
                        doc = full_doc
                        if 'doc_id' not in doc.metadata:
                            doc.metadata['doc_id'] = doc_id
                    else:
                        doc = vector_doc
                else:
                    doc = vector_doc

                # 提取图片/表格数据的逻辑保持不变...
                if hasattr(doc, 'metadata') and doc.metadata:
                    if 'text_as_html' in doc.metadata:
                        doc.metadata['table_html'] = doc.metadata['text_as_html']
                    elif hasattr(doc, 'text_as_html'):
                        doc.metadata['table_html'] = doc.text_as_html

                docs.append(doc)

            return docs

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()
            return []


    def generate_answer(self, query_text: str, docs: list[Document], answer_top_k: int = 3) -> tuple[str, list[dict]]:
        """生成增强回答"""
        if not docs:
            return "无法基于提供的文档回答", []

        # 选择top-k文档
        selected = docs[:answer_top_k]

        # 构建上下文片段
        ctx_parts = []
        sources = []
        for d in selected:
            content = (d.page_content or "").strip()
            # 截断避免过大的提示
            snippet = content[:1500]
            ctx_parts.append(snippet)
            sources.append({
                "id": d.metadata.get("id") or d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "title": d.metadata.get("title"),
                "source": d.metadata.get("source"),
            })

        context_text = "\n\n---\n\n".join(ctx_parts)

        # 构建提示
        if hasattr(self, "prompt") and isinstance(self.prompt, ChatPromptTemplate):
            try:
                prompt_text = self.prompt.format_messages({"context": context_text, "question": query_text})[0].content
            except Exception:
                prompt_text = f"Question: {query_text}\n\nContext:\n{context_text}"
        else:
            prompt_text = f"请基于以下文档片段回答问题：{query_text}\n\nContext:\n{context_text}"

        # 调用DeepSeek LLM
        answer_text = None
        try:
            if hasattr(self, "llm_deepseek") and self.llm_deepseek is not None:
                try:
                    answer_text = self.llm_deepseek.predict(prompt_text)
                except Exception:
                    try:
                        resp = self.llm_deepseek.generate([prompt_text])
                        if resp and hasattr(resp, "generations") and len(resp.generations) > 0:
                            answer_text = resp.generations[0][0].text
                    except Exception:
                        answer_text = None
            elif hasattr(self, "llm") and self.llm is not None:
                try:
                    answer_text = self.llm.predict(prompt_text)
                except Exception:
                    answer_text = None
        except Exception:
            answer_text = None

        # 回退：如果没有生成回答，返回安全的拒绝信息
        if not answer_text:
            return "无法基于提供的文档回答", sources

        return answer_text, sources