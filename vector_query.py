from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 检查Chroma数据库路径是否存在
chroma_path = "./chroma_db"
print(f"Chroma数据库路径: {chroma_path}")
print(f"路径是否存在: {os.path.exists(chroma_path)}")

# 检查目录内容
if os.path.exists(chroma_path):
    print(f"Chroma目录内容: {os.listdir(chroma_path)}")

# 获取并打印所有Collections
import chromadb
try:
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    print(f"当前数据库中的所有Collections: {[c.name for c in collections]}")
except Exception as e:
    print(f"获取Collections失败: {e}")

local_model_path = r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-base-en-v15"
print(f"嵌入模型路径: {local_model_path}")
print(f"模型路径是否存在: {os.path.exists(local_model_path)}")

embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path
)

vectorstore = Chroma(
    collection_name="transformers",
    embedding_function=embeddings,
    persist_directory=chroma_path,
)

# 检查集合中的文档数量
try:
    count = vectorstore._collection.count()
    print(f"集合中的文档数量: {count}")
except Exception as e:
    print(f"无法获取文档数量: {e}")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# 尝试不同的查询
queries = [
    "how to compute the attention"  # 添加与simple_query.py相同的查询
]

# 初始化Reranker
from sentence_transformers import CrossEncoder
reranker_model_path = r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-reranker-base"
print(f"加载Reranker模型: {reranker_model_path}")
reranker = CrossEncoder(reranker_model_path)

for query in queries:
    print(f"\n{'='*40}")
    print(f"查询: '{query}'")
    try:
        # 1. 向量检索召回 (Top-K)
        # 增加召回数量，给Rerank更多选择空间
        retriever.search_kwargs["k"] = 10
        docs = retriever.invoke(query)
        print(f"向量检索召回 {len(docs)} 个文档")
        
        if not docs:
            print("未召回任何文档")
            continue

        # 2. Rerank 重排序
        print("正在进行重排序(Rerank)...")
        
        # 准备 (query, doc_content) 对
        doc_contents = [doc.page_content for doc in docs]
        pairs = [[query, content] for content in doc_contents]
        
        # 计算相关性分数
        scores = reranker.predict(pairs)
        
        # 将文档和分数结合并排序
        doc_score_pairs = list(zip(docs, scores))
        # 按分数降序排序
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 输出重排序后的结果 (Top-5)
        print(f"重排序后 Top-5 结果:")
        for i, (doc, score) in enumerate(doc_score_pairs[:5]):
            print(f"\n  [Rank {i+1}] Score: {score:.4f}")
            print(f"    内容预览: {doc.page_content}...")
            print(f"    元数据: {doc.metadata}")
            
    except Exception as e:
        print(f"查询出错: {e}")

# 尝试直接搜索
# print("\n尝试直接相似性搜索:")
# try:
#     docs = vectorstore.similarity_search("What is multihead attention", k=5)
#     print(f"直接搜索返回 {len(docs)} 个文档")
#     for i, doc in enumerate(docs):
#         print(f"  文档 {i+1}:")
#         print(f"    内容预览: {doc.page_content[:100]}...")
#         print(f"    元数据: {doc.metadata}")
# except Exception as e:
#     print(f"直接搜索出错: {e}")