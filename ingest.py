"""
文档向量化处理脚本 - 支持PDF文档的多模态内容提取和向量化存储

功能特性:
- 使用unstructured库提取PDF中的文本、表格和图片
- 利用Qwen-VL模型生成图片描述
- 使用Chroma向量数据库存储文本摘要
- 使用MongoDB存储完整文档内容
- 支持多模态检索(文本、表格、图片)

使用方法:
- 默认运行: python ingest.py (保留现有数据)
- 清空数据后重新处理: python ingest.py --clear

注意: 
- 首次运行前请确保已安装所有依赖包
- 需要配置.env文件中的API密钥
- 处理过程可能需要较长时间，取决于PDF文件大小和内容复杂度
"""

import os
import argparse

# 在导入任何unstructured模块之前设置环境变量
# 禁用TensorRT以避免警告
os.environ["ORT_DISABLE_TENSORRT"] = "1"
# 设置ONNX Runtime的执行提供者，明确只使用CPU
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"
os.environ["UNSTRUCTURED_DOWNLOAD_MODELS"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制HuggingFace离线
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Transformers离线
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 数据集离线
os.environ["UNSTRUCTURED_HI_RES_USE_DETR"] = "False"
os.environ["UNSTRUCTURED_HI_RES_USE_YOLO_X"] = "True"
# Path to YOLOX/other models should be provided via environment variable for portability
if "UNSTRUCTURED_YOLO_X_MODEL_PATH" not in os.environ:
    os.environ["UNSTRUCTURED_YOLO_X_MODEL_PATH"] = os.getenv("UNSTRUCTURED_YOLO_X_MODEL_PATH", r"C:/Users/Zhi-F/.cache/huggingface/hub/yolox/yolox_l0.05.onnx")
os.environ["UNSTRUCTURED_DOWNLOAD_MODELS"] = "False"
os.environ["UNSTRUCTURED_PARALLELIZE"] = "True"
os.environ["UNSTRUCTURED_THREADS"] = os.getenv("UNSTRUCTURED_THREADS", "12")

from dotenv import load_dotenv
load_dotenv()

from unstructured.partition.pdf import partition_pdf
import base64
output_path=r"c:/Users/Zhi-F/repo/hugging_face/multimodel_rag/extracted_images/"
pdf_path=r"C:\Users\Zhi-F\repo\multimodel_rag\content\Transformers.pdf"

def get_images_base64(chunks):
    images_base64=[]
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            elements= chunk.metadata.orig_elements
            for el in elements:
                if 'Image' in str(type(el)):
                    images_base64.append(el.metadata.image_base64)
    return images_base64

# 使用partition_pdf函数将PDF文档分解为可处理的文本块
chunks=partition_pdf(filename=pdf_path,          # PDF文件的完整路径
                     infer_table_structure=True,  # 是否自动识别和提取PDF中的表格结构
                     strategy="hi_res",           # PDF解析策略："hi_res"(高精度)，"fast"(快速)，或"ocr_only"(仅OCR)
                     extract_image_block_types=["Image","Table"],  # 提取的图像块类型列表，如["Image"]表示提取所有图片
                     extract_image_block_output_dir=output_path,  # 图像块的保存目录
                     extract_image_block_to_payload=True,  # 是否将提取的图像块作为数据载荷包含在结果中
                     chunking_strategy="by_title",  # 文本分块策略："by_title"(按标题分块)，"by_page"(按页面分块)，或"basic"
                     #max_characters=10000,         # 每个文本块的最大字符数限制，超过则进一步分块
                     #combine_text_under_n_chars=2000,  # 合并小文本块的阈值：小于此长度的文本块会被合并到相邻块
                     #new_after_n_chars=6000)       # 强制新分块的字符数：当累计字符数达到此值时强制开始新分块
                    )
print(len(chunks))

tables = []
texts = []
for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    if "CompositeElement" in str(type(chunk)):
        texts.append(chunk)
print(len(tables))
print(len(texts))

images = get_images_base64(chunks)

print(f"Number of texts: {len(texts)}")
print(f"Number of images: {len(images)}")
print(f"Number of tables: {len(tables)}")
#print(tables[0].to_dict())

from  langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

os.environ["DEEPSEEK_API_KEY"] = "sk-2b53bca28369400ca20dcd08b904332b"
llm_deepseek = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),  # 使用os.getenv而不是os.environ直接访问
            temperature=0.3,  # 降低温度以提高准确性
        )

sys_prompt= """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.
Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.
Table or text chunk: {element}
"""

template= ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{element}")
])

summarize_chain = (template | llm_deepseek | StrOutputParser())
text_sum=summarize_chain.batch(texts, config={"max_concurrency": 5})
table_html=[table.metadata.text_as_html for table in tables]
table_sum=summarize_chain.batch(table_html, config={"max_concurrency": 5})

print(f"Number of text summaries: {len(text_sum)}")
print(f"Number of table summaries: {len(table_sum)}")

from llm_util import img_to_desc

print("开始调用千问获取图片描述......")

# Ensure DEEPSEEK API key is provided via environment variable
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set. Please set DEEPSEEK_API_KEY in environment before running ingest.py")

images_sum=[]
for base64_image in images:
    text=img_to_desc(base64_image)
    images_sum.append(text)

print("调用千问获取图片描述成功")
if images_sum:
    print(images_sum[0])

# =============================================================================  
# 使用新的双语嵌入模型重新索引数据
# =============================================================================

import uuid
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

print("🔄 开始使用新的双语嵌入模型重新索引...")

# 英中双语模型，支持图片英文描述与中文查询的混合检索
local_model_path = r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-base-en-v15"
embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path
)

# 处理命令行参数
parser = argparse.ArgumentParser(description='Ingest PDF documents into vector database')
parser.add_argument('--clear', action='store_true', help='Clear existing data before ingesting new documents')
args = parser.parse_args()

# 创建Chroma向量数据库（可选择是否清空现有数据）
if args.clear:
    print("🗑️ 清空现有Chroma向量数据库...")
    # 如果指定了--clear参数，则先删除现有的Chroma数据库
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("✅ 已删除现有Chroma数据库")

chroma = Chroma(
    collection_name="transformers",
    embedding_function=embeddings, 
    persist_directory="./chroma_db"
)

from mongo import get_mongo_doc_store
doc_store = get_mongo_doc_store()
id_key = "doc_id"

# 创建新的检索器
retriever = MultiVectorRetriever(
    vectorstore=chroma,
    docstore=doc_store,
    id_key=id_key,
)

print("✅ 新向量数据库和检索器初始化完成")
print(f"📊 准备索引：{len(text_sum)} 文本摘要，{len(table_sum)} 表格摘要，{len(images_sum)} 图片摘要")

# =============================================================================
# 开始重新索引所有内容
# =============================================================================

print("📝 开始索引文本摘要...")
doc_ids = [str(uuid.uuid4()) for _ in range(len(text_sum))]
text_sum_docs = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_sum)]
retriever.vectorstore.add_documents(text_sum_docs)  # 把文本摘要存入向量数据库
retriever.docstore.mset(list(zip(doc_ids, texts)))  # 把文本内容存入文档数据库
print(f"✅ 文本索引完成：{len(text_sum)} 个文档")

print("📊 开始索引表格摘要...")
tables_ids = [str(uuid.uuid4()) for _ in range(len(table_sum))]
table_sum_docs = [Document(page_content=summary, metadata={id_key: tables_ids[i]}) for i, summary in enumerate(table_sum)]
retriever.vectorstore.add_documents(table_sum_docs)  # 把表格摘要存入向量数据库

# 为表格数据添加doc_type标注，以便在召回时能够正确识别
tables_with_type = []
for i, table in enumerate(tables):
    # 如果表格对象支持转换为字典，则添加doc_type信息
    if hasattr(table, 'to_dict'):
        try:
            table_dict = table.to_dict()
            table_dict["doc_type"] = "table_document"
            # 保留表格的HTML表示
            if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
                table_dict["text_as_html"] = table.metadata.text_as_html
            tables_with_type.append(table_dict)
        except:
            # 如果转换失败，仍然使用原始对象
            tables_with_type.append(table)
    else:
        # 对于不支持to_dict的对象，创建一个带doc_type的字典
        table_dict = {
            "page_content": str(table),
            "metadata": getattr(table, 'metadata', {}),
            "doc_type": "table_document"
        }
        # 保留表格的HTML表示
        if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html'):
            table_dict["text_as_html"] = table.metadata.text_as_html
        tables_with_type.append(table_dict)

retriever.docstore.mset(list(zip(tables_ids, tables_with_type)))  # 把表格内容存入文档数据库
print(f"✅ 表格索引完成：{len(table_sum)} 个文档")

print("🖼️ 开始索引图片摘要...")
img_ids = [str(uuid.uuid4()) for _ in range(len(images_sum))]
img_sum_docs = [
    Document(
        page_content=summary[0]['text'] if summary and isinstance(summary, list) and len(summary) > 0 else "",
        metadata={id_key: img_ids[i]}
    ) 
    for i, summary in enumerate(images_sum)
]
retriever.vectorstore.add_documents(img_sum_docs)  # 把图片摘要存入向量数据库
print(f"✅ 图片摘要索引完成：{len(images_sum)} 个文档")

# 持久化向量库
chroma.persist()

# 将图片base64数据正确存储到MongoDB中，确保包含必要的文档结构
print("🗄️ 开始存储图片完整数据到MongoDB...")
img_docs_for_mongo = []
for i, (img_id, base64_data) in enumerate(zip(img_ids, images)):
    img_doc = {
        "page_content": images_sum[i][0]['text'] if images_sum[i] and isinstance(images_sum[i], list) and len(images_sum[i]) > 0 else "",
        "metadata": {
            id_key: img_id,
            "image_index": i,
            "base64_length": len(base64_data),
            "summary": "PDF图片内容",
            "has_base64": True
        },
        "doc_type": "image_document",
        "image_base64": base64_data
    }
    img_docs_for_mongo.append((img_id, img_doc))

retriever.docstore.mset(img_docs_for_mongo)  # 把图片内容（含base64）存入文档数据库
print(f"✅ 图片完整数据存储完成：{len(images)} 个图片文档")


print("🔍 测试查询：transformer architecture diagram?")

# 测试查询验证新模型的图片检索效果
docs = retriever.invoke("transformer architecture diagram?")

print(f"📋 查询结果：找到 {len(docs)} 个相关文档")
for i, doc in enumerate(docs, 1):
    # 处理retriever可能返回字典或Document对象的情况
    if hasattr(doc, 'metadata'):
        # Document对象
        doc_type = doc.metadata.get('doc_type', 'unknown')
        content = doc.page_content
    elif isinstance(doc, dict):
        # 字典对象
        doc_type = doc.get('metadata', {}).get('doc_type', 'unknown') if isinstance(doc.get('metadata'), dict) else 'unknown'
        content = doc.get('page_content', '')
    else:
        # 其他类型
        doc_type = 'unknown'
        content = str(doc)
    
    content_preview = content[:100] + "..." if len(content) > 100 else content
    print(f"  {i}. 类型: {doc_type} | 预览: {content_preview}")
    if doc_type == "image_document":
        print("     🖼️ 图片文档被成功检索并排序！")

print("\n" + "="*80)
print("🎉 向量化模型优化完成总结")
print("="*80)