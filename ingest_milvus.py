"""
Ingest script that parses PDF, summarizes text/table, describes images,
and indexes documents into Milvus (vector DB) and MongoDB (document store).

Configuration is taken from environment variables. Defaults are provided
but you should override secrets via environment (do NOT commit secrets).
"""
import os
# 在导入任何unstructured模块之前设置环境变量
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

import json
import uuid
import base64
from pathlib import Path

try:
    from unstructured.partition.pdf import partition_pdf
except Exception:
    raise

try:
    from langchain_community.vectorstores import Milvus
    from langchain_core.documents import Document
except Exception:
    raise RuntimeError("langchain_community Milvus and langchain_core are required. Install with `pip install langchain-community langchain-core`")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from llm_util import img_to_desc
from mongo import get_mongo_doc_store


# -------------------- Config / env --------------------
PDF_PATH = os.getenv("INGEST_PDF_PATH", r"c:/Users/Zhi-F/repo/hugging_face/content/attention.pdf")
OUTPUT_IMG_DIR = os.getenv("INGEST_OUTPUT_IMG_DIR", r"c:/Users/Zhi-F/repo/hugging_face/multimodel_rag/extracted_images/")

# Milvus config - prefer environment variables, fallback to provided values
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "transformers")
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-d209eefb9b23d4d.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
MILVUS_USER = os.getenv("MILVUS_USER", "db_d209eefb9b23d4d")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "!QAZ2wsx")

# Embedding model path (local) or name
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_PATH", r"C:\Users\Zhi-F\.cache\modelscope\hub\models\BAAI\bge-base-en-v15")


def parse_pdf(path):
    print(f"Parsing PDF: {path}")
    chunks = partition_pdf(
        filename=path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=OUTPUT_IMG_DIR,
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
    )
    return chunks


def extract_texts_tables_images(chunks):
    texts = []
    tables = []
    images_b64 = []

    for chunk in chunks:
        t = type(chunk).__name__
        if "Table" in t:
            tables.append(chunk)
        elif "CompositeElement" in t or "NarrativeText" in t or "Text" in t or "Title" in t:
            texts.append(chunk)
        # extract images from composite elements
        if hasattr(chunk, "metadata") and getattr(chunk.metadata, "orig_elements", None):
            for el in chunk.metadata.orig_elements:
                if "Image" in type(el).__name__:
                    b64 = getattr(el.metadata, "image_base64", None)
                    if b64:
                        images_b64.append(b64)

    return texts, tables, images_b64


def summarize_with_deepseek(texts, tables):
    # Require DEEPSEEK_API_KEY in env
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY must be set in environment to run ingest_milvus.py")

    llm = ChatOpenAI(
        model_name="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=deepseek_key,
        temperature=0.3,
    )

    sys_prompt = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additional comment.
    """

    template = ChatPromptTemplate.from_template(sys_prompt + "\n\n{element}")
    summarize_chain = (template | llm | StrOutputParser())

    # Convert text chunks to plain strings for summarization
    text_inputs = [getattr(t, "text", None) or getattr(t, "page_content", "") for t in texts]
    table_inputs = [getattr(tbl.metadata, "text_as_html", "") for tbl in tables]

    text_summaries = summarize_chain.batch(text_inputs, config={"max_concurrency": 5}) if text_inputs else []
    table_summaries = summarize_chain.batch(table_inputs, config={"max_concurrency": 5}) if table_inputs else []

    return text_summaries, table_summaries


def describe_images(images_b64):
    descriptions = []
    for b64 in images_b64:
        desc = img_to_desc(b64)
        descriptions.append(desc)
    return descriptions


# Using LangChain's Milvus vectorstore for simpler integration


def main():
    chunks = parse_pdf(PDF_PATH)
    texts, tables, images_b64 = extract_texts_tables_images(chunks)

    print(f"Found {len(texts)} text chunks, {len(tables)} tables, {len(images_b64)} images")

    text_summaries, table_summaries = summarize_with_deepseek(texts, tables)
    image_descriptions = describe_images(images_b64)

    print(f"Prepared {len(text_summaries)} text summaries, {len(table_summaries)} table summaries, {len(image_descriptions)} image descriptions")

    # Prepare documents to index: combine summaries and descriptions
    docs_for_index = []
    mongo_docs = []

    # text summaries
    for i, summary in enumerate(text_summaries):
        doc_id = str(uuid.uuid4())
        docs_for_index.append(summary)
        mongo_docs.append((doc_id, {"page_content": summary, "metadata": {"source": "text_summary", "index": i}}))

    # table summaries
    for i, summary in enumerate(table_summaries):
        doc_id = str(uuid.uuid4())
        docs_for_index.append(summary)
        mongo_docs.append((doc_id, {"page_content": summary, "metadata": {"source": "table_summary", "index": i}}))

    # image descriptions
    for i, desc in enumerate(image_descriptions):
        doc_id = str(uuid.uuid4())
        text_content = desc[0]['text'] if isinstance(desc, list) and desc and isinstance(desc[0], dict) and 'text' in desc[0] else (desc if isinstance(desc, str) else "")
        docs_for_index.append(text_content)
        mongo_docs.append((doc_id, {"page_content": text_content, "metadata": {"source": "image_description", "index": i}}))

    if not docs_for_index:
        print("No documents to index. Exiting.")
        return

    # initialize embeddings
    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # compute vectors
    print(f"Embedding {len(docs_for_index)} documents...")
    vectors = embeddings.embed_documents(docs_for_index)

    dim = len(vectors[0])
    print(f"Embedding dimension: {dim}")

    # Build LangChain Document objects for Milvus ingestion
    split_docs = []
    for doc_id, doc in mongo_docs:
        page_content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        # ensure doc_id present in metadata
        metadata["doc_id"] = doc_id
        split_docs.append(Document(page_content=page_content, metadata=metadata))

    # Build connection args for Milvus (read from env)
    connection_args = {"uri": MILVUS_URI}
    if MILVUS_USER:
        connection_args["user"] = MILVUS_USER
    if MILVUS_PASSWORD:
        connection_args["password"] = MILVUS_PASSWORD

    print(f"Indexing {len(split_docs)} documents into Milvus collection '{MILVUS_COLLECTION}' using LangChain vectorstore...")

    # Create/overwrite Milvus collection via LangChain helper
    vectorstore = Milvus.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=MILVUS_COLLECTION,
        connection_args=connection_args,
        drop_old=True,
        consistency_level="Strong",
    )

    # store original docs into Mongo
    doc_store = get_mongo_doc_store()
    print(f"Storing {len(mongo_docs)} documents into MongoDB via doc_store.mset ...")
    doc_store.mset(mongo_docs)

    print("Ingestion to Milvus + Mongo completed.")


if __name__ == "__main__":
    main()
