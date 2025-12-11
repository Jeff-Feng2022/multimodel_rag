"""
FastAPI 应用 - Milvus 版文档查询 API

基于 `api.py`，但使用 `query_milvus.DocumentQuery` 作为实际的查询对象。
启动时运行 `uvicorn api_milvus:app`。
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import os
import uvicorn
from datetime import datetime
from pathlib import Path

# 从 query_milvus 导入查询实现
from query_milvus import DocumentQuery
from langchain_core.output_parsers import StrOutputParser

# Pydantic模型（与 api.py 保持一致）
class QueryRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 3
    output_format: Optional[str] = "detailed"
    use_deepseek: Optional[bool] = False  # 是否启用 DeepSeek 增强回答
    answer_top_k: Optional[int] = 3  # 用于生成答案的前k个检索文档数量


class DocumentMetadata(BaseModel):
    page_number: Optional[int] = None
    filename: Optional[str] = None
    doc_type: Optional[str] = None
    file_directory: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None


class DocumentResult(BaseModel):
    id: int
    content: str
    content_type: str
    metadata: DocumentMetadata
    has_image: bool = False
    image_data: Optional[str] = None
    image_filename: Optional[str] = None


class QueryResponse(BaseModel):
    success: bool
    message: str
    query: str
    total_results: int
    results: List[DocumentResult]
    timestamp: str
    processing_time: Optional[float] = None
    


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str


# 初始化 FastAPI
app = FastAPI(
    title="文档查询API (Milvus)",
    description="基于RAG的文档检索API（Milvus向量数据库）",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态图片目录
static_dir = Path("./static/images")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局查询系统实例
query_system = None


def get_query_system():
    global query_system
    if query_system is None:
        try:
            query_system = DocumentQuery()
            print("✅ Milvus 查询系统初始化成功")
        except Exception as e:
            print(f"❌ Milvus 查询系统初始化失败: {e}")
            raise HTTPException(status_code=500, detail=f"查询系统初始化失败: {e}")
    return query_system


def extract_image_from_metadata(metadata: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    try:
        image_fields = ['image_base64', 'image_data', 'base64_image']
        for field in image_fields:
            if field in metadata and metadata[field]:
                image_base64 = metadata[field]
                if isinstance(image_base64, str) and len(image_base64) > 100:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"query_result_{timestamp}.png"
                    return image_base64, image_filename
        return None, None
    except Exception as e:
        print(f"⚠️ 提取图片数据时出错: {e}")
        return None, None


def save_image_to_static(image_base64: str, filename: str) -> Optional[str]:
    try:
        image_data = base64.b64decode(image_base64)
        image_path = static_dir / filename
        with open(image_path, "wb") as f:
            f.write(image_data)
        return f"/static/{filename}"
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        return None


def format_document_result(doc, index: int) -> DocumentResult:
    content = ""
    if hasattr(doc, 'page_content') and doc.page_content:
        content = doc.page_content
    else:
        content = str(doc)

    content_type = "文本内容"
    metadata_dict = {}
    if hasattr(doc, 'metadata') and doc.metadata:
        metadata_dict = dict(doc.metadata)

    image_data = None
    image_filename = None
    has_image = False

    base64_image, filename = extract_image_from_metadata(metadata_dict)
    if base64_image and filename:
        has_image = True
        image_data = base64_image
        image_url = save_image_to_static(base64_image, filename)
        if image_url:
            # keep image_filename as the relative static path (same as api.py)
            image_filename = image_url

    doc_metadata = DocumentMetadata(
        page_number=metadata_dict.get('page_number'),
        filename=metadata_dict.get('filename'),
        doc_type=metadata_dict.get('doc_type'),
        file_directory=metadata_dict.get('file_directory'),
        source=metadata_dict.get('source'),
        doc_id=metadata_dict.get('doc_id')
    )

    return DocumentResult(
        id=index + 1,
        content=content,
        content_type=content_type,
        metadata=doc_metadata,
        has_image=has_image,
        image_data=image_data,
        image_filename=image_filename
    )


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="文档查询API (Milvus)",
        version="1.0.0"
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    try:
        _ = get_query_system()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            service="文档查询API (Milvus)",
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务不可用: {e}")


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    start_time = datetime.now()
    try:
        qs = get_query_system()
        docs = qs.query(request.query_text, request.top_k)
        results = []
        for i, doc in enumerate(docs):
            formatted_result = format_document_result(doc, i)
            results.append(formatted_result)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return QueryResponse(
            success=True,
            message=f"查询成功，找到 {len(results)} 个相关文档",
            query=request.query_text,
            total_results=len(results),
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/query/simple")
async def simple_query(
    query: str = Query(..., description="查询文本"),
    top_k: int = Query(3, description="返回结果数量"),
    format: str = Query("detailed", description="输出格式: detailed/summary/raw")
):
    request = QueryRequest(
        query_text=query,
        top_k=top_k,
        output_format=format
    )
    return await query_documents(request)


@app.get("/api/formats")
async def get_supported_formats():
    return {
        "supported_formats": [
            {"value": "detailed", "label": "详细格式", "description": "包含完整内容预览、元数据和图片"},
            {"value": "summary", "label": "简洁格式", "description": "简化的内容摘要"},
            {"value": "raw", "label": "原始格式", "description": "原始文档数据"}
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int):
    try:
        raise HTTPException(status_code=404, detail=f"文档ID {doc_id} 未找到")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档失败: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"内部服务器错误: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    print("🚀 启动 Milvus 文档查询API 服务...")
    print("📖 API文档地址: http://localhost:8000/api/docs")
    print("🔍 简单查询接口: http://localhost:8000/api/query/simple")

    uvicorn.run(
        "api_milvus:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
