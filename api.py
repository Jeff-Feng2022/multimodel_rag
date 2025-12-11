"""文档查询API服务"""

import os

# 在导入任何其他模块之前设置环境变量以避免ONNX Runtime警告
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "CPUExecutionProvider"

from dotenv import load_dotenv
load_dotenv()   # 会加载项目根目录的 .env

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
import json
from pathlib import Path
import concurrent.futures

# Minimal MLflow integration (safe if mlflow not installed)
try:
    import mlflow
    MLFLOW_ENABLED = True
except Exception:
    mlflow = None
    MLFLOW_ENABLED = False

# Thread pool for async logging so we don't block requests
_mlflow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# If user provided a tracking URI via env, set it explicitly so mlflow client uses it
_mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_ENABLED and _mlflow_tracking_uri:
    try:
        mlflow.set_tracking_uri(_mlflow_tracking_uri)
    except Exception:
        pass

print(f"MLFLOW_ENABLED={MLFLOW_ENABLED}, MLFLOW_TRACKING_URI={_mlflow_tracking_uri}")

# 导入查询模块
from query import DocumentQuery

load_dotenv()

# Pydantic模型定义
class QueryRequest(BaseModel):
    """查询请求模型"""
    query_text: str
    top_k: Optional[int] = 3
    output_format: Optional[str] = "detailed"  # "detailed", "summary", "raw"
    use_deepseek: Optional[bool] = True
    answer_top_k: Optional[int] = 3
    prefer_diagrams: Optional[bool] = False  # 新增：是否优先返回图表相关文档


class DocumentMetadata(BaseModel):
    """文档元数据模型"""
    page_number: Optional[int] = None
    filename: Optional[str] = None
    doc_type: Optional[str] = None
    file_directory: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None


class DocumentResult(BaseModel):
    """文档查询结果模型"""
    id: int
    content: str
    content_type: str
    metadata: DocumentMetadata
    has_image: bool = False
    image_data: Optional[str] = None  # base64编码的图片数据
    image_filename: Optional[str] = None
    table_html: Optional[str] = None  # 表格的HTML表示


class QueryResponse(BaseModel):
    """API响应模型"""
    success: bool
    message: str
    query: str
    total_results: int
    results: List[DocumentResult]
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    timestamp: str
    processing_time: Optional[float] = None


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    service: str
    version: str


# 初始化FastAPI应用
app = FastAPI(
    title="文档查询API",
    description="文档检索API服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录用于图片访问
static_dir = Path("./static/images")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 全局查询系统实例
query_system = None


def get_query_system():
    """获取查询系统实例"""
    global query_system
    if query_system is None:
        try:
            query_system = DocumentQuery()
            print("查询系统初始化成功")
        except Exception as e:
            print(f"查询系统初始化失败: {e}")
            raise HTTPException(status_code=500, detail=f"查询系统初始化失败: {e}")
    return query_system


def _mlflow_log_query_async(query_text: str, params: dict, docs: list, answer_text: str | None, timings: dict | None = None):
    """Minimal async logger for MLflow. Uses only params/metrics to avoid large artifacts.

    This is intentionally small: it will not fail if mlflow isn't available.
    """
    if not MLFLOW_ENABLED:
        return
    try:
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "rag_experiments"))
        with mlflow.start_run(nested=False):
            # basic params
            mlflow.log_param("query_trunc", (query_text or "")[:1000])
            for k, v in (params or {}).items():
                try:
                    mlflow.log_param(str(k), str(v))
                except Exception:
                    pass
            # metrics
            mlflow.log_metric("retrieved_docs", float(len(docs or [])))
            if timings:
                for tk, tv in timings.items():
                    try:
                        mlflow.log_metric(tk, float(tv))
                    except Exception:
                        pass
            # answer (truncated)
            if answer_text:
                try:
                    mlflow.log_param("answer_trunc", answer_text[:1000])
                except Exception:
                    pass
            # provenance small summary
            try:
                prov = []
                for d in (docs or []):
                    mid = None
                    try:
                        mid = d.metadata.get('doc_id') if getattr(d, 'metadata', None) else None
                    except Exception:
                        mid = None
                    prov.append({"id": mid})
                import json as _json
                mlflow.log_param("provenance", (_json.dumps(prov, ensure_ascii=False))[:1000])
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")





def extract_image_from_metadata(metadata: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """从元数据中提取图片数据"""
    try:
        # 检查各种可能的图片字段
        image_fields = ['image_base64', 'image_data', 'base64_image']
        
        for field in image_fields:
            if field in metadata and metadata[field]:
                image_base64 = metadata[field]
                if isinstance(image_base64, str) and len(image_base64) > 100:  # 确保是有效的base64数据
                    # 生成唯一文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"query_result_{timestamp}.png"
                    
                    return image_base64, image_filename
        
        return None, None
    except Exception as e:
        print(f"⚠️ 提取图片数据时出错: {e}")
        return None, None


def save_image_to_static(image_base64: str, filename: str) -> Optional[str]:
    """将base64图片数据保存到静态目录"""
    try:
        # 解码base64数据
        image_data = base64.b64decode(image_base64)
        
        # 保存文件
        image_path = static_dir / filename
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # 返回可访问的URL
        return f"/static/{filename}"
    except Exception as e:
        print(f"❌ 保存图片失败: {e}")
        return None


def format_document_result(doc, index: int) -> DocumentResult:
    """格式化文档结果"""
    # 提取内容
    content = ""
    if hasattr(doc, 'page_content') and doc.page_content:
        content = doc.page_content
    else:
        content = str(doc)
    
    # 简化内容类型识别
    content_type = "文本内容"
    
    # 处理元数据
    metadata_dict = {}
    if hasattr(doc, 'metadata') and doc.metadata:
        metadata_dict = dict(doc.metadata)
    
    # 提取图片数据
    image_data = None
    image_filename = None
    has_image = False
    
    base64_image, filename = extract_image_from_metadata(metadata_dict)
    if base64_image and filename:
        has_image = True
        image_data = base64_image
        
        # 保存图片到静态目录并获取URL
        image_url = save_image_to_static(base64_image, filename)
        if image_url:
            image_filename = image_url
    
    # 提取表格HTML数据
    table_html = None
    if 'table_html' in metadata_dict:
        table_html = metadata_dict['table_html']
        content_type = "表格内容"
    
    # 构建元数据模型
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
        image_filename=image_filename,
        table_html=table_html
    )


# API路由定义

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="文档查询API",
        version="1.0.0"
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    try:
        # 测试查询系统
        qs = get_query_system()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            service="文档查询API",
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务不可用: {e}")


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    文档查询接口（支持混合搜索）
    """
    start_time = datetime.now()
    
    try:
        # 获取查询系统
        qs = get_query_system()
        
        docs = qs.query(request.query_text, request.top_k)
        
        # 格式化结果
        results = []
        for i, doc in enumerate(docs):
            formatted_result = format_document_result(doc, i)
            results.append(formatted_result)
        
        # 根据是否有图片对结果进行重新排序，包含图片的排在前面
        results.sort(key=lambda x: not (x.has_image or bool(x.image_filename)))
        
        # 更新排序后的结果ID，确保它们按新的顺序排列
        for i, result in enumerate(results):
            result.id = i + 1
        
        # 如果开启 DeepSeek 增强回答，则在此调用新方法（不修改原 query()）
        answer_text = None
        answer_sources = None
        if getattr(request, "use_deepseek", False):
            try:
                print("使用DeepSeek增强回答...")
                answer_text, answer_sources = qs.generate_answer(request.query_text, docs, request.answer_top_k or 3)
            except Exception as e:
                # 不要让增强回答失败阻断基础检索结果；记录并继续返回普通结果
                print(f"generate_answer失败: {e}")
                answer_text = None
                answer_sources = None
        # fire-and-forget MLflow logging (minimal)
        try:
            params = {
                "top_k": request.top_k,
                "use_deepseek": bool(getattr(request, "use_deepseek", False)),
                "answer_top_k": request.answer_top_k,
                "prefer_diagrams": bool(getattr(request, "prefer_diagrams", False))  # 记录新参数
            }
            timings = {"total": processing_time} if 'processing_time' in locals() else None
            _mlflow_executor.submit(_mlflow_log_query_async, request.query_text, params, docs, answer_text, timings)
        except Exception as e:
            print(f"⚠️ 提交 MLflow 日志任务失败: {e}")
        
        # 计算处理时间
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return QueryResponse(
            success=True,
            message=f"查询成功，找到 {len(results)} 个相关文档",
            query=request.query_text,
            total_results=len(results),
            results=results,
            answer=answer_text,
            sources=answer_sources,
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
    """
    简单的GET查询接口
    
    Args:
        query: 查询文本
        top_k: 返回结果数量
        format: 输出格式
        
    Returns:
        QueryResponse: 查询结果
    """
    request = QueryRequest(
        query_text=query,
        top_k=top_k,
        output_format=format
    )
    
    return await query_documents(request)


@app.get("/api/formats")
async def get_supported_formats():
    """获取支持的输出格式"""
    return {
        "supported_formats": [
            {
                "value": "detailed",
                "label": "详细格式",
                "description": "包含完整内容预览、元数据和图片"
            },
            {
                "value": "summary", 
                "label": "简洁格式",
                "description": "简化的内容摘要"
            },
            {
                "value": "raw",
                "label": "原始格式",
                "description": "原始文档数据"
            }
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int):
    """
    根据ID获取特定文档
    
    Args:
        doc_id: 文档ID
        
    Returns:
        DocumentResult: 文档详情
    """
    try:
        # 这里可以实现根据ID获取文档的逻辑
        # 当前版本返回示例数据
        raise HTTPException(status_code=404, detail=f"文档ID {doc_id} 未找到")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档失败: {str(e)}")


# 异常处理
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
    print("启动文档查询API服务...")
    print("API文档地址: http://localhost:8000/api/docs")
    print("简单查询接口: http://localhost:8000/api/query/simple")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )