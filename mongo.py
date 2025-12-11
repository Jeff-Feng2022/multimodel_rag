import pymongo
from typing import List, Any, Dict, Optional, Union, Iterator
import json
import uuid
from datetime import datetime
import traceback
from langchain_core.stores import BaseStore

class MongoDocStore(BaseStore):
    """MongoDB文档存储类，替代unstructured的InMemoryStore"""
    
    def __init__(self, 
                 connection_string: str = "mongodb://localhost:27017/",
                 database_name: str = "transformers",
                 collection_name: str = "documents"):
        """
        初始化MongoDB文档存储
        
        Args:
            connection_string: MongoDB连接字符串
            database_name: 数据库名称
            collection_name: 集合名称
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        
        # 使用同步客户端避免异步事件循环冲突
        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # 创建索引
        self._create_sync_indexes()
        
        print(f"✅ MongoDB文档存储初始化完成")
        print(f"📊 数据库: {database_name}")
        print(f"📁 集合: {collection_name}")
    
    def _create_sync_indexes(self):
        """创建必要的索引"""
        try:
            # 创建doc_id索引
            self.collection.create_index("doc_id", unique=True)
            # 创建创建时间索引
            self.collection.create_index("created_at")
            print("✅ MongoDB索引创建完成")
        except Exception as e:
            print(f"⚠️ 创建索引时出现警告: {e}")
    
    def test_connection(self) -> bool:
        """测试MongoDB连接"""
        try:
            self.client.admin.command('ping')
            print("✅ MongoDB连接成功")
            return True
        except Exception as e:
            print(f"❌ MongoDB连接失败: {e}")
            return False
    
    def set(self, key: str, value: Any) -> None:
        """存储单个键值对"""
        self.mset([(key, value)])
    
    def mset(self, key_value_pairs: List[tuple], **kwargs) -> None:
        """
        批量存储键值对到MongoDB
        
        Args:
            key_value_pairs: [(key1, value1), (key2, value2), ...]
        """
        try:
            # 准备批量插入文档
            documents = []
            for key, value in key_value_pairs:
                # 将unstructured的文档对象转换为可序列化的格式
                doc_data = self._serialize_document(value)
                doc_dict = {
                    "_id": key,  # 使用key作为文档ID
                    "doc_id": key,
                    "doc_data": doc_data,  # 存储序列化后的文档数据
                    "created_at": datetime.now()
                }
                documents.append(doc_dict)
            
            # 批量插入到MongoDB
            if documents:
                try:
                    result = self.collection.insert_many(documents, ordered=False)
                    inserted_count = len(result.inserted_ids)
                except pymongo.errors.BulkWriteError as e:
                    # 计算成功插入的文档数量
                    inserted_count = len(documents) - len(e.details.get('writeErrors', []))
                    print(f"⚠️ 部分文档插入失败，但成功插入了 {inserted_count} 个文档")
                
                print(f"✅ 成功存储 {inserted_count} 个文档到MongoDB")
            
        except Exception as e:
            print(f"❌ 存储文档到MongoDB失败: {e}")
            raise
    
    def get(self, key: str) -> Any:
        """获取单个键对应的值"""
        result = self.mget([key])
        return result[0] if result else None
    
    def mget(self, keys: List[str], **kwargs) -> List[Any]:
        """
        批量获取文档
        
        Args:
            keys: 要获取的文档键列表
            
        Returns:
            对应的文档对象列表
        """
        try:
            if not keys:
                return []
            
            # 查询文档
            cursor = self.collection.find({"doc_id": {"$in": keys}})
            docs = list(cursor)
            
            # 构建键值映射
            doc_dict = {}
            for doc in docs:
                if doc.get("doc_data"):
                    # 从JSON字符串反序列化文档对象
                    doc_obj = self._deserialize_document(doc)
                    doc_dict[doc["doc_id"]] = doc_obj
            
            # 按原始顺序返回
            result = [doc_dict.get(key) for key in keys]
            found_count = sum(1 for r in result if r is not None)
            print(f"✅ 从MongoDB检索到 {found_count}/{len(keys)} 个文档")
            
            return result
                
        except Exception as e:
            print(f"❌ 从MongoDB获取文档失败: {e}")
            traceback.print_exc()
            return [None] * len(keys)
    
    def _serialize_document(self, doc) -> Dict[str, Any]:
        """将文档对象序列化为MongoDB可存储的格式"""
        try:
            # 如果已经是字典，直接返回（包含元数据）
            if isinstance(doc, dict):
                # 确保包含必要的字段，包括image_base64
                result = {
                    "page_content": doc.get("page_content", ""),
                    "metadata": doc.get("metadata", {}),
                    "doc_type": doc.get("doc_type", "dict")
                }
                
                # 如果文档包含image_base64字段，保留它
                if "image_base64" in doc and doc["image_base64"]:
                    result["image_base64"] = doc["image_base64"]
                
                # 如果文档包含其他图片相关字段，也保留
                for key in ["orig_elements", "raw_doc_data", "text"]:
                    if key in doc and doc[key]:
                        result[key] = doc[key]
                
                return result
            
            # 处理unstructured的文档对象
            if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                # LangChain Document对象
                result = {
                    "page_content": getattr(doc, 'page_content', ''),
                    "metadata": getattr(doc, 'metadata', {}),
                    "doc_type": "langchain_document"
                }
                return result
            elif hasattr(doc, 'to_dict'):
                # 其他有to_dict方法的文档对象
                try:
                    result = doc.to_dict()
                    result["doc_type"] = "unstructured_element"
                    return result
                except Exception as e:
                    print(f"to_dict()序列化失败: {e}")
            
            # 兜底：将对象转换为字符串
            return {
                "page_content": str(doc),
                "metadata": {"type": type(doc).__name__},
                "doc_type": "fallback"
            }
            
        except Exception as e:
            print(f"序列化文档时出错: {e}")
            return {
                "page_content": str(doc),
                "metadata": {"type": type(doc).__name__, "error": str(e)},
                "doc_type": "error"
            }
    
    def _deserialize_document(self, doc_dict: Dict[str, Any]) -> Any:
        """从MongoDB检索的字典反序列化回原始文档对象"""
        try:
            # 保持原始格式，返回完整的doc_data内容，但保留原始doc_id信息
            if "doc_data" in doc_dict:
                doc_data = doc_dict["doc_data"]
                # 如果doc_data是字符串，尝试解析为JSON
                if isinstance(doc_data, str):
                    import json
                    try:
                        doc_data = json.loads(doc_data)
                    except:
                        # 如果解析失败，返回原始字符串
                        return doc_data
                
                # 如果doc_data是字典，返回完整内容，但保留原始doc_id
                if isinstance(doc_data, dict):
                    # 优先使用text字段，然后是page_content字段
                    content = ""
                    if "text" in doc_data and doc_data["text"]:
                        content = doc_data["text"]
                    elif "page_content" in doc_data and doc_data["page_content"]:
                        content = doc_data["page_content"]
                    
                    # 构造返回结果，确保包含图片base64数据，并保留原始doc_id
                    result = {
                        "page_content": content,
                        "metadata": doc_data.get("metadata", {}),
                        "doc_type": doc_data.get("doc_type", "unknown"),
                        "raw_doc_data": doc_data,  # 保留原始完整数据
                        # 保留原始MongoDB文档的doc_id信息
                        "doc_id": doc_dict.get("doc_id", ""),  # 外层doc_id
                        "_id": str(doc_dict.get("_id", ""))  # MongoDB的_id
                    }
                    
                    # 如果包含图片base64数据，确保返回
                    if "image_base64" in doc_data and doc_data["image_base64"]:
                        result["image_base64"] = doc_data["image_base64"]
                    
                    # 打印调试信息
                    print(f"🔍 反序列化文档，保留原始doc_id: {doc_dict.get('doc_id', '未知')}")
                    
                    return result
                
                # 其他情况返回原始数据，但仍保留doc_id
                return {
                    "doc_id": doc_dict.get("doc_id", ""),
                    "_id": str(doc_dict.get("_id", "")),
                    "data": doc_data
                }
            
            # 兜底：返回基本字段，但仍保留doc_id信息
            result = {
                "page_content": doc_dict.get("page_content", ""),
                "metadata": doc_dict.get("metadata", {}),
                "doc_type": doc_dict.get("doc_type", "unknown"),
                # 保留原始doc_id信息
                "doc_id": doc_dict.get("doc_id", ""),
                "_id": str(doc_dict.get("_id", ""))
            }
            
            # 如果包含图片base64数据，确保返回
            if "image_base64" in doc_dict and doc_dict["image_base64"]:
                result["image_base64"] = doc_dict["image_base64"]
            
            return result
        except Exception as e:
            print(f"反序列化文档时出错: {e}")
            return {
                "page_content": str(doc_dict), 
                "metadata": {}, 
                "doc_type": "error",
                "doc_id": doc_dict.get("doc_id", ""),
                "_id": str(doc_dict.get("_id", ""))
            }
    
    def delete(self, key: str) -> bool:
        """删除指定的键"""
        try:
            if self.collection is None:
                raise Exception("未初始化MongoDB连接")
            
            result = self.collection.delete_one({"doc_id": key})
            return result.deleted_count > 0
        except Exception as e:
            print(f"删除文档时出错: {e}")
            traceback.print_exc()
            return False
    
    def mdelete(self, keys: List[str]) -> None:
        """批量删除键"""
        try:
            if not keys:
                return
            
            if self.collection is None:
                raise Exception("未初始化MongoDB连接")
            
            result = self.collection.delete_many({"doc_id": {"$in": keys}})
            print(f"✅ 批量删除了 {result.deleted_count} 个文档")
        except Exception as e:
            print(f"批量删除文档时出错: {e}")
            traceback.print_exc()
    
    def yield_keys(self, prefix: str = None) -> Iterator[str]:
        """迭代器，返回所有键"""
        try:
            if self.collection is None:
                raise Exception("未初始化MongoDB连接")
            
            query = {}
            if prefix:
                query["doc_id"] = {"$regex": f"^{prefix}"}
            
            cursor = self.collection.find(query, {"doc_id": 1})
            for doc in cursor:
                yield doc["doc_id"]
        except Exception as e:
            print(f"获取键列表时出错: {e}")
            traceback.print_exc()

    def clear_collection(self):
        """清空集合（用于测试）"""
        try:
            result = self.collection.delete_many({})
            print(f"✅ 清空集合，删除了 {result.deleted_count} 个文档")
        except Exception as e:
            print(f"❌ 清空集合失败: {e}")
    
    def close(self):
        """关闭MongoDB连接"""
        try:
            self.client.close()
            print("✅ MongoDB连接已关闭")
        except Exception as e:
            print(f"⚠️ 关闭MongoDB连接时出错: {e}")

# 全局变量用于存储实例
_mongo_doc_store = None

def get_mongo_doc_store() -> MongoDocStore:
    """获取或创建MongoDB文档存储实例"""
    global _mongo_doc_store
    if _mongo_doc_store is None:
        _mongo_doc_store = MongoDocStore()
    return _mongo_doc_store
 