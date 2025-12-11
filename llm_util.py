import os
import base64
import requests
import json


def img_to_desc(base64_image)->str:
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
    # 获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        raise RuntimeError("请设置DASHSCOPE_API_KEY或QWEN_API_KEY环境变量")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # 代理设置
    proxies = {
        "http": None,
        "https": None,
    }
    
    # 如果用户设置了代理，使用系统代理设置
    if os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY"):
        proxies = None
    
    # 简化的prompt，更自然的表达
    prompt = (
        '这张图片来自RAG相关的研究论文，请用英文详细描述图片内容，重点关注：\n'
        '1) 图表、架构、公式等技术元素\n'
        '2) 与RAG系统、知识图谱的关系\n'
        '3) 重要的技术细节如节点关系、数据流\n'
        '4) 图片传达的核心信息\n'
        '请用简洁的列表格式，适合向量检索'
    )
    
    payload = {
        "model": "qwen-vl-max",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{base64_image}"
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "parameters": {"max_tokens": 1500}
        }
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), proxies=proxies, timeout=30)
        result = response.json()
        
        if "output" in result and "choices" in result["output"]:
            description = result["output"]["choices"][0]["message"]["content"]
        else:
            description = f"Exception: {result}"
            
    except requests.exceptions.ProxyError as e:
        print(f"Proxy error connecting to DashScope API: {e}")
        print("Try setting HTTP_PROXY and HTTPS_PROXY environment variables or disable proxy if not needed.")
        description = "Error: Could not connect to image analysis service due to proxy issues."
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error connecting to DashScope API: {e}")
        description = "Error: Could not connect to image analysis service."
    except requests.exceptions.Timeout as e:
        print(f"Timeout connecting to DashScope API: {e}")
        description = "Error: Connection to image analysis service timed out."
    except Exception as e:
        print(f"Unexpected error calling DashScope API: {e}")
        description = f"Error: Unexpected error calling image analysis service: {str(e)}"
    print(description)  
    return description