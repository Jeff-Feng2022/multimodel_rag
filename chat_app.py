import streamlit as st
import requests
import base64
from io import BytesIO

# 安全地读取 secrets 中的 api_base；如果没有 secrets，回退到本地地址，避免 Streamlit 抛出异常
try:
    API_BASE = st.secrets["api_base"] if "api_base" in st.secrets else "http://localhost:8000"
except Exception:
    API_BASE = "http://localhost:8000"


def call_query_api(query_text: str, top_k: int = 3, output_format: str = "detailed"):
    url = f"{API_BASE}/api/query"
    payload = {
        "query_text": query_text,
        "top_k": top_k,
        "output_format": output_format
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"success": False, "message": f"Call to API failed: {e}"}


def render_result(res_item: dict):
    # 获取结果ID用于创建唯一key
    result_id = res_item.get('id', 'unknown')
    
    st.markdown(f"**Result #{result_id}**")
    
    # 检查是否有表格HTML数据
    table_html = res_item.get('table_html')
    content = res_item.get('content', '')
    
    # 如果有表格HTML，则显示表格；否则显示文本内容
    if table_html:
        st.markdown("**Table Content:**")
        # 添加CSS样式使表格边框更细
        styled_table_html = f"""
        <style>
        .styled-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9em;
        }}
        .styled-table th, .styled-table td {{
            border: 0.5px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}
        .styled-table th {{
            background-color: #f1f5f9;
            font-weight: 600;
        }}
        .styled-table tr:hover {{
            background-color: #f0f9ff;
        }}
        </style>
        <div class="styled-table">{table_html}</div>
        """
        st.components.v1.html(styled_table_html, height=400, scrolling=True)
    elif content:
        st.text_area("Content", value=content, height=200, key=f"content_{result_id}")
    
    metadata = res_item.get('metadata') or {}
    if metadata:
        st.write("**Metadata**")
        # st.json不支持key参数，所以我们直接使用它
        st.json(metadata)
    
    # 图片优先展示 image_filename（静态URL），否则尝试 image_data（base64）
    img_url = res_item.get('image_filename')
    img_b64 = res_item.get('image_data')
    
    # 调试信息
    # st.write(f"Debug - img_url: {img_url}, img_b64 length: {len(img_b64) if img_b64 else 0}")
    
    if img_url:
        # 如果是相对路径（以 /static 开头），把主机名补上
        if img_url.startswith("/"):
            full_url = API_BASE.rstrip("/") + img_url
        else:
            full_url = img_url
        try:
            # 用 HTML img 标签显示图片，兼容本地 static 路径
            st.markdown(
                f"<img src='{full_url}' alt='Retrieved image' style='max-width: 100%; height: auto;'>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.write(f"Could not load image from URL: {full_url}")
            st.write(f"Error: {str(e)}")
    elif img_b64:
        try:
            img_bytes = base64.b64decode(img_b64)
            st.image(BytesIO(img_bytes), caption="Retrieved image", key=f"image_b64_{result_id}")
        except Exception as e:
            st.write("Could not decode returned base64 image data")
            st.write(f"Error: {str(e)}")


def main():
    st.set_page_config(page_title="Multimodal RAG Chat", layout="wide")
    # Reduce top spacing and center title
    st.markdown(
        """
        <style>
        /* Reduce top padding of the main container */
        .block-container { padding-top: 8px; }
        /* Reduce default margin above h1 so title is closer to top */
        h1 { margin-top: 8px !important; margin-bottom: 8px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Centered title (kept as HTML for centering)
    st.markdown("<h1 style='text-align: center;'>Multimodal RAG Chatbot</h1>", unsafe_allow_html=True)

    with st.sidebar:
        # Provide a compact label and a collapsible settings area
        st.markdown("### Controls")
        with st.expander("Settings", expanded=True):
            top_k = st.slider("Results (top_k)", min_value=1, max_value=10, value=3)
            output_format = st.selectbox("Output format", ["detailed", "summary", "raw"], index=0)
            st.markdown("---")
            st.markdown("API settings (optional)")
            api_input = st.text_input("API base URL", value=API_BASE)
            if st.button("Save API base"):
                st.experimental_set_query_params(api_base=api_input)
                st.success("Saved for this session. To persist, use `secrets.toml` or env variables.")

    st.write("Enter your query in the box (multi-line supported), then click 'Send'.")
    # Place the query text area and the Send button on the same row
    cols = st.columns([8, 1])
    with cols[0]:
        # Larger multi-line input for longer queries or context; placed in left column
        query_text = st.text_area("Query text", value="", height=140, key="query_input")
    with cols[1]:
        # Button in the right column so it stays on the same row
        # Add a small spacer above the button so it appears vertically centered
        text_area_height = 140  # px, must match the height used for text_area
        approx_button_height = 36  # approximate button height in px
        padding_top = max(0, (text_area_height - approx_button_height) // 2)
        st.markdown(f"<div style='height:{padding_top}px'></div>", unsafe_allow_html=True)
        send = st.button("Send")

    if send and query_text.strip():
        with st.spinner("Querying, please wait..."):
            resp = call_query_api(query_text, top_k=top_k, output_format=output_format)

        if not resp or not resp.get('success'):
            st.error(resp.get('message', 'Unknown error'))
            return

        total = resp.get('total_results', 0)
        st.success(f"Query completed, found {total} results (processing time: {resp.get('processing_time')}s)")

        for item in resp.get('results', []):
            render_result(item)


if __name__ == "__main__":
    main()