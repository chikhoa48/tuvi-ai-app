import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader

# --- IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# Sử dụng RetrievalQA cho ổn định, tránh lỗi import version
from langchain.chains import RetrievalQA 
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from PIL import Image

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Đại Sư Tử Vi - AI Tổng Hợp", page_icon="⛩️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #fdfbf7;}
    h1, h2, h3 {font-family: 'Times New Roman', serif; color: #5a1e1e;}
    .stChatInput {position: fixed; bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# --- 1. HÀM TỰ ĐỘNG LẤY MODEL MỚI NHẤT ---
def get_available_gemini_models(api_key):
    if not api_key: return ["Chưa nhập API Key"]
    try:
        genai.configure(api_key=api_key)
        models = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and "gemini" in m.name]
        models.sort(reverse=True)
        return models
    except: return ["gemini-1.5-pro"]

# --- 2. XỬ LÝ SÁCH (RAG) ---
@st.cache_resource
def get_vector_store(_text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=_text_chunks, embedding=embeddings)
    return vectorstore

def process_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t: text += t
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_text(text)

# --- 3. VISION & DATA EXTRACTION ---
def extract_chart_data(image, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    msg = HumanMessage(content=[
        {"type": "text", "text": "Hãy trích xuất thông tin lá số tử vi từ ảnh này: Ngày giờ sinh, Mệnh, Thân, Cục, Vị trí 12 cung và các sao. Trả về dạng text rõ ràng."},
        {"type": "image_url", "image_url": image}
    ])
    res = llm.invoke([msg])
    return res.content

# --- 4. LOGIC ĐẠI SƯ (DÙNG RETRIEVAL QA - SIÊU ỔN ĐỊNH) ---
def get_master_response(query, chart_data, vector_store, model_name, api_key):
    
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.5)

    # Prompt Template tùy chỉnh
    template = """
    Bạn là "Bạch Vân Cư Sĩ" - Đại sư Tử Vi. Hãy dùng kiến thức trong phần Context để luận giải câu hỏi.
    
    Kiến thức từ sách (Context):
    {context}

    Thông tin & Câu hỏi:
    {question}

    Yêu cầu luận giải:
    1. Đối chiếu kiến thức sách với thông tin lá số.
    2. Tổng hợp, quy nạp để đưa ra kết luận sâu sắc, có tính triết lý.
    3. Trả lời chi tiết, giọng văn cổ trang, uy nghiêm.
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Sử dụng RetrievalQA thay vì create_retrieval_chain (để tránh lỗi import)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Gộp thông tin lá số và câu hỏi vào input
    full_query = f"THÔNG TIN LÁ SỐ:\n{chart_data}\n\nCÂU HỎI CỦA ĐƯƠNG SỐ:\n{query}"
    
    result = qa_chain.invoke({"query": full_query})
    return result["result"]

# --- GIAO DIỆN CHÍNH ---
def main():
    st.title("⛩️ THIÊN CƠ CÁC - V3 (STABLE)")

    with st.sidebar:
        st.header("Cấu hình")
        api_key = st.text_input("Nhập Google AI Key", type="password")
        
        if api_key:
            models = get_available_gemini_models(api_key)
            selected_model = st.selectbox("Chọn Model:", models)
        else:
            selected_model = "gemini-1.5-pro"

        pdf_docs = st.file_uploader("Upload sách (.pdf)", accept_multiple_files=True)
        if st.button("Nạp Kiến Thức") and pdf_docs and api_key:
            with st.spinner("Đang học..."):
                chunks = process_pdfs(pdf_docs)
                st.session_state.vector_store = get_vector_store(chunks, api_key)
                st.success("Xong!")

    # Main Area
    if "chart_data" not in st.session_state: st.session_state.chart_data = None
    if "messages" not in st.session_state: st.session_state.messages = []

    uploaded_img = st.file_uploader("Upload ảnh lá số", type=['png', 'jpg', 'jpeg'])
    if uploaded_img and st.button("Đọc lá số"):
        if api_key:
            with st.spinner("Đang đọc..."):
                st.session_state.chart_data = extract_chart_data(uploaded_img, selected_model, api_key)
                st.success("Đã đọc xong lá số!")

    # Chat Interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Hỏi Đại sư..."):
        if not api_key or "vector_store" not in st.session_state or not st.session_state.chart_data:
            st.warning("Vui lòng nhập Key, Nạp sách và Upload ảnh trước!")
        else:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Đại sư đang bấm độn..."):
                    response = get_master_response(prompt, st.session_state.chart_data, st.session_state.vector_store, selected_model, api_key)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
