import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import time

# --- IMPORT LIBS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# --- CONFIG ---
st.set_page_config(page_title="Đại Sư Tử Vi - V6 (Anti-Crash)", page_icon="🛡️", layout="wide")
st.markdown("""<style>.main {background-color: #f4f6f9;}</style>""", unsafe_allow_html=True)

# --- 1. MODEL UTILS ---
def get_available_gemini_models(api_key):
    if not api_key: return ["Nhập Key trước"]
    try:
        genai.configure(api_key=api_key)
        return [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and "gemini" in m.name]
    except: return ["gemini-1.5-flash"]

# --- 2. XỬ LÝ SÁCH AN TOÀN (ANTI-CRASH) ---
def ocr_batch_safe(pdf_bytes, api_key, start_page, end_page):
    """Chỉ convert và đọc một nhóm nhỏ trang để không nổ RAM"""
    try:
        # Chỉ chuyển đổi đúng số trang cần thiết (Ví dụ: từ trang 1 đến 10)
        images = convert_from_bytes(pdf_bytes, first_page=start_page, last_page=end_page)
        
        if not images: return ""

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        
        content_message = [
            {"type": "text", "text": "Chép lại chính xác nội dung văn bản trong các trang sách này. Chỉ lấy nội dung chữ."}
        ]
        for img in images:
            content_message.append({"type": "image_url", "image_url": img})
            
        res = llm.invoke([HumanMessage(content=content_message)])
        
        # Xóa ảnh khỏi bộ nhớ ngay lập tức
        del images
        return res.content
    except Exception as e:
        return ""

def process_pdfs_smart(pdf_docs, api_key):
    all_text = ""
    status_box = st.status("Đang phân tích...", expanded=True)
    
    for pdf in pdf_docs:
        file_name = pdf.name
        status_box.write(f"📂 Đang kiểm tra: {file_name}")
        
        # Đọc file vào bộ nhớ đệm
        pdf_bytes = pdf.read()
        
        # 1. Thử đọc Text trước (Nhanh)
        try:
            pdf_reader = PdfReader(pdf)
            raw_text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t: raw_text += t
        except:
            raw_text = ""

        # 2. Nếu là Sách Scan -> Dùng chế độ 'Cuốn Chiếu' (Safe Mode)
        if len(raw_text) < 100:
            status_box.write(f"📸 {file_name} là Sách Scan. Đang kích hoạt chế độ Tiết Kiệm RAM...")
            
            try:
                # Lấy tổng số trang mà không cần convert ảnh (Nhẹ)
                info = pdfinfo_from_bytes(pdf_bytes)
                total_pages = info["Pages"]
                
                # Chia nhỏ: Mỗi lần chỉ làm 10 trang
                CHUNK_SIZE = 10 
                ocr_full_text = ""
                
                prog_bar = status_box.progress(0, text=f"Đang đọc {file_name}...")
                
                for start in range(1, total_pages + 1, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE - 1, total_pages)
                    
                    # Gọi hàm đọc từng phần nhỏ
                    chunk_text = ocr_batch_safe(pdf_bytes, api_key, start, end)
                    ocr_full_text += chunk_text + "\n"
                    
                    # Cập nhật tiến độ
                    prog_bar.progress(end/total_pages, text=f"Đã đọc xong trang {end}/{total_pages}...")
                    time.sleep(1) # Nghỉ 1 xíu để giải phóng RAM
                
                all_text += ocr_full_text
                status_box.write(f"✅ Đã xử lý xong {total_pages} trang scan.")
                
            except Exception as e:
                status_box.write(f"⚠️ Lỗi đọc file {file_name}: {e}. Hãy kiểm tra file packages.txt")
        else:
            status_box.write(f"📝 {file_name} là văn bản thường. Đã đọc xong.")
            all_text += raw_text

    status_box.update(label="Hoàn tất!", state="complete", expanded=False)
    
    if not all_text: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_text(all_text)

# --- 3. CORE LOGIC ---
@st.cache_resource
def get_vector_store(_text_chunks, api_key):
    if not _text_chunks: return None
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_texts(texts=_text_chunks, embedding=embeddings)

def get_master_response(query, chart_data, vector_store, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    template = "Bạn là Đại Sư Tử Vi. Dựa vào Context và Lá số để luận giải.\nContext: {context}\nLá số: {question}\nYêu cầu: Luận giải chi tiết."
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )
    return qa_chain.invoke({"query": f"LÁ SỐ:\n{chart_data}\n\nCÂU HỎI:\n{query}"})["result"]

def extract_chart_data(image, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    return llm.invoke([HumanMessage(content=[{"type":"text","text":"Trích xuất thông tin lá số tử vi thành văn bản."},{"type":"image_url","image_url":image}])]).content

# --- MAIN ---
def main():
    st.title("🛡️ ĐẠI SƯ TỬ VI - V6 (ANTI-CRASH)")
    
    with st.sidebar:
        api_key = st.text_input("Google API Key", type="password")
        if not api_key:
            st.info("👈 Nhập Key để bắt đầu")
            st.stop()
            
        pdf_docs = st.file_uploader("Upload Sách", accept_multiple_files=True)
        if st.button("Luyện Hóa"):
            if not pdf_docs: st.warning("Chưa chọn sách!"); st.stop()
            
            with st.spinner("Đang khởi động..."):
                try:
                    chunks = process_pdfs_smart(pdf_docs, api_key)
                    if chunks:
                        st.session_state.vector_store = get_vector_store(chunks, api_key)
                        st.success("Thành công!")
                    else:
                        st.error("Không đọc được nội dung.")
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {e}. Vui lòng thử file nhỏ hơn hoặc kiểm tra packages.txt")

    if "chart_data" not in st.session_state: st.session_state.chart_data = None
    if "messages" not in st.session_state: st.session_state.messages = []

    img = st.file_uploader("Ảnh lá số", type=['png','jpg'])
    if img and st.button("Đọc Lá Số"):
        st.session_state.chart_data = extract_chart_data(img, "gemini-1.5-flash", api_key)
        st.success("Đã xong!")

    for m in st.session_state.messages: st.chat_message(m["role"]).markdown(m["content"])
    
    if prompt := st.chat_input("Hỏi đại sư..."):
        if "vector_store" not in st.session_state: st.warning("Nạp sách trước!"); st.stop()
        st.session_state.messages.append({"role":"user", "content":prompt})
        st.chat_message("user").markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Đang suy luận..."):
                res = get_master_response(prompt, st.session_state.chart_data, st.session_state.vector_store, "gemini-1.5-pro", api_key)
                st.markdown(res)
                st.session_state.messages.append({"role":"assistant", "content":res})

if __name__ == "__main__":
    main()
