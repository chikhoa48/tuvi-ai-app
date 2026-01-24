import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import io
import time

# --- IMPORT LIBS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="Đại Sư Tử Vi - Siêu Tốc (Batch)", page_icon="⚡", layout="wide")
st.markdown("""<style>.main {background-color: #f0f2f6;}</style>""", unsafe_allow_html=True)

# --- 1. MODEL UTILS ---
def get_available_gemini_models(api_key):
    if not api_key: return ["Nhập Key trước"]
    try:
        genai.configure(api_key=api_key)
        return [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and "gemini" in m.name]
    except: return ["gemini-1.5-flash"]

# --- 2. XỬ LÝ SÁCH SCAN (BATCH PROCESSING) ---
def process_images_in_batches(images, api_key, batch_size=20):
    """
    Gửi 1 lúc nhiều ảnh (batch_size) cho Gemini đọc để tiết kiệm thời gian
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    full_text = ""
    
    # Chia danh sách ảnh thành các gói nhỏ (chunks)
    total_images = len(images)
    
    # Tạo thanh tiến trình
    progress_bar = st.progress(0, text="Đang khởi động bộ máy đọc thần tốc...")
    
    for i in range(0, total_images, batch_size):
        # Lấy ra 1 lô ảnh (ví dụ từ ảnh 0 đến 19)
        batch = images[i : i + batch_size]
        current_batch_num = (i // batch_size) + 1
        total_batches = (total_images + batch_size - 1) // batch_size
        
        progress_bar.progress((i / total_images), text=f"Đang đọc lô {current_batch_num}/{total_batches} (Trang {i+1} đến {min(i+batch_size, total_images)})...")
        
        # Tạo nội dung gửi đi: [Câu lệnh text, Ảnh 1, Ảnh 2, ..., Ảnh 20]
        content_message = [
            {"type": "text", "text": "Bạn là một thư ký đánh máy chuyên nghiệp. Nhiệm vụ của bạn là nhìn vào các trang sách đính kèm dưới đây và chép lại CHÍNH XÁC toàn bộ nội dung văn bản trong đó. Hãy chép liền mạch, không cần mô tả ảnh, chỉ lấy nội dung chữ."}
        ]
        
        # Thêm từng ảnh vào message
        for img in batch:
            content_message.append({"type": "image_url", "image_url": img})
            
        # Gửi đi 1 lần duy nhất cho cả lô
        try:
            msg = HumanMessage(content=content_message)
            res = llm.invoke([msg])
            full_text += res.content + "\n\n"
        except Exception as e:
            st.error(f"Lỗi khi đọc lô {current_batch_num}: {e}")
            # Nếu lỗi, thử chờ 2s rồi tiếp tục lô sau
            time.sleep(2)
            
    progress_bar.progress(1.0, text="Đã đọc xong toàn bộ sách!")
    time.sleep(1)
    progress_bar.empty()
    return full_text

def process_pdfs_smart(pdf_docs, api_key):
    all_text = ""
    status_box = st.status("Đang phân tích tài liệu...", expanded=True)
    
    for pdf in pdf_docs:
        status_box.write(f"📂 Đang kiểm tra file: {pdf.name}")
        
        # 1. Thử đọc Text trước (Nhanh nhất)
        try:
            pdf_reader = PdfReader(pdf)
            raw_text = ""
            for page in pdf_reader.pages:
                t = page.extract_text()
                if t: raw_text += t
        except:
            raw_text = ""

        # 2. Nếu ít chữ quá -> Chuyển sang chế độ Batch OCR (Scan)
        if len(raw_text) < 100:
            status_box.write(f"📸 File {pdf.name} là dạng SCAN. Đang chuyển đổi sang ảnh...")
            pdf.seek(0)
            # Chuyển PDF thành list các ảnh
            images = convert_from_bytes(pdf.read())
            status_box.write(f"✅ Đã tách thành {len(images)} trang ảnh. Bắt đầu đọc Batch...")
            
            # Gọi hàm xử lý hàng loạt
            ocr_text = process_images_in_batches(images, api_key, batch_size=20)
            all_text += ocr_text
        else:
            status_box.write(f"📝 File {pdf.name} là dạng văn bản. Đã đọc xong.")
            all_text += raw_text

    status_box.update(label="Hoàn tất!", state="complete", expanded=False)
    
    if not all_text: return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return text_splitter.split_text(all_text)

# --- 3. VECTOR STORE & RAG ---
@st.cache_resource
def get_vector_store(_text_chunks, api_key):
    if not _text_chunks: return None
    # Dùng embedding-001 cho ổn định
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_texts(texts=_text_chunks, embedding=embeddings)

def get_master_response(query, chart_data, vector_store, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    
    template = """
    Bạn là Đại Sư Tử Vi. Dựa vào kiến thức sách (Context) và Lá số để luận giải.
    Context: {context}
    Lá số: {question}
    Yêu cầu: Luận giải sâu sắc, có dẫn chứng từ sách.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )
    full_input = f"LÁ SỐ:\n{chart_data}\n\nCÂU HỎI:\n{query}"
    return qa_chain.invoke({"query": full_input})["result"]

# --- 4. VISION (ĐỌC LÁ SỐ) ---
def extract_chart_data(image, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    msg = HumanMessage(content=[
        {"type": "text", "text": "Trích xuất thông tin lá số tử vi: Ngày giờ, Mệnh, Thân, Cục, Các sao tại 12 cung. Trả về text."},
        {"type": "image_url", "image_url": image}
    ])
    return llm.invoke([msg]).content

# --- MAIN APP ---
def main():
    st.title("⚡ ĐẠI SƯ TỬ VI - BATCH OCR")
    
    with st.sidebar:
        api_key = st.text_input("Google API Key", type="password")
        if not api_key: st.stop()
        
        # Nạp Sách
        pdf_docs = st.file_uploader("Upload Sách (PDF/Scan)", accept_multiple_files=True)
        if st.button("Luyện Hóa (Batch Mode)"):
            with st.spinner("Đang xử lý..."):
                chunks = process_pdfs_smart(pdf_docs, api_key)
                if chunks:
                    st.session_state.vector_store = get_vector_store(chunks, api_key)
                    st.success(f"Đã nạp {len(chunks)} đoạn kiến thức!")
                else:
                    st.error("Không có nội dung!")

    # Main Interface
    if "chart_data" not in st.session_state: st.session_state.chart_data = None
    if "messages" not in st.session_state: st.session_state.messages = []

    # Upload Lá Số
    img = st.file_uploader("Ảnh lá số", type=['png','jpg'])
    if img and st.button("Đọc Lá Số"):
        with st.spinner("Đang đọc..."):
            # Lấy model tốt nhất từ list
            models = get_available_gemini_models(api_key)
            best_model = models[0] if models else "gemini-1.5-flash"
            st.session_state.chart_data = extract_chart_data(img, best_model, api_key)
            st.success("Đã xong!")

    # Chat
    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])
        
    if prompt := st.chat_input("Hỏi đại sư..."):
        if "vector_store" not in st.session_state: st.warning("Nạp sách trước!"); return
        
        st.session_state.messages.append({"role":"user", "content":prompt})
        st.chat_message("user").markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Đang suy ngẫm..."):
                # Dùng Gemini Pro cho câu trả lời thông minh
                models = get_available_gemini_models(api_key)
                # Tìm model nào có chữ Pro, nếu không thì dùng cái đầu tiên
                chat_model = next((m for m in models if "pro" in m), models[0])
                
                res = get_master_response(prompt, st.session_state.chart_data, st.session_state.vector_store, chat_model, api_key)
                st.markdown(res)
                st.session_state.messages.append({"role":"assistant", "content":res})

if __name__ == "__main__":
    main()
