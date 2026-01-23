import streamlit as st
import os
import time
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Đại Sư Tử Vi - AI Tổng Hợp", page_icon="⛩️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #fdfbf7;}
    h1, h2, h3 {font-family: 'Times New Roman', serif; color: #5a1e1e;}
    .stChatInput {position: fixed; bottom: 20px;}
    .report-card {
        padding: 20px; border-radius: 10px; background-color: white;
        border-left: 5px solid #8B0000; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px; font-family: 'Times New Roman', serif; font-size: 1.1em;
    }
    .reasoning-box {
        font-size: 0.9em; color: #666; font-style: italic; 
        background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. HÀM TỰ ĐỘNG LẤY MODEL MỚI NHẤT ---
def get_available_gemini_models(api_key):
    """Quét API Google để lấy danh sách model thực tế đang khả dụng"""
    if not api_key:
        return ["Chưa nhập API Key"]
    
    try:
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            # Lọc lấy các model có khả năng tạo nội dung (generateContent)
            if 'generateContent' in m.supported_generation_methods:
                # Ưu tiên các model Gemini
                if "gemini" in m.name:
                    models.append(m.name.replace("models/", ""))
        
        # Sắp xếp để model pro/mới nhất lên đầu (tùy logic)
        models.sort(reverse=True)
        return models
    except Exception as e:
        return [f"Lỗi: {str(e)}"]

# --- 2. XỬ LÝ SÁCH (RAG) ---
@st.cache_resource
def get_vector_store(_text_chunks, api_key):
    # Dùng hàm cache để không phải load lại khi đổi model
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
    
    # Chia nhỏ văn bản để tra cứu chi tiết
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    return text_splitter.split_text(text)

# --- 3. VISION & DATA EXTRACTION ---
def extract_chart_data(image, model_name, api_key):
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    msg = HumanMessage(content=[
        {"type": "text", "text": "Bạn là chuyên gia số hóa. Hãy nhìn ảnh lá số tử vi này và trích xuất lại TOÀN BỘ thông tin: Ngày giờ sinh, Âm dương nam/nữ, Cục, Mệnh, Thân, vị trí 12 cung và các sao trong từng cung. Trả về dạng văn bản có cấu trúc rõ ràng."},
        {"type": "image_url", "image_url": image}
    ])
    res = llm.invoke([msg])
    return res.content

# --- 4. LOGIC ĐẠI SƯ (REASONING CHAIN) ---
def get_master_response(query, chart_data, vector_store, model_name, api_key, history):
    
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.5)

    # Prompt Quy nạp & Tổng hợp kiến thức
    system_prompt = """
    Bạn là "Bạch Vân Cư Sĩ" - một bậc thầy Tử Vi Đẩu Số, người kết hợp tinh hoa của nhiều trường phái.
    
    NHIỆM VỤ CỦA BẠN:
    Luận giải câu hỏi của người dùng dựa trên:
    1. Thông tin lá số (được cung cấp bên dưới).
    2. Kiến thức từ các sách tử vi (được cung cấp trong phần Context).
    
    QUY TRÌNH SUY LUẬN (BẮT BUỘC):
    Bước 1 - Đối chiếu: Tìm kiếm xem các cuốn sách khác nhau nói gì về vấn đề này (Ví dụ: Sách A nói sao này tốt, nhưng sách B nói xấu khi gặp sao kia).
    Bước 2 - Phân tích Cục diện: Xem xét ngũ hành, âm dương, vị trí đắc hãm để xem ý kiến nào trong sách là phù hợp nhất với lá số này.
    Bước 3 - Tổng hợp (Quy nạp): Đừng chỉ trích dẫn. Hãy kết hợp các ý kiến để đưa ra lời luận đoán cuối cùng của riêng bạn.
    
    PHONG CÁCH:
    - Lời văn thâm trầm, sâu sắc, có tính triết lý.
    - Luôn giải thích lý do: "Sách Tử Vi Hàm Số cho rằng..., tuy nhiên trong trường hợp này Mệnh bạn có Tuần Không nên..."
    - Tránh máy móc. Nếu sách không có thông tin, hãy dùng kiến thức nền tảng của bạn để suy luận.

    Thông tin lá số của đương số:
    {chart_data}

    Kiến thức tham khảo từ sách (Context):
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 7}) # Lấy nhiều đoạn văn bản hơn để tổng hợp
    rag_chain = create_retrieval_chain(retriever, chain)

    response = rag_chain.invoke({
        "input": query,
        "chart_data": chart_data,
        "chat_history": history
    })
    
    return response["answer"]

# --- GIAO DIỆN CHÍNH ---
def main():
    st.title("⛩️ THIÊN CƠ CÁC - V3")
    st.caption("Phiên bản Đại Sư AI: Tự động cập nhật Model & Tư duy quy nạp đa nguồn sách")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔑 Chìa khóa & Tàng thư")
        api_key = st.text_input("Nhập Google AI Key", type="password")
        
        # --- AUTO UPDATE MODEL SELECTOR ---
        if api_key:
            st.success("Đã kết nối Google AI!")
            available_models = get_available_gemini_models(api_key)
            selected_model = st.selectbox("Chọn 'Linh Hồn' (Model) cho Đại Sư:", available_models, index=0)
            if "gemini-1.5-pro" in selected_model or "gemini-2" in selected_model:
                st.info("💡 Model này có khả năng suy luận mạnh mẽ nhất.")
        else:
            selected_model = "gemini-1.5-pro" # Default ảo
            st.warning("Vui lòng nhập API Key để tải danh sách Model mới nhất.")

        st.divider()
        st.subheader("📚 Nạp Kiến Thức (Sách)")
        pdf_docs = st.file_uploader("Upload sách (.pdf)", accept_multiple_files=True)
        
        if st.button("Luyện Hóa Kiến Thức"):
            if not pdf_docs or not api_key:
                st.error("Thiếu nguyên liệu!")
            else:
                with st.spinner("Đang đọc và đối chiếu các sách..."):
                    chunks = process_pdfs(pdf_docs)
                    st.session_state.vector_store = get_vector_store(chunks, api_key)
                    st.success(f"Đã hấp thụ {len(chunks)} đơn vị kiến thức!")

    # --- MAIN AREA ---
    
    # 1. Upload & Phân tích ảnh (Chỉ làm 1 lần)
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = None

    uploaded_img = st.file_uploader("Bước 1: Tải ảnh lá số lên để Đại sư xem qua", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_img and not st.session_state.chart_data:
        if st.button("Trích xuất thông tin lá số"):
            if not api_key: st.error("Cần API Key."); return
            with st.spinner("Đang quan sát tinh bàn..."):
                st.image(uploaded_img, width=300)
                # Dùng model vision đọc ảnh
                data = extract_chart_data(uploaded_img, selected_model, api_key)
                st.session_state.chart_data = data
                st.success("Đã nắm rõ cách cục lá số!")
                with st.expander("Xem thông tin thô (Debug)"):
                    st.write(data)

    # 2. Khu vực Trò chuyện / Luận đoán
    if st.session_state.chart_data:
        st.divider()
        st.subheader("🔮 Đối thoại cùng Đại Sư")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Hiển thị lịch sử chat
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        # Input người dùng
        user_query = st.chat_input("Hỏi Đại sư (VD: 'Luận cung Tài Bạch của tôi?', 'Năm nay vận hạn ra sao?')")
        
        if user_query:
            if "vector_store" not in st.session_state:
                st.error("Đại sư chưa được học sách (Chưa upload sách bên trái)!")
            else:
                # Hiển thị câu hỏi
                st.chat_message("user").markdown(user_query)
                st.session_state.messages.append(HumanMessage(content=user_query))
                
                # AI xử lý
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Hiển thị trạng thái "Suy nghĩ"
                    with st.status("Đang tra cứu và quy nạp kiến thức...", expanded=True) as status:
                        st.write("🔍 Đang tìm các đoạn liên quan trong sách...")
                        st.write("⚖️ Đang so sánh các thuyết khác nhau...")
                        st.write("✍️ Đang tổng hợp lời luận...")
                        
                        # Gọi hàm xử lý chính
                        response_text = get_master_response(
                            user_query,
                            st.session_state.chart_data,
                            st.session_state.vector_store,
                            selected_model,
                            api_key,
                            st.session_state.messages
                        )
                        status.update(label="Đã luận giải xong!", state="complete", expanded=False)
                    
                    message_placeholder.markdown(response_text)
                    st.session_state.messages.append(AIMessage(content=response_text))

if __name__ == "__main__":
    main()
