import streamlit as st
import google.generativeai as genai
from lunardate import LunarDate
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Thiên Cơ Các - Tử Vi & Bát Tự AI", page_icon="☯️", layout="wide")

# --- CSS GIAO DIỆN ---
st.markdown("""
<style>
    .tuvi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; background-color: #fce4ec; padding: 10px; border: 2px solid #880e4f; }
    .cung { background-color: white; border: 1px solid #ddd; padding: 10px; min-height: 150px; font-size: 14px; }
    .cung-header { font-weight: bold; color: #b71c1c; text-align: center; border-bottom: 1px solid #eee; margin-bottom: 5px; }
    .center-box { grid-column: 2 / 4; grid-row: 2 / 4; background-color: #fff3e0; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; border: 2px double #e65100; }
    .stChatMessage { background-color: #f1f8e9; }
</style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_data_context" not in st.session_state:
    st.session_state.user_data_context = ""
if "has_run" not in st.session_state:
    st.session_state.has_run = False

# --- SIDEBAR: CẤU HÌNH & API KEY ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    
    # 1. Xử lý API Key (Ưu tiên lấy từ Secrets, nếu không có thì nhập tay)
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        st.success("✅ Đã kết nối API Key hệ thống")
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Nhập Gemini API Key", type="password", help="Nhập xong nhớ nhấn Enter")
        if not api_key:
            st.warning("⚠️ Chưa nhập API Key")
    
    st.divider()
    
    # 2. Chọn phiên bản AI
    st.subheader("🧠 Chọn Trí Tuệ AI")
    model_option = st.selectbox(
        "Phiên bản Gemini:",
        ("gemini-1.5-flash", "gemini-1.5-pro"),
        index=0,
        help="Flash: Nhanh, mượt. Pro: Suy luận sâu sắc hơn nhưng chậm hơn."
    )
    
    st.info(f"Đang dùng: {model_option}")

# --- HÀM XỬ LÝ LỊCH ---
CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]

def get_can_chi(year):
    return f"{CAN[(year + 6) % 10]} {CHI[(year + 8) % 12]}"

def convert_solar_to_lunar(d, m, y):
    lunar = LunarDate.fromSolarDate(y, m, d)
    return lunar

# --- HÀM GỌI GEMINI AI ---
def ask_gemini(prompt, history=[], model_name="gemini-1.5-flash"):
    if not api_key:
        return "⚠️ Lỗi: Chưa có API Key."
    
    genai.configure(api_key=api_key)
    
    # Cấu hình model dựa trên lựa chọn của người dùng
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction="Bạn là một Đại Sư Tử Vi và Bát Tự thâm thúy. Bạn luận giải dựa trên Nam Phái và Tứ Trụ Tử Bình. Giọng văn cổ trang, sâu sắc, rành mạch."
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"❌ Lỗi kết nối AI: {str(e)}\n(Kiểm tra lại API Key hoặc đổi Model khác)"

# --- GIAO DIỆN CHÍNH ---
st.title("☯️ THIÊN CƠ CÁC - LUẬN GIẢI TỬ VI")

col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Họ tên tín chủ", "Nguyễn Văn A")
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
with col2:
    dob = st.date_input("Ngày sinh (Dương)", datetime(1990, 1, 1))
with col3:
    tob = st.time_input("Giờ sinh", datetime.strptime("12:00", "%H:%M").time())

# Nút bấm xử lý
if st.button("🔮 Lập Lá Số & Luận Giải", type="primary"):
    # Kiểm tra Key ngay lập tức
    if not api_key:
        st.error("⛔ Vui lòng nhập API Key ở menu bên trái và nhấn Enter trước khi bấm nút này!")
        st.stop() # Dừng chương trình tại đây, không chạy tiếp code bên dưới

    with st.spinner(f"Đại sư ({model_option}) đang bấm độn... xin chờ giây lát..."):
        # 1. Tính toán dữ liệu
        lunar_date = convert_solar_to_lunar(dob.day, dob.month, dob.year)
        can_chi_nam = get_can_chi(lunar_date.year)
        
        user_info = f"""
        THÔNG TIN:
        - Tên: {name} ({gender})
        - Dương lịch: {dob.day}/{dob.month}/{dob.year} - {tob.strftime('%H:%M')}
        - Âm lịch: {lunar_date.day}/{lunar_date.month}/{lunar_date.year}
        - Năm: {can_chi_nam}
        """
        st.session_state.user_data_context = user_info
        
        # 2. Tạo Prompt
        full_prompt = f"""
        {user_info}
        
        YÊU CẦU ĐẠI SƯ ({model_option}):
        1. **Lập Bát Tự:** Xác định Can Chi của Giờ, Ngày, Tháng, Năm.
        2. **An Sao (Mô phỏng):** Xác định Mệnh cung, Thân cung và các chính tinh tọa thủ.
        3. **Luận Giải:**
           - Phân tích Ngũ hành, dụng thần.
           - Luận về tính cách, công danh, tài lộc, tình duyên.
           - Vận hạn năm nay.
        4. **Lời khuyên:** Phong thủy cải mệnh.
        
        Dùng Markdown trình bày đẹp, dùng các icon đầu dòng.
        """
        
        # 3. Gọi AI
        response = ask_gemini(full_prompt, model_name=model_option)
        st.session_state.result = response
        st.session_state.has_run = True

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.has_run:
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📜 Lá Số Cơ Bản", "🔮 Lời Bình Của Đại Sư", "🎓 Hỏi Đáp & Nghiên Cứu"])
    
    with tab1:
        # Code vẽ lá số (Visual)
        cung_names = ["Tỵ", "Ngọ", "Mùi", "Thân", "Thìn", "", "", "Dậu", "Mão", "", "", "Tuất", "Dần", "Sửu", "Tý", "Hợi"]
        html_content = '<div class="tuvi-grid">'
        for i, name_cung in enumerate(cung_names):
            if name_cung == "":
                if i == 5:
                    html_content += f'<div class="center-box"><h3>{name}</h3><p>{can_chi_nam}</p></div>'
                continue
            html_content += f'<div class="cung"><div class="cung-header">Cung {name_cung}</div></div>'
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)

    with tab2:
        st.markdown(st.session_state.result)

    with tab3:
        st.info(f"Đang trò chuyện với: {model_option}")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Hỏi thêm về lá số..."):
            if not api_key:
                st.error("Mất kết nối API Key!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.spinner("Đại sư đang suy ngẫm..."):
                    research_prompt = f"Ngữ cảnh lá số: {st.session_state.user_data_context}\nCâu hỏi: {prompt}\nGiải thích chi tiết:"
                    ai_reply = ask_gemini(research_prompt, history=[], model_name=model_option)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    with st.chat_message("assistant"):
                        st.markdown(ai_reply)
