import streamlit as st
import google.generativeai as genai
from lunardate import LunarDate
from datetime import datetime
import pandas as pd

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Thiên Cơ Các - Tử Vi & Bát Tự AI", page_icon="☯️", layout="wide")

# --- CSS ĐỂ VẼ LÁ SỐ TỬ VI (MÔ PHỎNG) ---
st.markdown("""
<style>
    .tuvi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 5px;
        background-color: #fce4ec;
        padding: 10px;
        border: 2px solid #880e4f;
    }
    .cung {
        background-color: white;
        border: 1px solid #ddd;
        padding: 10px;
        min-height: 150px;
        font-size: 14px;
    }
    .cung-header {
        font-weight: bold;
        color: #b71c1c;
        text-align: center;
        border-bottom: 1px solid #eee;
        margin-bottom: 5px;
    }
    .center-box {
        grid-column: 2 / 4;
        grid-row: 2 / 4;
        background-color: #fff3e0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 2px double #e65100;
    }
    .stChatMessage {
        background-color: #f1f8e9;
    }
</style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_data_context" not in st.session_state:
    st.session_state.user_data_context = ""

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Nhập Google Gemini API Key", type="password")
    st.markdown("[👉 Lấy API Key miễn phí tại đây](https://aistudio.google.com/app/apikey)")
    st.info("Nhập API Key để kích hoạt tính năng Luận giải và Chat.")

# --- HÀM XỬ LÝ LỊCH ---
CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]

def get_can_chi(year):
    return f"{CAN[(year + 6) % 10]} {CHI[(year + 8) % 12]}"

def convert_solar_to_lunar(d, m, y):
    lunar = LunarDate.fromSolarDate(y, m, d)
    return lunar

# --- HÀM GỌI GEMINI AI ---
def ask_gemini(prompt, history=[]):
    if not api_key:
        return "⚠️ Vui lòng nhập API Key trước."
    
    genai.configure(api_key=api_key)
    # Cấu hình model
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Hoặc gemini-1.5-pro nếu muốn mạnh hơn
        generation_config=generation_config,
        system_instruction="Bạn là một Đại Sư Tử Vi và Bát Tự thâm thúy, thông thạo Tử Vi Đẩu Số (Nam Phái/Bắc Phái) và Tứ Trụ. Bạn có nhiệm vụ luận giải lá số và giải thích các thuật ngữ chuyên môn cho người học nghiên cứu. Giọng văn cổ trang, tôn trọng, nhưng phân tích khoa học, logic."
    )

    try:
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}"

# --- GIAO DIỆN CHÍNH ---
st.title("☯️ THIÊN CƠ CÁC - TỬ VI & BÁT TỰ")

# Input Form
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Họ tên", "Nguyễn Văn A")
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
with col2:
    dob = st.date_input("Ngày sinh (Dương)", datetime(1990, 1, 1))
with col3:
    tob = st.time_input("Giờ sinh", datetime.strptime("12:00", "%H:%M").time())

if st.button("🔮 Lập Lá Số & Luận Giải"):
    if not api_key:
        st.error("Vui lòng nhập API Key ở menu bên trái!")
    else:
        with st.spinner("Đang tính toán thiên bàn, địa bàn..."):
            # 1. Tính toán dữ liệu cơ bản
            lunar_date = convert_solar_to_lunar(dob.day, dob.month, dob.year)
            can_chi_nam = get_can_chi(lunar_date.year)
            
            # Context string để nuôi AI
            user_info = f"""
            THÔNG TIN TÍN CHỦ:
            - Họ tên: {name}
            - Giới tính: {gender}
            - Dương lịch: {dob.day}/{dob.month}/{dob.year} lúc {tob.strftime('%H:%M')}
            - Âm lịch: Ngày {lunar_date.day}, Tháng {lunar_date.month}, Năm {lunar_date.year}
            - Năm Can Chi: {can_chi_nam}
            """
            st.session_state.user_data_context = user_info # Lưu context
            
            # Prompt luận giải chi tiết
            full_prompt = f"""
            {user_info}
            
            YÊU CẦU CỦA TÔI:
            1. **Lập Bát Tự (Tứ Trụ):** Hãy xác định chính xác 4 trụ: Giờ, Ngày, Tháng, Năm (Can/Chi).
            2. **An Sao Tử Vi (Mô phỏng):** Xác định Cung Mệnh và Cung Thân đóng tại đâu? Các sao chính tinh tọa thủ tại Mệnh là gì? (Dựa trên kiến thức an sao của bạn).
            3. **Luận Giải Chuyên Sâu:**
               - **Bát Tự:** Phân tích ngũ hành vượng suy, dụng thần, kỵ thần.
               - **Tử Vi:** Luận về tính cách, sự nghiệp, tài bạch, phu thê.
            4. **Lời khuyên:** Cải vận theo phong thủy.
            
            Hãy trình bày định dạng Markdown rõ ràng, chuyên nghiệp.
            """
            
            response = ask_gemini(full_prompt)
            st.session_state.result = response
            st.session_state.has_run = True

# --- HIỂN THỊ KẾT QUẢ (TABS) ---
if "has_run" in st.session_state and st.session_state.has_run:
    tab1, tab2, tab3 = st.tabs(["📜 Lá Số (Mô Phỏng)", "🔮 Luận Giải Chi Tiết", "🎓 Nghiên Cứu & Hỏi Đáp"])
    
    with tab1:
        st.subheader(f"Lá Số: {name}")
        st.caption("Lưu ý: Đây là khung mô phỏng vị trí 12 cung. Vị trí sao được AI suy luận.")
        
        # Grid layout mô phỏng lá số (đây là HTML tĩnh, AI sẽ điền nội dung vào phần Luận giải)
        # Để lá số "sống", cần code JS/Python phức tạp hơn nhiều. Đây là khung visual.
        cung_html = ""
        cung_names = ["Tỵ", "Ngọ", "Mùi", "Thân", "Thìn", "", "", "Dậu", "Mão", "", "", "Tuất", "Dần", "Sửu", "Tý", "Hợi"]
        
        html_content = '<div class="tuvi-grid">'
        for i, name_cung in enumerate(cung_names):
            if name_cung == "":
                if i == 5: # Ô giữa chứa thông tin
                    html_content += f'''
                    <div class="center-box">
                        <h3>{name}</h3>
                        <p>{st.session_state.user_data_context.replace(chr(10), "<br>")}</p>
                    </div>
                    '''
                continue
            else:
                html_content += f'<div class="cung"><div class="cung-header">Cung {name_cung}</div><small>(Thông tin chi tiết xem tại tab Luận Giải)</small></div>'
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)

    with tab2:
        st.markdown(st.session_state.result)

    with tab3:
        st.info("Tại đây bạn có thể hỏi Đại Sư (AI) về các thuật ngữ trong lá số vừa lập hoặc kiến thức tử vi.")
        
        # Hiển thị lịch sử chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input chat mới
        if prompt := st.chat_input("Hỏi về thuật ngữ (VD: Sao Thiên Đồng là gì? Dụng thần là gì?)"):
            # Hiển thị câu hỏi người dùng
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Gọi AI trả lời (kèm context lá số)
            with st.spinner("Đại sư đang tra cứu thư tịch..."):
                research_prompt = f"""
                Ngữ cảnh: Đang thảo luận về lá số của {st.session_state.user_data_context}
                
                Câu hỏi người dùng: {prompt}
                
                Hãy giải thích sâu sắc, trích dẫn sách cổ (nếu có thể như Ma Thị, Thái Vi Phú...) để người dùng vừa học vừa hiểu.
                """
                ai_reply = ask_gemini(research_prompt)
                
                st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)

else:
    st.write("👈 Vui lòng nhập thông tin bên trên và bấm nút Lập Lá Số.")
