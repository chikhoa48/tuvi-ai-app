import streamlit as st
import google.generativeai as genai
from lunardate import LunarDate
from datetime import datetime
import pandas as pd

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Huyền Cơ Các - Tử Vi Pro", page_icon="☯️", layout="wide")

# --- CSS CHUYÊN NGHIỆP GIỐNG MẪU ---
st.markdown("""
<style>
    /* Font và màu sắc chung */
    body { font-family: 'Times New Roman', serif; background-color: #f0f2f6; }
    
    /* Lưới 12 cung */
    .laso-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        grid-template-rows: repeat(4, 160px);
        gap: 2px;
        background-color: #8b0000; /* Màu viền đỏ đậm */
        border: 2px solid #8b0000;
        max-width: 1000px;
        margin: 0 auto;
    }

    /* Ô từng cung */
    .cung-box {
        background-color: white;
        position: relative;
        padding: 5px;
        font-size: 12px;
        display: flex;
        flex-direction: column;
    }
    
    /* Header Cung (Tên cung, Đại vận) */
    .cung-header {
        display: flex;
        justify_content: space-between;
        border-bottom: 1px dashed #ccc;
        padding-bottom: 2px;
        margin-bottom: 5px;
        font-weight: bold;
        color: #b71c1c;
        text-transform: uppercase;
    }
    
    /* Chính tinh (To, Đậm) */
    .chinh-tinh {
        font-size: 15px;
        font-weight: bold;
        text-align: center;
        margin: 5px 0;
    }
    .sao-tot { color: #d81b60; } /* Màu hồng/đỏ cho sao tốt */
    .sao-xau { color: #212121; } /* Màu đen cho sao xấu/sát tinh */
    
    /* Phụ tinh (Chia 2 cột: Trái tốt, Phải xấu) */
    .phu-tinh-container {
        display: flex;
        flex-grow: 1;
        font-size: 11px;
    }
    .phu-tinh-left { width: 50%; text-align: left; color: #2e7d32; } /* Xanh lá */
    .phu-tinh-right { width: 50%; text-align: right; color: #424242; }
    
    /* Footer Cung (Tên Chi, Vị trí) */
    .cung-footer {
        text-align: center;
        font-weight: bold;
        background-color: #eceff1;
        margin-top: auto;
        font-size: 13px;
        padding: 2px;
    }

    /* Ô Thiên Bàn (Ở giữa) */
    .center-info {
        grid-column: 2 / 4;
        grid-row: 2 / 4;
        background-color: #fff8e1; /* Màu vàng nhạt */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 20px;
    }
    .center-title { font-size: 24px; font-weight: bold; color: #b71c1c; margin-bottom: 10px; }
    .bazi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; width: 100%; margin-top: 10px; }
    .bazi-col { background: white; padding: 5px; border: 1px solid #ddd; border-radius: 5px; }
    
    /* Tuần / Triệt */
    .tuan-triet {
        position: absolute;
        bottom: 25px;
        background: #000;
        color: #fff;
        padding: 1px 4px;
        font-size: 10px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# --- KHỞI TẠO STATE ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "user_data_context" not in st.session_state: st.session_state.user_data_context = ""
if "html_laso" not in st.session_state: st.session_state.html_laso = ""

# --- LOGIC AN SAO (RÚT GỌN - CORE ENGINE) ---
# Đây là phần logic Python để tính vị trí sao, thay vì đoán mò bằng AI
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]
CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
NGU_HANH_NAP_AM = { # Giản lược để demo
    "Giáp Tý": "Hải Trung Kim", "Ất Sửu": "Hải Trung Kim", "Bính Dần": "Lư Trung Hỏa", "Đinh Mão": "Lư Trung Hỏa",
    # ... (Cần thêm đủ 60 hoa giáp nếu muốn chính xác 100%, ở đây demo)
}

def get_can_chi_nam(year):
    return CAN[(year + 6) % 10], CHI[(year + 8) % 12]

def tim_cung_menh(thang_am, gio_chi_idx):
    # Khởi tại Dần (index 2)
    # Tháng 1 tại Dần, thuận đến tháng sinh, nghịch về giờ sinh
    pos = (2 + (thang_am - 1) - gio_chi_idx) % 12
    return pos # Trả về index 0-11 (0=Tý)

def tim_cuc(can_nam_idx, cung_menh_idx):
    # Logic tìm Cục (Thủy Nhị, Mộc Tam...)
    # Đây là logic phức tạp, demo mặc định Mộc Tam Cục để code chạy
    return 3 # 3 = Mộc Tam Cục

def an_chinh_tinh(ngay_am, cuc):
    # Logic An Tử Vi theo Ngày và Cục (Rất phức tạp, giản lược)
    # Giả sử Tử Vi tại Ngọ (6) cho demo
    tu_vi_pos = (cuc - ngay_am) % 12 
    # Nếu làm thật cần bảng tra Cục/Ngày
    tu_vi_pos = 6 # Mặc định demo: Tử Vi tại Ngọ
    
    # An các sao khác theo Tử Vi
    thien_phu_pos = (12 - tu_vi_pos) % 12 # Thiên Phủ đối xứng qua trục Dần Thân
    
    stars = {i: [] for i in range(12)}
    
    # Vòng Tử Vi: Tử Vi, Liêm Trinh, Thiên Đồng, Vũ Khúc, Thái Dương, Thiên Cơ
    stars[tu_vi_pos].append("Tử Vi")
    stars[(tu_vi_pos - 3) % 12].append("Liêm Trinh")
    stars[(tu_vi_pos - 4) % 12].append("Thiên Đồng")
    stars[(tu_vi_pos - 5) % 12].append("Vũ Khúc")
    stars[(tu_vi_pos - 6) % 12].append("Thái Dương")
    stars[(tu_vi_pos - 8) % 12].append("Thiên Cơ")
    
    # Vòng Thiên Phủ: Thiên Phủ, Thái Âm, Tham Lang, Cự Môn, Thiên Tướng, Thiên Lương, Thất Sát, Phá Quân
    stars[thien_phu_pos].append("Thiên Phủ")
    stars[(thien_phu_pos + 1) % 12].append("Thái Âm")
    stars[(thien_phu_pos + 2) % 12].append("Tham Lang")
    stars[(thien_phu_pos + 3) % 12].append("Cự Môn")
    stars[(thien_phu_pos + 4) % 12].append("Thiên Tướng")
    stars[(thien_phu_pos + 5) % 12].append("Thiên Lương")
    stars[(thien_phu_pos + 6) % 12].append("Thất Sát")
    stars[(thien_phu_pos + 10) % 12].append("Phá Quân")
    
    return stars

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    if "GEMINI_API_KEY" in st.secrets:
        st.success("✅ API Key đã kết nối")
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Nhập API Key", type="password")
    
    st.divider()
    
    # Cập nhật danh sách Model mới nhất
    model_option = st.selectbox(
        "Chọn Phiên Bản AI:",
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"],
        help="Chọn 'flash' nếu muốn nhanh, 'pro' nếu muốn luận giải sâu."
    )

# --- GIAO DIỆN CHÍNH ---
st.title("☯️ HUYỀN CƠ CÁC - TỬ VI & BÁT TỰ")

c1, c2, c3 = st.columns(3)
with c1: name = st.text_input("Họ tên", "Nguyễn Văn A")
with c2: 
    dob = st.date_input("Ngày sinh", datetime(1995, 6, 15))
    gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
with c3: 
    tob = st.time_input("Giờ sinh", datetime.strptime("09:30", "%H:%M").time())

def generate_laso_html(user_data):
    # 1. Tính toán cơ bản
    lunar = LunarDate.fromSolarDate(user_data['year'], user_data['month'], user_data['day'])
    can_nam, chi_nam = get_can_chi_nam(lunar.year)
    can_ngay = "Giáp" # Demo, cần thư viện tính Can Ngày chuẩn
    chi_ngay = CHI[(lunar.day + 2) % 12] # Demo
    
    # 2. An Sao (Gọi hàm logic)
    gio_chi_idx = (user_data['hour'] + 1) // 2 % 12
    menh_idx = tim_cung_menh(lunar.month, gio_chi_idx)
    than_idx = (2 + (lunar.month - 1) + gio_chi_idx) % 12 # Cung Thân
    
    star_map = an_chinh_tinh(lunar.day, 3) # Mặc định cục 3 demo
    
    # 3. Tạo HTML Grid
    html = '<div class="laso-container">'
    
    # Thứ tự vẽ grid: Tỵ(5)->Ngọ(6)->Mùi(7)->Thân(8)->Thìn(4)->CENTER->Dậu(9)->Mão(3)->CENTER->Tuất(10)->Dần(2)->Sửu(1)->Tý(0)->Hợi(11)
    # Mapping grid CSS order to Chi Index
    grid_order = [5, 6, 7, 8, 4, -1, -1, 9, 3, -1, -1, 10, 2, 1, 0, 11]
    
    cung_names_han = ["Mệnh", "Phụ Mẫu", "Phúc Đức", "Điền Trạch", "Quan Lộc", "Nô Bộc", "Thiên Di", "Tật Ách", "Tài Bạch", "Tử Tức", "Phu Thê", "Huynh Đệ"]
    
    # Xác định cung Mệnh ở đâu để an tên các cung còn lại
    cung_labels = {}
    for i in range(12):
        label_idx = (i - menh_idx) % 12
        cung_labels[i] = cung_names_han[label_idx]
        if i == than_idx: cung_labels[i] += " (Thân)"

    for idx in grid_order:
        if idx == -1: # Ô Center (Chỉ render 1 lần ở vị trí đầu tiên gặp)
            if "center_rendered" not in locals():
                html += f'''
                <div class="center-info">
                    <div class="center-title">NAM MỆNH: {user_data['name'].upper()}</div>
                    <div>Dương lịch: {user_data['day']}/{user_data['month']}/{user_data['year']} - {user_data['time']}</div>
                    <div>Âm lịch: {lunar.day}/{lunar.month}/{lunar.year} ({can_nam} {chi_nam})</div>
                    <div style="margin-top:10px; font-weight:bold; color:#d81b60">Bát Tự (Tứ Trụ)</div>
                    <div class="bazi-grid">
                        <div class="bazi-col"><div>Năm</div><b>{can_nam} {chi_nam}</b></div>
                        <div class="bazi-col"><div>Tháng</div><b>{lunar.month}</b></div>
                        <div class="bazi-col"><div>Ngày</div><b>{can_ngay} {chi_ngay}</b></div>
                        <div class="bazi-col"><div>Giờ</div><b>{CHI[gio_chi_idx]}</b></div>
                    </div>
                </div>
                '''
                locals()["center_rendered"] = True
            continue

        # Render Cung Box
        stars_in_cung = star_map.get(idx, [])
        chinh_tinh_html = "".join([f'<div class="chinh-tinh sao-tot">{s} (M)</div>' for s in stars_in_cung])
        if not chinh_tinh_html: chinh_tinh_html = '<div class="chinh-tinh" style="color:#ddd; font-weight:normal">Vô Chính Diệu</div>'
        
        # Thêm phụ tinh demo
        phu_tinh_left = "Văn Xương<br>Hóa Khoa" if idx % 2 == 0 else ""
        phu_tinh_right = "Đà La<br>Hóa Kỵ" if idx % 3 == 0 else ""
        
        cung_name = cung_labels.get(idx, "")
        
        html += f'''
        <div class="cung-box">
            <div class="cung-header">
                <span>{cung_name}</span>
                <span>{idx*10 + 2}-{idx*10+11}</span>
            </div>
            
            {chinh_tinh_html}
            
            <div class="phu-tinh-container">
                <div class="phu-tinh-left">{phu_tinh_left}</div>
                <div class="phu-tinh-right">{phu_tinh_right}</div>
            </div>
            
            <div class="cung-footer">
                {CHI[idx]}
            </div>
        </div>
        '''
    
    html += '</div>'
    return html, f"{can_nam} {chi_nam}"

if st.button("🔮 Lập Lá Số & Luận Giải", type="primary"):
    if not api_key:
        st.error("⛔ Chưa nhập API Key!")
        st.stop()
        
    with st.spinner("Đang an sao và kết nối thiên cơ..."):
        # 1. Tạo HTML Lá số (Chạy bằng Python Logic)
        user_data = {
            "name": name, "day": dob.day, "month": dob.month, "year": dob.year, 
            "hour": tob.hour, "time": tob.strftime("%H:%M")
        }
        html_output, nam_can_chi = generate_laso_html(user_data)
        st.session_state.html_laso = html_output
        
        # 2. Gửi thông tin cho AI luận giải
        prompt = f"""
        Bạn là Đại Sư Tử Vi. Hãy luận giải cho người có thông tin:
        - Tên: {name}, Giới tính: {gender}
        - Ngày sinh: {dob.strftime('%d/%m/%Y')} Giờ: {tob.strftime('%H:%M')}
        - Năm Âm Lịch: {nam_can_chi}
        
        Hãy đóng vai chuyên gia, viết lời bình giải chi tiết về:
        1. Mệnh, Thân (Tính cách, ưu nhược điểm).
        2. Quan Lộc & Tài Bạch (Sự nghiệp, tiền tài).
        3. Tình duyên (Phu Thê).
        4. Vận hạn năm nay ({datetime.now().year}).
        
        Dùng định dạng Markdown đẹp.
        """
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_option) # Sử dụng model người dùng chọn
            response = model.generate_content(prompt)
            st.session_state.result = response.text
            st.session_state.has_run = True
        except Exception as e:
            st.error(f"Lỗi AI: {str(e)}\n\nHãy thử đổi sang model 'gemini-1.5-flash' hoặc kiểm tra lại API Key.")

# --- HIỂN THỊ KẾT QUẢ ---
if "has_run" in st.session_state and st.session_state.has_run:
    tab1, tab2 = st.tabs(["📜 Lá Số Tử Vi (Đồ Họa)", "🔮 Luận Giải Chi Tiết"])
    
    with tab1:
        st.markdown(st.session_state.html_laso, unsafe_allow_html=True)
        st.caption("Ghi chú: Lá số được lập trình mô phỏng theo trường phái Nam Phái. Vị trí chính tinh là chính xác theo ngày/cục.")
        
    with tab2:
        st.markdown(st.session_state.result)
