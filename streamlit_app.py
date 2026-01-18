import json
import os
import base64
import io
import requests
from dataclasses import dataclass
from typing import List
from openai import OpenAI
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


# --- 1. 基础配置与工具 ---
@dataclass
class ErrorItem:
    description: str
    box: List[int]


@dataclass
class GradeResult:
    score: int
    max_score: int
    short_comment: str
    errors: List[ErrorItem]
    analysis_md: str


@st.cache_resource
def load_font(size: int):
    """
    加载字体：优先下载 Google 开源字体用于显示漂亮的数字。
    如果失败，自动回退到系统默认字体，防止报错。
    """
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Bold.ttf"
    local_font = "NotoSansSC-Bold.ttf"

    # 1. 尝试下载
    if not os.path.exists(local_font):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(font_url, headers=headers, timeout=3)  # 设置短超时，不阻塞
            if r.status_code == 200:
                with open(local_font, 'wb') as f:
                    f.write(r.content)
        except:
            pass

    # 2. 尝试加载
    if os.path.exists(local_font):
        try:
            return ImageFont.truetype(local_font, size=size)
        except:
            pass

    # 3. 保底
    return ImageFont.load_default()


def process_image_for_ai(image_file):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    # 压缩图片以加快上传速度，宽800通常足够识别手写
    base_width = 800
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    return img


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=65)  # 质量65平衡体积与清晰度
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 2. AI 核心逻辑 ---
def grade_with_qwen(image: Image.Image, current_max_score: int, api_key: str) -> GradeResult:
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    base64_img = pil_to_base64(image)

    # Prompt 升级：要求 AI 用文字描述具体位置
    prompt = f"""
    你是严厉的英语阅卷老师。
    用户设定这张图片的总分值为：【{current_max_score} 分】。

    【任务】
    1. 找出拼写、语法等错误。
    2. **关键**：由于不在图上画框，请在 description 中明确指出错误的位置（例如：“第2行句首”、“倒数第二段”或引用上下文）。

    【输出 JSON】
    {{
        "score": 整数,
        "short_comment": "简评(中文)",
        "errors": [ 
            {{"description": "位置+错误说明 (如: 第3行 'aple' 拼写错误)", "box": []}} 
        ],
        "analysis_md": "Markdown格式分析"
    }}
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # 速度快，性价比高
            messages=[
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": prompt},
                ]}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        error_list = [ErrorItem(**e) for e in data.get("errors", [])]
        return GradeResult(
            score=int(data.get("score", 0)),
            max_score=current_max_score,
            short_comment=data.get("short_comment", "已批改"),
            errors=error_list,
            analysis_md=data.get("analysis_md", "")
        )
    except Exception as e:
        return GradeResult(0, current_max_score, "Error", [], f"错误: {str(e)}")


# --- 3. 绘图逻辑 (无红框版) ---
def draw_result(image: Image.Image, result: GradeResult) -> Image.Image:
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_draw.size

    # --- 1. 已移除红框绘制代码 ---

    # --- 2. 绘制极简印章 (只显示分数) ---
    stamp_size = int(w * 0.25)  # 印章宽度占图片1/4
    stamp_h = int(stamp_size * 0.55)
    margin = 20

    # 位置：右上角
    box_coords = [w - stamp_size - margin, margin, w - margin, margin + stamp_h]

    # 背景：半透明白色，带深红色边框
    draw.rounded_rectangle(box_coords, radius=15, fill=(255, 255, 255, 235), outline=(200, 30, 30, 255), width=4)

    # 字体大小计算
    font_score_size = int(stamp_h * 0.65)
    font_small_size = int(stamp_h * 0.3)

    font_score = load_font(font_score_size)
    font_small = load_font(font_small_size)

    score_text = str(result.score)
    max_text = f"/{result.max_score}"

    # 绘制分数 (红色)
    # 稍微调整坐标，使其视觉居中
    draw.text((box_coords[0] + 20, box_coords[1] + stamp_h * 0.1), score_text, font=font_score, fill=(220, 20, 20, 255))

    # 绘制满分 (灰色小字)
    offset_x = font_score.getlength(score_text) + 25
    draw.text((box_coords[0] + offset_x, box_coords[1] + stamp_h * 0.45), max_text, font=font_small,
              fill=(120, 120, 120, 255))

    return Image.alpha_composite(img_draw, overlay).convert("RGB")


# --- 4. 主程序 ---
def main():
    st.set_page_config(page_title="AI阅卷", layout="centered", initial_sidebar_state="collapsed")

    if "page" not in st.session_state: st.session_state.page = "setup"
    if "api_key" not in st.session_state: st.session_state.api_key = ""
    if "current_score_setting" not in st.session_state: st.session_state.current_score_setting = 100
    if "score_locked" not in st.session_state: st.session_state.score_locked = False

    # ------------------ 1. 设置页 ------------------
    if st.session_state.page == "setup":
        st.markdown("## 🤖 AI 阅卷老师")
        with st.container(border=True):
            with st.form("login_form"):
                key_input = st.text_input("请输入阿里云 API Key", value=st.session_state.api_key, type="password")
                submitted = st.form_submit_button("🚀 确认并进入系统", use_container_width=True, type="primary")
                if submitted:
                    if not key_input:
                        st.error("Key 不能为空")
                    else:
                        st.session_state.api_key = key_input
                        st.session_state.page = "scan"
                        st.rerun()

    # ------------------ 2. 拍摄页 ------------------
    elif st.session_state.page == "scan":
        # CSS 强制全屏拍摄体验
        st.markdown("""
            <style>
            header {visibility: hidden;} 
            .main .block-container { padding: 10px !important; max-width: 100%; }
            [data-testid="stCameraInput"] { width: 100% !important; height: 75vh !important; margin-top: 5px; }
            [data-testid="stCameraInput"] video { height: 100% !important; object-fit: cover !important; border-radius: 15px; }
            .stButton button { border-radius: 25px; height: 3rem; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1.2, 2, 1.2])
        with c1:
            st.markdown("#### 📸 拍题")
        with c2:
            new_score = st.number_input("满分", value=st.session_state.current_score_setting,
                                        min_value=1, max_value=200, step=1, label_visibility="collapsed",
                                        disabled=st.session_state.score_locked)
            if not st.session_state.score_locked:
                st.session_state.current_score_setting = new_score
        with c3:
            # 简化锁定逻辑
            st.session_state.score_locked = st.checkbox("🔒锁定", value=st.session_state.score_locked)

        if st.session_state.score_locked:
            st.caption(f"🔒 满分锁定: {st.session_state.current_score_setting}")
        else:
            st.caption(f"🔓 当前满分: {st.session_state.current_score_setting}")

        shot = st.camera_input(" ", label_visibility="collapsed")
        with st.expander("🖼️ 从相册选择", expanded=False):
            upload = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        if st.button("⬅️ 设置 Key"):
            st.session_state.page = "setup"
            st.rerun()

        input_img = shot if shot else upload
        if input_img:
            if "last_processed" not in st.session_state or st.session_state.last_processed != input_img.name:
                st.session_state.last_processed = input_img.name
                with st.spinner(f"⚡ 正在阅卷 (满分: {st.session_state.current_score_setting})..."):
                    st.session_state.clean_image = process_image_for_ai(input_img)
                    st.session_state.page = "review"
                    st.rerun()

    # ------------------ 3. 结果页 ------------------
    elif st.session_state.page == "review":
        st.markdown("### 📝 批改结果")

        # 确保不重复调用 API
        if "grade_result" not in st.session_state or st.session_state.get("current_img_id") != id(
                st.session_state.clean_image):
            st.session_state.current_img_id = id(st.session_state.clean_image)
            with st.status("AI 阅卷中...", expanded=True) as status:
                res = grade_with_qwen(st.session_state.clean_image, st.session_state.current_score_setting,
                                      st.session_state.api_key)
                st.session_state.grade_result = res
                st.session_state.final_image = draw_result(st.session_state.clean_image, res)
                status.update(label="完成!", state="complete", expanded=False)

        # 1. 展示干净的带分图片
        st.image(st.session_state.final_image, use_container_width=True)

        # 2. 展示扣分详情
        if st.session_state.grade_result.errors:
            st.warning(f"发现 {len(st.session_state.grade_result.errors)} 处扣分点：")
            for i, err in enumerate(st.session_state.grade_result.errors, 1):
                # 这里的 description 会包含 AI 生成的位置信息
                st.error(f"**{i}.** {err.description}")
        else:
            st.success("🎉 全对！完美！")

        st.caption("💡 简评: " + st.session_state.grade_result.short_comment)

        # 3. 下一位按钮
        if st.button("📸 下一位 (分值不变)", type="primary", use_container_width=True):
            # 清除图片缓存，保留分值设置
            for k in ["clean_image", "grade_result", "final_image", "last_processed", "current_img_id"]:
                if k in st.session_state: del st.session_state[k]
            st.session_state.page = "scan"
            st.rerun()


if __name__ == "__main__":
    main()