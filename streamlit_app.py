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


# 核心修复：增加防崩溃护盾的字体加载函数
@st.cache_resource
def load_font(size: int):
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Bold.ttf"
    local_font = "NotoSansSC-Bold.ttf"

    # 1. 如果本地没有，尝试下载
    if not os.path.exists(local_font):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(font_url, headers=headers, timeout=10)
            if r.status_code == 200:
                with open(local_font, 'wb') as f:
                    f.write(r.content)
        except Exception:
            pass  # 下载失败忽略，后续会处理

    # 2. 尝试加载字体 (带异常捕获)
    if os.path.exists(local_font):
        try:
            return ImageFont.truetype(local_font, size=size)
        except OSError:
            # 🚨 关键修复：如果文件损坏(OSError)，删除它，避免下次还报错
            try:
                os.remove(local_font)
                print(f"检测到字体损坏，已删除: {local_font}")
            except:
                pass
            return ImageFont.load_default()

    # 3. 保底方案
    return ImageFont.load_default()


def process_image_for_ai(image_file):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    base_width = 800
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    return img


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=65)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 2. AI 核心逻辑 ---
def grade_with_qwen(image: Image.Image, current_max_score: int, api_key: str) -> GradeResult:
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    base64_img = pil_to_base64(image)

    prompt = f"""
    你是严厉的英语阅卷老师。
    用户设定这张图片的总分值为：【{current_max_score} 分】。

    【任务要求】
    1. **分值权重**：以 {current_max_score} 分为满分上限。
    2. **精准定位**："box"坐标必须精确框住错误的单词。
    3. **错误描述**：请指明错误类型和位置（如"Q1: 拼写错误"）。

    【输出 JSON】
    {{
        "score": 整数,
        "short_comment": "简评(中文, 20字内)",
        "errors": [ 
            {{"description": "错误说明", "box": [x1, y1, x2, y2]}}
        ],
        "analysis_md": "Markdown格式分析"
    }}
    注意：box基于1000x1000坐标系。
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
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


# --- 3. 绘图逻辑 ---
def draw_result(image: Image.Image, result: GradeResult) -> Image.Image:
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_draw.size

    # 错误高亮
    for error in result.errors:
        if len(error.box) == 4:
            x1, y1, x2, y2 = [c * (w if i % 2 == 0 else h) / 1000 for i, c in enumerate(error.box)]
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 60), outline=(255, 0, 0, 180), width=3)

    # 印章绘制
    stamp_size = int(w * 0.22)
    stamp_h = int(stamp_size * 0.65)
    margin = 20
    box_coords = [w - stamp_size - margin, margin, w - margin, margin + stamp_h]

    draw.rounded_rectangle(box_coords, radius=15, fill=(255, 255, 255, 210), outline=None)

    # 这里调用 load_font 即使字体坏了也不会崩溃，而是用默认字体
    font_score = load_font(int(stamp_h * 0.6))
    font_text = load_font(int(stamp_h * 0.25))

    draw.text((box_coords[0] + 15, box_coords[1] + 5), str(result.score), font=font_score, fill=(220, 20, 60, 255))
    draw.text((box_coords[0] + 15 + font_score.getlength(str(result.score)), box_coords[1] + stamp_h / 2.5),
              f"/{result.max_score}", font=font_text, fill=(100, 100, 100, 255))
    draw.text((box_coords[0] + 15, box_coords[3] - stamp_h * 0.35),
              result.short_comment[:8], font=font_text, fill=(220, 20, 60, 255))

    return Image.alpha_composite(img_draw, overlay).convert("RGB")


# --- 4. 主程序 ---
def main():
    st.set_page_config(page_title="AI阅卷", layout="centered", initial_sidebar_state="collapsed")

    # Session 初始化
    if "page" not in st.session_state: st.session_state.page = "setup"
    if "api_key" not in st.session_state: st.session_state.api_key = ""
    if "current_score_setting" not in st.session_state: st.session_state.current_score_setting = 100
    if "score_locked" not in st.session_state: st.session_state.score_locked = False

    # ---------------------------------------------------------
    # 页面 1: 初始配置
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 页面 2: 拍摄页
    # ---------------------------------------------------------
    elif st.session_state.page == "scan":
        st.markdown("""
            <style>
            header {visibility: hidden;} 
            .main .block-container { padding: 10px !important; max-width: 100%; }
            [data-testid="stCameraInput"] { width: 100% !important; height: 75vh !important; margin-top: 5px; }
            [data-testid="stCameraInput"] video { height: 100% !important; object-fit: cover !important; border-radius: 15px; }
            .stButton button { border-radius: 25px; height: 3rem; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)

        # 顶部控制区
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
            is_locked = st.checkbox("🔒锁定", value=st.session_state.score_locked)
            st.session_state.score_locked = is_locked

        if st.session_state.score_locked:
            st.caption(f"🔒 分值已锁定为 **{st.session_state.current_score_setting} 分**")
        else:
            st.caption(f"🔓 当前满分 **{st.session_state.current_score_setting} 分**")

        # 摄像头与上传
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
                with st.spinner(f"⚡ 正在批改 (满分: {st.session_state.current_score_setting})..."):
                    st.session_state.clean_image = process_image_for_ai(input_img)
                    st.session_state.page = "review"
                    st.rerun()

    # ---------------------------------------------------------
    # 页面 3: 结果页
    # ---------------------------------------------------------
    elif st.session_state.page == "review":
        st.markdown("### 📝 批改结果")

        if "grade_result" not in st.session_state or st.session_state.get("current_img_id") != id(
                st.session_state.clean_image):
            st.session_state.current_img_id = id(st.session_state.clean_image)
            with st.status("AI 阅卷中...", expanded=True) as status:
                res = grade_with_qwen(st.session_state.clean_image, st.session_state.current_score_setting,
                                      st.session_state.api_key)
                st.session_state.grade_result = res
                st.session_state.final_image = draw_result(st.session_state.clean_image, res)
                status.update(label="完成!", state="complete", expanded=False)

        st.image(st.session_state.final_image, use_container_width=True)

        if st.session_state.grade_result.errors:
            with st.expander(f"查看 {len(st.session_state.grade_result.errors)} 处扣分点", expanded=True):
                for i, err in enumerate(st.session_state.grade_result.errors, 1):
                    st.error(f"**{i}.** {err.description}")
        else:
            st.success("🎉 全对！完美！")

        st.caption(st.session_state.grade_result.analysis_md)

        if st.button("📸 下一位同学 (分值不变)", type="primary", use_container_width=True):
            for k in ["clean_image", "grade_result", "final_image", "last_processed", "current_img_id"]:
                if k in st.session_state: del st.session_state[k]
            st.session_state.page = "scan"
            st.rerun()


if __name__ == "__main__":
    main()