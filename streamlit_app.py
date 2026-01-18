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
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Bold.ttf"
    local_font = "NotoSansSC-Bold.ttf"
    if not os.path.exists(local_font):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(font_url, headers=headers, timeout=15)
            with open(local_font, 'wb') as f:
                f.write(r.content)
        except:
            pass
    if os.path.exists(local_font):
        return ImageFont.truetype(local_font, size=size)
    return ImageFont.load_default()


def process_image_for_ai(image_file):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)
    base_width = 1024
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    return img


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 2. AI 核心逻辑 ---
def grade_with_qwen(image: Image.Image, max_score: int, api_key: str) -> GradeResult:
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    base64_img = pil_to_base64(image)

    prompt = f"""
    你是严厉的英语老师。批改这张作业，满分 {max_score}。

    任务：
    1. 找出拼写/语法错误。
    2. "box"坐标必须精确框住错误单词。
    3. 若无错误，errors为空。

    输出JSON：
    {{
        "score": 整数,
        "short_comment": "简评(中文)",
        "errors": [ {{"description": "错误说明", "box": [x1, y1, x2, y2]}} ],
        "analysis_md": "Markdown解析"
    }}
    注意：box基于1000x1000坐标系。
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max",
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
            max_score=max_score,
            short_comment=data.get("short_comment", "已批改"),
            errors=error_list,
            analysis_md=data.get("analysis_md", "")
        )
    except Exception as e:
        return GradeResult(0, max_score, "Error", [], f"错误: {str(e)}")


# --- 3. 绘图逻辑 ---
def draw_result(image: Image.Image, result: GradeResult) -> Image.Image:
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_draw.size

    for error in result.errors:
        if len(error.box) == 4:
            x1, y1, x2, y2 = [c * (w if i % 2 == 0 else h) / 1000 for i, c in enumerate(error.box)]
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 60), outline=(255, 0, 0, 180), width=3)

    stamp_size = int(w * 0.22)
    stamp_h = int(stamp_size * 0.65)
    margin = 20
    box_coords = [w - stamp_size - margin, margin, w - margin, margin + stamp_h]

    draw.rounded_rectangle(box_coords, radius=15, fill=(255, 255, 255, 210), outline=None)

    font_score = load_font(int(stamp_h * 0.6))
    font_text = load_font(int(stamp_h * 0.25))

    draw.text((box_coords[0] + 15, box_coords[1] + 5), str(result.score), font=font_score, fill=(220, 20, 60, 255))
    draw.text((box_coords[0] + 15 + font_score.getlength(str(result.score)), box_coords[1] + stamp_h / 2.5),
              f"/{result.max_score}", font=font_text, fill=(100, 100, 100, 255))
    draw.text((box_coords[0] + 15, box_coords[3] - stamp_h * 0.35),
              result.short_comment[:8], font=font_text, fill=(220, 20, 60, 255))

    return Image.alpha_composite(img_draw, overlay).convert("RGB")


# --- 4. 主程序流程 ---
def main():
    st.set_page_config(page_title="AI阅卷", layout="centered", initial_sidebar_state="collapsed")

    # 初始化 Session State
    if "page" not in st.session_state: st.session_state.page = "setup"
    if "api_key" not in st.session_state: st.session_state.api_key = ""
    if "max_score" not in st.session_state: st.session_state.max_score = 100

    # ---------------------------------------------------------
    # 页面 1: 设置页 (解决侧边栏看不到的问题)
    # ---------------------------------------------------------
    if st.session_state.page == "setup":
        st.markdown("## 🤖 AI 阅卷老师")
        st.info("首次使用，请配置以下信息：")

        with st.container(border=True):
            # 使用 form 避免每次输入都刷新，必须点按钮才提交
            with st.form("settings_form"):
                key_input = st.text_input("1. 输入阿里云 API Key",
                                          value=st.session_state.api_key,
                                          type="password",
                                          placeholder="sk-xxxxxxxx")

                score_input = st.number_input("2. 设定试卷满分",
                                              min_value=1, max_value=200,
                                              value=st.session_state.max_score, step=1)

                # 显眼的提交按钮
                submitted = st.form_submit_button("🚀 确认并开始", use_container_width=True, type="primary")

                if submitted:
                    if not key_input:
                        st.error("请输入 API Key 才能继续！")
                    else:
                        st.session_state.api_key = key_input
                        st.session_state.max_score = score_input
                        st.session_state.page = "scan"  # 切换到拍摄页
                        st.rerun()

    # ---------------------------------------------------------
    # 页面 2: 沉浸式拍摄页 (应用暴力全屏 CSS)
    # ---------------------------------------------------------
    elif st.session_state.page == "scan":
        # ⚠️ 只有在拍摄页才注入这个 CSS，防止影响设置页
        st.markdown("""
            <style>
            /* 隐藏顶部Header */
            header {visibility: hidden;} 
            /* 移除页面边距 */
            .main .block-container {
                padding: 0rem !important;
                max-width: 100%;
            }
            /* 摄像头全屏 */
            [data-testid="stCameraInput"] {
                width: 100% !important;
                height: 85vh !important;
                margin-bottom: 0px !important;
            }
            [data-testid="stCameraInput"] video {
                height: 100% !important;
                object-fit: cover !important;
                border-radius: 0px 0px 20px 20px;
            }
            /* 底部按钮区域美化 */
            .stButton button {
                border-radius: 25px;
                height: 3rem;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        # 1. 摄像头区域
        shot = st.camera_input(" ", label_visibility="collapsed")

        # 2. 相册上传区域 (折叠)
        with st.expander("🖼️ 从相册选择图片", expanded=False):
            upload = st.file_uploader(" ", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        # 3. 返回设置按钮 (放在最下面)
        if st.button("⚙️ 修改 Key 或 分数"):
            st.session_state.page = "setup"
            st.rerun()

        # 处理逻辑
        input_img = shot if shot else upload
        if input_img:
            # 防止重复处理
            if "last_processed" not in st.session_state or st.session_state.last_processed != input_img.name:
                st.session_state.last_processed = input_img.name
                with st.spinner("⚡ 正在分析..."):
                    st.session_state.clean_image = process_image_for_ai(input_img)
                    st.session_state.page = "review"  # 切换到结果页
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
                res = grade_with_qwen(st.session_state.clean_image, st.session_state.max_score,
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

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📸 下一位", type="primary", use_container_width=True):
                st.session_state.page = "scan"
                st.rerun()
        with col2:
            if st.button("⚙️ 设置", use_container_width=True):
                st.session_state.page = "setup"
                st.rerun()


if __name__ == "__main__":
    main()