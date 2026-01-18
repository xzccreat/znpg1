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


# --- 1. 数据结构 ---
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


# --- 2. 字体加载 (防乱码) ---
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


# --- 3. 图片处理 (坐标修正) ---
def process_image_for_ai(image_file):
    img = Image.open(image_file)
    img = ImageOps.exif_transpose(img)  # 修正手机拍照旋转
    base_width = 1024
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
    return img


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 4. AI 引擎 ---
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


# --- 5. 绘图 (高亮+印章) ---
def draw_result(image: Image.Image, result: GradeResult) -> Image.Image:
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_draw.size

    # 荧光笔标记
    for error in result.errors:
        if len(error.box) == 4:
            x1, y1, x2, y2 = [c * (w if i % 2 == 0 else h) / 1000 for i, c in enumerate(error.box)]
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 60), outline=(255, 0, 0, 180), width=3)

    # 印章
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


# --- 6. 主程序 ---
def main():
    st.set_page_config(page_title="AI阅卷", layout="centered", initial_sidebar_state="collapsed")

    # --- CSS 暴力全屏优化 ---
    st.markdown("""
        <style>
        /* 1. 移除顶部的大片空白，让内容直接顶到头 */
        .main .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100%;
        }

        /* 2. 隐藏 Header 和 Footer，极致纯净 */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* 3. 摄像头组件：强制全屏高度 */
        [data-testid="stCameraInput"] {
            width: 100% !important;
            /* 计算高度：屏幕高度减去底部的上传按钮区域，留出一点点空间 */
            height: 85vh !important; 
            margin-bottom: 0px !important;
        }

        /* 4. 摄像头内的视频画面：强制填充，不留黑边 */
        [data-testid="stCameraInput"] video {
            height: 100% !important;
            width: 100% !important;
            object-fit: cover !important; /* 关键：像原生相机一样充满 */
            border-radius: 15px;
        }

        /* 5. 拍照按钮美化：悬浮在画面下方 */
        .stButton button {
            border-radius: 30px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    if "api_key" not in st.session_state: st.session_state.api_key = ""
    if "mode" not in st.session_state: st.session_state.mode = "scan"

    # --- 侧边栏配置 ---
    with st.sidebar:
        st.header("⚙️ 设置")
        st.session_state.api_key = st.text_input("阿里 API Key", value=st.session_state.api_key, type="password")
        # 需求2：改为手动输入分值
        max_score = st.number_input("满分设定", min_value=10, max_value=200, value=100, step=1)

    if not st.session_state.api_key:
        st.warning("请点击左上角箭头 > 打开侧边栏输入 Key")
        return

    # --- 界面 A: 拍摄模式 ---
    if st.session_state.mode == "scan":
        # 需求3：直接展示巨大的摄像头，不使用 Tabs
        shot = st.camera_input(" ", label_visibility="collapsed")

        # 需求1：修复相册上传无反应 -> 使用 Expander 折叠，不干扰主界面，但点击即用
        with st.expander("🖼️ 从相册选择图片 (点击展开)", expanded=False):
            upload = st.file_uploader("支持 JPG/PNG", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        # 逻辑处理：优先用拍照，其次用上传
        input_img = shot if shot else upload

        if input_img:
            # 防止重复刷新
            if "last_processed" not in st.session_state or st.session_state.last_processed != input_img.name:
                st.session_state.last_processed = input_img.name
                with st.spinner("⚡ 正在上传并识别..."):
                    st.session_state.clean_image = process_image_for_ai(input_img)
                    st.session_state.mode = "review"
                    st.rerun()

    # --- 界面 B: 结果模式 ---
    else:
        # 只在第一次进入时调用 API
        if "grade_result" not in st.session_state:
            with st.status("📝 AI 正在阅卷中...", expanded=True) as status:
                st.write("正在识别笔迹...")
                res = grade_with_qwen(st.session_state.clean_image, max_score, st.session_state.api_key)
                st.session_state.grade_result = res
                st.write("正在生成批注...")
                st.session_state.final_image = draw_result(st.session_state.clean_image, res)
                status.update(label="批改完成!", state="complete", expanded=False)

        # 结果展示
        st.image(st.session_state.final_image, use_container_width=True)

        # 错误详情
        if st.session_state.grade_result.errors:
            with st.expander(f"查看 {len(st.session_state.grade_result.errors)} 处扣分详情", expanded=False):
                for i, err in enumerate(st.session_state.grade_result.errors, 1):
                    st.error(f"**{i}.** {err.description}")
        else:
            st.success("🎉 全对！完美！")

        st.caption(st.session_state.grade_result.analysis_md)

        # 下一个按钮
        if st.button("📸 下一位同学", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k not in ["api_key", "mode"]:
                    del st.session_state[k]
            st.session_state.mode = "scan"
            st.rerun()


if __name__ == "__main__":
    main()