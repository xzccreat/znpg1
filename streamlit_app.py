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


# --- 1. 核心配置 ---
@dataclass
class ErrorItem:
    description: str
    box: List[int]  # [x1, y1, x2, y2]


@dataclass
class GradeResult:
    score: int
    max_score: int
    short_comment: str
    errors: List[ErrorItem]
    analysis_md: str


# --- 2. 增强型字体加载 (防乱码) ---
@st.cache_resource
def load_font(size: int):
    """
    三重保险加载字体：
    1. 尝试下载稳健的开源字体 (WenQuanYi Micro Hei)
    2. 尝试系统字体
    3. 只有全失败才用默认
    """
    font_url = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC-Bold.ttf"
    local_font = "NotoSansSC-Bold.ttf"

    # 方案A: 使用本地缓存或下载
    if not os.path.exists(local_font):
        try:
            # 伪装浏览器头，防止被拦截
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(font_url, headers=headers, timeout=15)
            with open(local_font, 'wb') as f:
                f.write(r.content)
        except:
            pass

    if os.path.exists(local_font):
        return ImageFont.truetype(local_font, size=size)

    # 方案B: Linux 系统常见字体
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
    except:
        pass

    return ImageFont.load_default()


# --- 3. 图片标准化处理 (关键步骤) ---
def process_image_for_ai(image_file):
    """
    核心修复：解决红框乱飞问题。
    1. 修正手机拍照的旋转信息 (EXIF)。
    2. 统一缩放到宽度 1024px，AI 坐标基于此图，画图也基于此图。
    """
    img = Image.open(image_file)
    # 1. 修正旋转
    img = ImageOps.exif_transpose(img)

    # 2. 统一尺寸 (保持比例，宽度固定1024)
    base_width = 1024
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)

    return img


def pil_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- 4. AI 批改引擎 ---
def grade_with_qwen(image: Image.Image, max_score: int, api_key: str) -> GradeResult:
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    base64_img = pil_to_base64(image)

    # Prompt 优化：要求更精准的单词级坐标
    prompt = f"""
    你是严厉的英语老师。请批改这张作业，满分 {max_score}。

    【重要任务】
    1. 找出具体的拼写错误、语法错误。
    2. "box"坐标必须尽可能精确地框住**错误的单词**，不要框整行。
    3. 如果没有明显错误，errors 为空。

    请输出纯 JSON：
    {{
        "score": 整数,
        "short_comment": "20字以内简评(中文)",
        "errors": [ 
            {{"description": "错误说明", "box": [x1, y1, x2, y2]}} 
        ],
        "analysis_md": "Markdown详细解析"
    }}
    注意：box坐标基于 1000x1000 的归一化坐标系。
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-max",  # 必须用 Max，定位能力最强
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
            short_comment=data.get("short_comment", "批改完成"),
            errors=error_list,
            analysis_md=data.get("analysis_md", "无分析内容")
        )
    except Exception as e:
        return GradeResult(0, max_score, "API错误", [], f"错误: {str(e)}")


# --- 5. 绘图：高亮模式 + 精致印章 ---
def draw_result(image: Image.Image, result: GradeResult) -> Image.Image:
    # 在副本上画图
    img_draw = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img_draw.size

    # 1. 绘制错误高亮 (荧光笔风格)
    for error in result.errors:
        if len(error.box) == 4:
            # 坐标换算
            x1 = error.box[0] * w / 1000
            y1 = error.box[1] * h / 1000
            x2 = error.box[2] * w / 1000
            y2 = error.box[3] * h / 1000

            # 画半透明红色填充块 (Highlighter)
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 60), outline=(255, 0, 0, 180), width=2)

    # 2. 绘制右上角印章 (极简风格)
    stamp_size = int(w * 0.25)  # 宽度占画布 25%
    stamp_h = int(stamp_size * 0.6)
    margin = 20

    # 印章背景 (圆角矩形，半透明白底，不遮挡文字)
    box_coords = [w - stamp_size - margin, margin, w - margin, margin + stamp_h]
    draw.rounded_rectangle(box_coords, radius=15, fill=(255, 255, 255, 200), outline=None)

    # 加载字体
    font_score = load_font(int(stamp_h * 0.6))
    font_text = load_font(int(stamp_h * 0.25))

    # 绘制分数 (鲜红色)
    score_str = f"{result.score}"
    draw.text((box_coords[0] + 15, box_coords[1] + 5), score_str, font=font_score, fill=(255, 50, 50, 255))

    # 绘制总分 (小一点，灰色)
    draw.text((box_coords[0] + 15 + font_score.getlength(score_str), box_coords[1] + stamp_h / 2.5),
              f"/{result.max_score}", font=font_text, fill=(100, 100, 100, 255))

    # 绘制评语 (如果字体加载失败，这一步可能不显示中文，但不会报错)
    draw.text((box_coords[0] + 15, box_coords[3] - stamp_h * 0.35),
              result.short_comment[:8], font=font_text, fill=(255, 50, 50, 255))

    # 合并图层
    return Image.alpha_composite(img_draw, overlay).convert("RGB")


# --- 6. 界面 UI ---
def main():
    st.set_page_config(page_title="英语批改", layout="centered", initial_sidebar_state="collapsed")

    # CSS 魔法：强制摄像头变大，修正样式
    st.markdown("""
        <style>
        /* 1. 摄像头区域极大化 */
        [data-testid="stCameraInput"] {
            width: 100% !important;
            min-height: 60vh !important; /* 占据屏幕高度的60% */
        }
        [data-testid="stCameraInput"] video {
            object-fit: cover !important; /* 画面填满，不留黑边 */
            border-radius: 12px;
        }

        /* 2. 按钮优化 */
        .stButton button {
            height: 3rem;
            font-weight: bold;
            border-radius: 20px;
        }

        /* 3. 隐藏顶部多余空白 */
        .block-container {
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    if "api_key" not in st.session_state: st.session_state.api_key = ""
    if "mode" not in st.session_state: st.session_state.mode = "scan"

    # 侧边栏
    with st.sidebar:
        st.session_state.api_key = st.text_input("阿里 API Key", value=st.session_state.api_key, type="password")
        max_score = st.slider("满分", 100, 150, 100)

    # 检查 Key
    if not st.session_state.api_key:
        st.info("👈 请点击左上角箭头，输入 API Key 开始使用")
        return

    # 状态 A: 拍照
    if st.session_state.mode == "scan":
        st.markdown("### 📸 拍摄作业")

        # 两个选项：大摄像头 OR 传图
        # 注意：在手机上 file_uploader 也可以直接调起相机
        tab1, tab2 = st.tabs(["📷 相机拍摄", "🖼️ 相册/原图"])

        with tab1:
            shot = st.camera_input("点击下方按钮拍照", label_visibility="collapsed")

        with tab2:
            upload = st.file_uploader("上传清晰图片", type=["jpg", "png", "jpeg"])

        # 处理图片
        input_img = shot if shot else upload
        if input_img:
            with st.spinner("🤖 正在处理图片并连接 AI..."):
                # 关键步骤：标准化图片
                st.session_state.clean_image = process_image_for_ai(input_img)
                st.session_state.mode = "review"
                st.rerun()

    # 状态 B: 结果
    else:
        st.markdown("### ✅ 批改结果")

        # 懒加载：只有第一次才调用 AI
        if "grade_result" not in st.session_state:
            with st.status("正在识别笔迹与批改...", expanded=True):
                res = grade_with_qwen(st.session_state.clean_image, max_score, st.session_state.api_key)
                st.session_state.grade_result = res
                st.session_state.final_image = draw_result(st.session_state.clean_image, res)

        # 显示结果图
        st.image(st.session_state.final_image, use_container_width=True)

        # 显示分析文本
        with st.expander("🔍 查看详细分析", expanded=True):
            if not st.session_state.grade_result.errors:
                st.success("🎉 全对！没有发现明显错误。")
            else:
                for i, err in enumerate(st.session_state.grade_result.errors, 1):
                    st.write(f"**{i}.** {err.description}")
            st.markdown("---")
            st.markdown(st.session_state.grade_result.analysis_md)

        # 重置按钮
        if st.button("📸 下一位"):
            for k in list(st.session_state.keys()):
                if k not in ["api_key", "mode"]:  # 保留 Key
                    del st.session_state[k]
            st.session_state.mode = "scan"
            st.rerun()


if __name__ == "__main__":
    main()