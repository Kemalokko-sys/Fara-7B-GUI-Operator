import os
import re
import json
import time
import numpy as np
import torch
import spaces
import gradio as gr
from typing import Iterable, List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from qwen_vl_utils import process_vision_info

# --- Theme Configuration ---
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

print("🔄 Loading Fara-7B...")
MODEL_ID = "microsoft/Fara-7B" 

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device).eval()
    print("✅ Fara-7B Loaded Successfully.")
except Exception as e:
    print(f"❌ Failed to load Fara: {e}")
    model = None
    processor = None

# --- Helper Functions ---

def array_to_image(image_array: np.ndarray) -> Image.Image:
    if image_array is None: raise ValueError("No image provided.")
    return Image.fromarray(np.uint8(image_array))

def trim_generated(generated_ids, inputs):
    """Removes the prompt tokens from the generated output."""
    in_ids = getattr(inputs, "input_ids", None)
    if in_ids is None and isinstance(inputs, dict):
        in_ids = inputs.get("input_ids", None)
    if in_ids is None:
        return generated_ids
    return [out_ids[len(in_seq):] for in_seq, out_ids in zip(in_ids, generated_ids)]

def get_fara_prompt(task, image):
    """Constructs the specific prompt format required by Fara."""
    OS_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the current status.
    You need to generate the next action to complete the task.
    Output your action inside a <tool_call> block using JSON format.
    Include "coordinate": [x, y] in pixels for interactions.
    Examples:
    <tool_call>{"name": "User", "arguments": {"action": "click", "coordinate": [400, 300]}}</tool_call>
    <tool_call>{"name": "User", "arguments": {"action": "type", "coordinate": [100, 200], "text": "hello"}}</tool_call>
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": OS_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"Instruction: {task}"}]},
    ]

def parse_fara_response(response: str) -> List[Dict]:
    """Extracts JSON tool calls from Fara's output."""
    actions = []
    # Regex to find content inside <tool_call> tags
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            args = data.get("arguments", {})
            coords = args.get("coordinate", [])
            action_type = args.get("action", "unknown")
            text_content = args.get("text", "")
            
            # Ensure coordinates exist and have x, y
            if coords and len(coords) == 2:
                actions.append({
                    "type": action_type, 
                    "x": float(coords[0]), 
                    "y": float(coords[1]), 
                    "text": text_content
                })
        except Exception as e:
            print(f"Error parsing Fara JSON: {e}")
            pass
    return actions

def create_localized_image(original_image: Image.Image, actions: list[dict]) -> Optional[Image.Image]:
    """Draws visual indicators (dots/boxes) on the image based on model predictions."""
    if not actions: return None
    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.load_default(size=18)
    except IOError:
        font = ImageFont.load_default()
    
    for act in actions:
        x = act['x']
        y = act['y']
        
        pixel_x, pixel_y = int(x), int(y)
            
        color = 'red' if 'click' in act['type'].lower() else 'blue'
        
        r = 20
        line_width = 5
        
        # Draw target circle
        draw.ellipse([pixel_x - r, pixel_y - r, pixel_x + r, pixel_y + r], outline=color, width=line_width)
        draw.ellipse([pixel_x - 4, pixel_y - 4, pixel_x + 4, pixel_y + 4], fill=color)
        
        # Create label text
        label = f"{act['type'].capitalize()}"
        if act.get('text'): label += f": \"{act['text']}\""
        
        text_pos = (pixel_x + 25, pixel_y - 15)
        
        # Draw label background and text
        try:
            bbox = draw.textbbox(text_pos, label, font=font)
            padded_bbox = (bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2)
            draw.rectangle(padded_bbox, fill="black", outline=color)
            draw.text(text_pos, label, fill="white", font=font)
        except Exception as e:
            draw.text(text_pos, label, fill="white")

    return img_copy

# --- Main Inference Function ---

@spaces.GPU
def process_screenshot(input_numpy_image: np.ndarray, task: str):
    if input_numpy_image is None: 
        return "⚠️ Please upload an image.", None
    if not task.strip(): 
        return "⚠️ Please provide a task instruction.", None
    if model is None: 
        return "Error: Fara model failed to load on startup.", None

    input_pil_image = array_to_image(input_numpy_image)
    
    print(f"Using Fara-7B to process task: {task}")
    
    # Prepare inputs using Qwen_VL_Utils and AutoProcessor
    messages = get_fara_prompt(task, input_pil_image)
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
    generated_ids = trim_generated(generated_ids, inputs)
    raw_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Parse and Visualize
    actions = parse_fara_response(raw_response)
    print(f"Raw Output: {raw_response}")
    print(f"Parsed Actions: {actions}")

    output_image = input_pil_image
    if actions:
        vis = create_localized_image(input_pil_image, actions)
        if vis: output_image = vis
            
    return raw_response, output_image

# --- Gradio UI ---

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **Fara-7B GUI Operator 🖥️**", elem_id="main-title")
    gr.Markdown("Perform Computer Use Agent tasks using the [Microsoft Fara-7B](https://huggingface.co/microsoft/Fara-7B) model.")

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Upload UI Image", type="numpy", height=500)
            
            task_input = gr.Textbox(
                label="Task Instruction",
                placeholder="e.g. Click on the search bar",
                lines=2
            )
            submit_btn = gr.Button("Execute Agent", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Visualized Action Points", elem_id="out_img", height=500)
            output_text = gr.Textbox(label="Agent Model Response", lines=10)

    submit_btn.click(
        fn=process_screenshot,
        inputs=[input_image, task_input],
        outputs=[output_text, output_image]
    )
    
    gr.Examples(
        examples=[
            ["examples/1.png", "Click on the start menu."],
            ["examples/2.png", "Type 'Hello World' in the search box."],
        ],
        inputs=[input_image, task_input],
        label="Quick Examples"
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)
