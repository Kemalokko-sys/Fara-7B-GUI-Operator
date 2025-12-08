# **Fara-7B GUI Operator**

> A Gradio-based demonstration for the Microsoft Fara-7B model, designed as a computer use agent. Users upload UI screenshots (e.g., desktop or app interfaces), provide task instructions (e.g., "Click on the search bar"), and receive parsed actions (clicks, types) with visualized indicators (circles and labels) overlaid on the image. Supports JSON-formatted tool calls for precise coordinate-based interactions.

> Demo: https://huggingface.co/spaces/prithivMLmods/CUA-GUI-Operator

<img width="1918" height="1437" alt="Screenshot 2025-12-07 at 11-41-16 CUA GUI Operator - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/ef8b25b8-810d-4b2e-a143-d35c2211a66f" />

![12](https://github.com/user-attachments/assets/914d39a9-afb2-4787-a522-651c2ede68e5)

## Features

- **UI Image Processing**: Upload screenshots; model analyzes and suggests actions like clicks or text input at specific coordinates.
- **Task-Driven Inference**: Natural language instructions generate structured JSON actions (e.g., {"action": "click", "coordinate": [400, 300]}).
- **Action Visualization**: Overlays red circles for clicks and blue for others, with labels (e.g., "Click" or "Type: 'Hello'") on the output image.
- **Response Parsing**: Extracts tool calls from model output using regex; handles multiple actions per task.
- **Custom Theme**: OrangeRedTheme with gradients for an intuitive interface.
- **Examples Integration**: Pre-loaded samples for quick testing (e.g., Windows start menu, search box).
- **Queueing Support**: Handles up to 50 concurrent inferences for efficient use.
- **Error Resilience**: Fallbacks for model loading failures or invalid inputs; console logging for debugging.

## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for float16; falls back to CPU).
- Git for cloning dependencies.
- Hugging Face account (optional, for model caching via `huggingface_hub`).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Fara-7B-Action-Points-Demo.git
   cd Fara-7B-Action-Points-Demo
   ```

2. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   transformers==4.57.1
   webdriver-manager
   huggingface_hub
   python-dotenv
   sentencepiece
   qwen-vl-utils
   gradio_modal
   torchvision
   matplotlib
   accelerate
   num2words
   pydantic
   requests
   pillow
   openai
   spaces
   einops
   torch
   peft
   ```

3. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

1. **Upload Image**: Provide a UI screenshot (e.g., PNG of a desktop or app window).

2. **Enter Task**: Describe the action in the textbox (e.g., "Click on the start menu" or "Type 'Hello World' in the search box").

3. **Execute**: Click "Execute Agent" to run inference.

4. **View Results**:
   - Text: Raw model response with parsed JSON actions.
   - Image: Annotated screenshot showing action points (circles with labels).

### Example Workflow
- Upload a Windows desktop image.
- Task: "Click on the start menu."
- Output: Response with click action at coordinates; image with red circle labeled "Click" on the start button.

## Troubleshooting

- **Model Loading Errors**: Ensure transformers 4.57.1; check CUDA with `torch.cuda.is_available()`. Use `torch.float32` if float16 OOM occurs.
- **No Actions Parsed**: Verify task clarity; raw output logged in console. Adjust max_new_tokens if truncated.
- **Visualization Issues**: PIL font errors fallback to default; ensure images are RGB.
- **Queue Full**: Increase `max_size` in `demo.queue()` for higher traffic.
- **Vision Utils**: Install `qwen-vl-utils` for image processing; test with examples.
- **UI Rendering**: Set `ssr_mode=True` if gradients fail; check CSS for custom styles.

## Contributing

Contributions encouraged! Fork, create a feature branch (e.g., for multi-step tasks), and submit PRs with tests. Focus areas:
- Support for video inputs or real-time GUI control.
- Additional action types (e.g., scroll, drag).
- Integration with browser automation.

Repository: [https://github.com/PRITHIVSAKTHIUR/Fara-7B-Action-Points-Demo.git](https://github.com/PRITHIVSAKTHIUR/Fara-7B-Action-Points-Demo.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
