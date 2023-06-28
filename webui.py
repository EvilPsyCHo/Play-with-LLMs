import os
import torch
from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html

from model import load_quantize_llm
# from utils import load_model_on_gpus


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    chatbot.append((parse_text(input), ""))
    if len(history) > 2 * MAX_TURNS:
        history = history[-MAX_TURNS:]
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


def parse_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model name optional [baichuan, chatGLM]")
    parser.add_argument("--model_ckpt", help="model checkpoint folder")
    parser.add_argument("--lora_ckpt", default=None, help="lora checkpoint folder")
    parser.add_argument("--max_turns", default=20, help="max multi-rounds chat turns")
    parser.add_argument("--quantize", default=None, help="quantization config optional [None, int4, int8]")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()
    MAX_TURNS = args.max_turns  # int参数无法传入，在外部全局定义
    name = args.model + (" lora" if args.lora_ckpt else "") + (f" quantize {args.quantize}" if args.quantize else "")
    model, tokenizer = load_quantize_llm(args.model, args.model_ckpt, args.quantize, torch.cuda.current_device())
    model = model.eval()

    with gr.Blocks() as demo:
        gr.HTML(f"""<h1 align="center">{name}</h1>""")
        chatbot = gr.Chatbot(scale=8)
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])
        past_key_values = gr.State(None)

        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                        [chatbot, history, past_key_values], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    gr.Chatbot.postprocess = postprocess
    demo.queue().launch(share=True, inbrowser=True)
