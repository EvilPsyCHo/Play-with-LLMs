import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_quantize_llm(model, model_ckpt, quantize='4bit', local_rank=None):
    if model.lower().startswith("baichuan"):
        AutoLoad = AutoModelForCausalLM
    elif model.lower().startswith("chatglm"):
        AutoLoad = AutoModel
    else:
        NotImplementedError

    device_map = {"": torch.cuda.current_device()}
    # if we are in a distributed setting, we need to set the device map and max memory per device
    local_rank = os.environ.get('LOCAL_RANK', local_rank)
    if local_rank is not None:
        device_map = {'': int(local_rank)}
        print(f"local rank {local_rank} map to {local_rank}")
    if quantize == '4bit':
        model = AutoLoad.from_pretrained(
            model_ckpt,
            device_map=device_map,
            #                 load_in_4bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            ),
        )
    elif quantize == '8bit':
        model = AutoLoad.from_pretrained(
            model_ckpt,
            device_map=device_map,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    else:
        model = AutoLoad.from_pretrained(
            model_ckpt,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        print("pass unk_token_id to pad_token_id")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print(f'memory usage of model: {model.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
    return model, tokenizer