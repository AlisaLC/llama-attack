import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def get_llama(model_path=os.getenv('LLAMA_PATH')):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer
