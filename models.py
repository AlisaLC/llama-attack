import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, Qwen2VLProcessor, MllamaProcessor
import os
from dotenv import load_dotenv
load_dotenv()


def get_llama(model_path=os.getenv('LLAMA_PATH')):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
    )
    return model, processor


def get_qwen2_vl(model_path=os.getenv('QWEN2_VL_PATH')):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def generate_openai(client, openai_model, messages: list[dict[str, str]], max_tokens=128):
    return client.chat.completions.create(
        model=openai_model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )


def generate_llama(
    model: MllamaForConditionalGeneration,
    processor: MllamaProcessor,
    messages,
    max_tokens=128,
):
    input_text = processor.apply_chat_template(
        messages, add_generation_prompt=True)
    inputs = processor(text=input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs, max_new_tokens=max_tokens, pad_token_id=0)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def generate_llama_with_image(model, processor, messages, images, max_tokens=128):
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], images=images, padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def generate_qwen2_vl(model, processor, messages, max_tokens=128):
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def generate_qwen2_vl_with_image(model, processor, messages, images, max_tokens=128):
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt], images=images, padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]
