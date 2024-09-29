import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, MllamaProcessor, AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
load_dotenv()

MM_LLAMA_MODELS = {
    'Llama 3.2 11B': '',
    'Llama 3.2 90B': '',
}

MM_QWEN_MODELS = {
    'Qwen2 VL 2B': '',
    'Qwen2 VL 7B': '',
    'Qwen2 VL 72B': '',

}

LLAMA_MODELS = {
    'Llama 3.2 1B': '',
    'Llama 3.2 3B': '',
    'Llama 3.2 11B': '',
    'Llama 3.2 90B': '',
    'Llama 3.1 8B': '',
    'Llama 3.1 90B': '',
    'Llama 3.1 405B': '',
    'Llama 3 8B': '',
    'Llama 3 70B': '',
    'Llama 2 7B': '',
    'Llama 2 13B': '',
    'Llama 2 70B': '',
}


def get_llama_mm(model_path=LLAMA_MODELS['Llama 3.2 11B']):
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
    )
    return model, processor


def get_llama(model_path=LLAMA_MODELS['Llama 3.1 8B']):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    return model, tokenizer


def get_qwen2_vl(model_path=MM_QWEN_MODELS['Qwen2 VL 7B']):
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


def generate_llama_mm(
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
