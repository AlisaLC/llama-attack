import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, MllamaForConditionalGeneration, MllamaProcessor
from .utils import reverse_transformation_llama, reverse_transformation_qwen2_vl

def multi_modal_attack_qwen2_vl(
        model: Qwen2VLForConditionalGeneration,
        processor: Qwen2VLProcessor,
        img,
        message,
        target,
        negative,
        num_steps=100,
        lr=0.01,
        alpha=10,
    ):
    conversation_input = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": message},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target}
            ]
        }
    ]
    conversation_input_negative = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": message},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": negative}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(conversation_input)
    text_prompt_negative = processor.apply_chat_template(conversation_input_negative)
    expected_output_ids = processor.tokenizer([target], return_tensors="pt").input_ids.to("cuda")
    negative_output_ids = processor.tokenizer([negative], return_tensors="pt").input_ids.to("cuda")

    transform_to_pil = T.ToPILImage()

    inputs = processor(
        text=[text_prompt], images=[img], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs['pixel_values'] = inputs['pixel_values'].type(model.visual.get_dtype())
    inputs['pixel_values'] = inputs['pixel_values'].requires_grad_(True)
    inputs_negative = processor(
        text=[text_prompt_negative], images=[img], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs_negative['pixel_values'] = inputs['pixel_values']

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam([inputs['pixel_values']], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs = model(**inputs)
        logits = outputs.logits
        shift = inputs['input_ids'].shape[1] - expected_output_ids.shape[1]
        logits_shifted = logits[..., shift-1:-1, :].contiguous()

        target_loss = criterion(logits_shifted.view(-1, logits_shifted.size(-1)), expected_output_ids.view(-1))
        print("TARGET", target_loss.item())
        target_loss.backward()

        outputs = model(**inputs_negative)
        logits = outputs.logits
        shift = inputs_negative['input_ids'].shape[1] - negative_output_ids.shape[1]
        logits_shifted = logits[..., shift-1:-1, :].contiguous()
        
        probs = F.softmax(logits_shifted, dim=-1)
        negative_mask = F.one_hot(negative_output_ids, num_classes=probs.size(-1)).float().to("cuda")
        unlikelihood_loss = -alpha * torch.sum(negative_mask * torch.log(1 - probs + 1e-10))
        print("UNLIKELIHOOD", unlikelihood_loss.item())
        unlikelihood_loss.backward()
        
        optimizer.step()
        optimized_img_safe_tensor = inputs['pixel_values'].type(torch.float32).detach().cpu().numpy()
        optimized_img_safe_tensor = reverse_transformation_qwen2_vl(
            optimized_img_safe_tensor,
            inputs['image_grid_thw'][0, 0].item(),
            inputs['image_grid_thw'][0, 1].item(),
            inputs['image_grid_thw'][0, 2].item(),
            temporal_patch_size=2,
            patch_size=14,
            merge_size=2,
        )
        optimized_img_safe_pil = transform_to_pil(optimized_img_safe_tensor)
        yield optimized_img_safe_pil, target_loss.item() + unlikelihood_loss.item()

def multi_modal_attack_llama(
        model: MllamaForConditionalGeneration,
        processor: MllamaProcessor,
        img,
        message,
        target,
        negative,
        num_steps=100,
        lr=0.01,
        alpha=10,
    ):
    conversation_input = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": message},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": target}
            ]
        }
    ]
    conversation_input_negative = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": message},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": negative}
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(conversation_input)
    text_prompt_negative = processor.apply_chat_template(conversation_input_negative)
    expected_output_ids = processor.tokenizer([target], return_tensors="pt").input_ids.to("cuda")
    negative_output_ids = processor.tokenizer([negative], return_tensors="pt").input_ids.to("cuda")

    inputs = processor(
        text=[text_prompt], images=[img], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs['pixel_values'] = inputs['pixel_values'].type(model.vision_model.dtype)
    inputs['pixel_values'] = inputs['pixel_values'].requires_grad_(True)
    inputs_negative = processor(
        text=[text_prompt_negative], images=[img], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs_negative['pixel_values'] = inputs['pixel_values']

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam([inputs['pixel_values']], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs = model(**inputs)
        logits = outputs.logits
        shift = inputs['input_ids'].shape[1] - expected_output_ids.shape[1]
        logits_shifted = logits[..., shift-1:-1, :].contiguous()

        target_loss = criterion(logits_shifted.view(-1, logits_shifted.size(-1)), expected_output_ids.view(-1))
        target_loss.backward()

        outputs = model(**inputs_negative)
        logits = outputs.logits
        shift = inputs_negative['input_ids'].shape[1] - negative_output_ids.shape[1]
        logits_shifted = logits[..., shift-1:-1, :].contiguous()

        probs = F.softmax(logits_shifted, dim=-1)
        negative_mask = F.one_hot(negative_output_ids, num_classes=probs.size(-1)).float().to("cuda")
        unlikelihood_loss = -alpha * torch.sum(negative_mask * torch.log(1 - probs + 1e-10))
        unlikelihood_loss.backward()
        
        optimizer.step()
        optimized_img_safe_tensor = inputs['pixel_values'].type(torch.float32).detach().cpu().numpy()
        optimized_img_safe_pil = reverse_transformation_llama(
            optimized_img_safe_tensor,
            processor.image_processor.image_mean,
            processor.image_processor.image_std,
        )
        yield optimized_img_safe_pil, target_loss.item() + unlikelihood_loss.item()
