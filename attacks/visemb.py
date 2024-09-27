import torch
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, MllamaForConditionalGeneration, MllamaProcessor

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def reverse_transformation_qwen2_vl(flatten_patches, grid_t, grid_h, grid_w, temporal_patch_size, merge_size, patch_size):
    patches = flatten_patches.reshape(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        3,
        temporal_patch_size,
        patch_size,
        patch_size,
    )

    patches = patches.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)

    patches = patches.reshape(
        grid_t * temporal_patch_size,
        3,
        grid_h * patch_size,
        grid_w * patch_size,
    )

    patches = patches[:1]
    patches = patches[0].transpose(1, 2, 0)
    patches = patches * OPENAI_CLIP_STD + OPENAI_CLIP_MEAN

    return patches

def reverse_transformation_llama(patches, mean, std):
    patches = patches[0, 0, 0]
    patches = patches.transpose(1, 2, 0)
    patches = patches * std + mean
    patches *= 255
    patches = np.clip(patches, 0, 255)
    patches = patches.astype(np.uint8)
    patches = Image.fromarray(patches)
    return patches



def similar_visual_embedding_attack_qwen2_vl(
        model: Qwen2VLForConditionalGeneration,
        processor: Qwen2VLProcessor,
        img_safe,
        img_unsafe,
        num_steps=100,
        lr=0.01,
    ):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs_unsafe = processor(
        text=[text_prompt], images=[img_unsafe], padding=True, return_tensors="pt"
    )
    inputs_unsafe = inputs_unsafe.to("cuda")
    inputs_unsafe['pixel_values'] = inputs_unsafe['pixel_values'].type(model.visual.get_dtype())

    with torch.no_grad():
        image_embeds_unsafe = model.visual(inputs_unsafe['pixel_values'], grid_thw=inputs_unsafe['image_grid_thw']).to("cuda")

    transform_to_pil = T.ToPILImage()

    inputs_safe = processor(
        text=[text_prompt], images=[img_safe], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs_safe['pixel_values'] = inputs_safe['pixel_values'].type(model.visual.get_dtype())
    inputs_safe['pixel_values'] = inputs_safe['pixel_values'].requires_grad_(True)

    optimizer = optim.Adam([inputs_safe['pixel_values']], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        image_embeds_safe = model.visual(inputs_safe['pixel_values'], grid_thw=inputs_safe['image_grid_thw'])
        loss = torch.nn.functional.mse_loss(image_embeds_safe, image_embeds_unsafe)
        loss.backward()
        optimizer.step()
        optimized_img_safe_tensor = inputs_safe['pixel_values'].type(torch.float32).detach().cpu().numpy()
        optimized_img_safe_tensor = reverse_transformation_qwen2_vl(
            optimized_img_safe_tensor,
            inputs_safe['image_grid_thw'][0, 0].item(),
            inputs_safe['image_grid_thw'][0, 1].item(),
            inputs_safe['image_grid_thw'][0, 2].item(),
            temporal_patch_size=2,
            patch_size=14,
            merge_size=2,
        )
        optimized_img_safe_pil = transform_to_pil(optimized_img_safe_tensor)
        yield optimized_img_safe_pil, loss.item()

def similar_visual_embedding_attack_llama(
        model: MllamaForConditionalGeneration,
        processor: MllamaProcessor,
        img_safe,
        img_unsafe,
        num_steps=100,
        lr=0.01,
    ):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs_unsafe = processor(
        text=[text_prompt], images=[img_unsafe], padding=True, return_tensors="pt"
    )
    inputs_unsafe = inputs_unsafe.to("cuda")
    inputs_unsafe['pixel_values'] = inputs_unsafe['pixel_values'].type(model.vision_model.dtype)

    with torch.no_grad():
        image_embeds_unsafe = model.vision_model(
            pixel_values=inputs_unsafe['pixel_values'],
            aspect_ratio_ids=inputs_unsafe['aspect_ratio_ids'],
            aspect_ratio_mask=inputs_unsafe['aspect_ratio_mask'],
        )[0].to("cuda")

    inputs_safe = processor(
        text=[text_prompt], images=[img_safe], padding=True, return_tensors="pt"
    ).to("cuda")
    inputs_safe['pixel_values'] = inputs_safe['pixel_values'].type(model.vision_model.dtype)
    inputs_safe['pixel_values'] = inputs_safe['pixel_values'].requires_grad_(True)

    optimizer = optim.Adam([inputs_safe['pixel_values']], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        image_embeds_safe = model.vision_model(
            pixel_values=inputs_safe['pixel_values'],
            aspect_ratio_ids=inputs_safe['aspect_ratio_ids'],
            aspect_ratio_mask=inputs_safe['aspect_ratio_mask'],
        )[0].to("cuda")
        loss = torch.nn.functional.mse_loss(image_embeds_safe, image_embeds_unsafe)
        loss.backward()
        optimizer.step()
        optimized_img_safe_tensor = inputs_safe['pixel_values'].type(torch.float32).detach().cpu().numpy()
        optimized_img_safe_pil = reverse_transformation_llama(
            optimized_img_safe_tensor,
            processor.image_processor.image_mean,
            processor.image_processor.image_std,
        )
        yield optimized_img_safe_pil, loss.item()
