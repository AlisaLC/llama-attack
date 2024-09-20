import torch
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def reverse_transformation(flatten_patches, grid_t, grid_h, grid_w, temporal_patch_size, merge_size, patch_size):
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


def similar_visual_embedding_attack(model, processor, img_safe, img_unsafe, num_steps=100, lr=0.01):
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
        optimized_img_safe_tensor = reverse_transformation(
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
