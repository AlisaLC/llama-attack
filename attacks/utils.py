from PIL import Image
import numpy as np

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