from .nanogcg import GCGConfig, run
from models import get_llama, get_qwen2_vl


def GCG(
    model,
    tokenizer,
    messages,
    target,
    num_steps=500,
    search_width=512,
    batch_size=256,
    topk=256,
):
    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        batch_size=batch_size,
        topk=topk,
        verbosity="WARNING"
    )

    return run(model, tokenizer, messages, target, config)


def GCG_llama(
    model,
    tokenizer,
    message,
    target,
    num_steps=500,
    search_width=512,
    batch_size=256,
    topk=256,
):
    messages = [
        {"role": "user", "content": message + "{optim_str}"}
    ]

    return GCG(model, tokenizer, messages, target, num_steps, search_width, batch_size, topk)


def GCG_qwen2_vl(
    model,
    processor,
    message,
    target,
    num_steps=500,
    search_width=512,
    batch_size=128,
    topk=256,
):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message + "{optim_str}",
                }
            ],
        },
    ]

    return GCG(model, processor.tokenizer, messages, target, num_steps, search_width, batch_size, topk)
