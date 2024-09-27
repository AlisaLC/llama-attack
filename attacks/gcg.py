from .nanogcg import GCGConfig, run_gcg
from models import get_llama, get_qwen2_vl


def GCG(
    model,
    tokenizer,
    messages,
    target,
    search_width=512,
    batch_size=256,
    topk=256,
):
    config = GCGConfig(
        search_width=search_width,
        batch_size=batch_size,
        topk=topk,
        verbosity="WARNING"
    )

    return run_gcg(model, tokenizer, messages, target, config)


def GCG_llama(
    model,
    processor,
    message,
    target,
    search_width=512,
    batch_size=256,
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

    return GCG(model, processor.tokenizer, messages, target, search_width, batch_size, topk)


def GCG_qwen2_vl(
    model,
    processor,
    message,
    target,
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

    return GCG(model, processor.tokenizer, messages, target, search_width, batch_size, topk)
