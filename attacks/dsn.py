from .nanogcg import DSNConfig, run_dsn
from models import get_llama, get_qwen2_vl


def DSN(
    model,
    tokenizer,
    messages,
    target,
    negative_target,
    alpha=10.0,
    num_steps=500,
    search_width=512,
    batch_size=256,
    topk=256,
):
    config = DSNConfig(
        alpha=alpha,
        num_steps=num_steps,
        search_width=search_width,
        batch_size=batch_size,
        topk=topk,
        verbosity="WARNING"
    )

    return run_dsn(model, tokenizer, messages, target, negative_target, config)


def DSN_llama(
    model,
    tokenizer,
    message,
    target,
    negative_target,
    alpha=10.0,
    num_steps=500,
    search_width=512,
    batch_size=256,
    topk=256,
):
    messages = [
        {"role": "user", "content": message + "{optim_str}"}
    ]

    return DSN(model, tokenizer, messages, target, negative_target, alpha, num_steps, search_width, batch_size, topk)


def DSN_qwen2_vl(
    model,
    processor,
    message,
    target,
    negative_target,
    alpha=10.0,
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

    return DSN(model, processor.tokenizer, messages, target, negative_target, alpha, num_steps, search_width, batch_size, topk)
