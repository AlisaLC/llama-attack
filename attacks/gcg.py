from .nanogcg import GCGConfig, run
from models import get_llama


def GCG(
        message,
        target,
        num_steps=500,
        search_width=512,
        batch_size=256,
        topk=256,
    ):

    model, tokenizer = get_llama()

    messages = [
        {"role": "user", "content": message + "{optim_str}"}
    ]

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        batch_size=batch_size,
        topk=topk,
        verbosity="WARNING"
    )

    return run(model, tokenizer, messages, target, config)
