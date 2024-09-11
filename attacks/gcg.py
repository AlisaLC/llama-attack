import nanogcg

from nanogcg import GCGConfig


def GCG(model, tokenizer, message, target):
    messages = [
        {"role": "user", "content": message + "{optim_str}"}
    ]

    config = GCGConfig(
        num_steps=500,
        search_width=512,
        batch_size=256,
        topk=256,
        verbosity="WARNING"
    )

    return nanogcg.run(model, tokenizer, messages, target, config)
