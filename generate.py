"""Inference script: load a checkpoint and generate Shakespeare-like text.

Restores architecture from the checkpoint config so training and sampling
stay in sync, then runs temperature-controlled autoregressive decoding.
"""

from __future__ import annotations

import torch

from dataset import DATA_PATH, CharacterTokenizer, download_shakespeare
from model import GPT


def load_model(
    checkpoint_path: str = "checkpoints/model.pt",
) -> tuple[GPT, CharacterTokenizer, str]:
    """Loads weights, rebuilds the GPT, and reconstructs the tokenizer.

    The tokenizer is refit on the same corpus file used at train time so
    character ids match the checkpoint vocabulary.

    Args:
        checkpoint_path: Path to the ``torch.save`` dict written by
            ``train.py``.

    Returns:
        A tuple ``(model, tokenizer, device)`` with the model in eval mode.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config = checkpoint["config"]

    download_shakespeare()
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()
    tokenizer = CharacterTokenizer(text)

    # TR: Üretimde dropout kapalı; stokastiklik yalnızca temperature + sample.
    # EN: Dropout is a train-time regularizer; inference uses dropout=0.
    model = GPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Model loaded!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {checkpoint['iteration']}")

    return model, tokenizer, device


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharacterTokenizer,
    device: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.8,
) -> str:
    """Encodes ``prompt``, samples new characters, and decodes the result.

    Autoregressive loop (one character per step)::

        "ROMEO:"  → predict 'O'
        "ROMEO:O" → predict ' '
        ... until max_tokens

    Temperature scales logits before softmax: ``0.5`` is conservative,
    ``0.8`` is the default balance, ``1.5`` is high-entropy.

    Args:
        model: GPT in eval mode.
        tokenizer: Matching character tokenizer.
        device: ``"cuda"`` or ``"cpu"``.
        prompt: Seed text (included in the returned string).
        max_tokens: Number of new characters to sample.
        temperature: Softmax temperature; must be > 0.

    Returns:
        Prompt plus generated continuation.
    """
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # (1, seq_len)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return tokenizer.decode(output_ids[0])


if __name__ == "__main__":
    print("=" * 60)
    print("Shakespeare GPT - Text Generation")
    print("=" * 60)

    model, tokenizer, device = load_model("checkpoints/model.pt")

    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "JULIET:\nO Romeo, ",
    ]

    print("\n" + "=" * 60)
    print("Generating Text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: {repr(prompt)} ---\n")
        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_tokens=300,
            temperature=0.8,
        )
        print(generated_text)
        print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("Try Your Own Prompts!")
    print("=" * 60)
    print("Enter a prompt to generate text, or 'quit' to exit.")
    print("Tip: Try character names like 'HAMLET:', 'KING:', 'First Citizen:'")

    while True:
        try:
            prompt = input("\nYour prompt: ")
            if prompt.lower() in ("quit", "exit", "q"):
                print("Farewell!")
                break
            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8,
            )
            print("\n" + generated_text)
        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break
