"""Training loop for the Shakespeare Mini-GPT.

Runs next-character language modeling with AdamW, linear learning-rate
warmup, gradient clipping, periodic train/val evaluation, and checkpoint
save to ``checkpoints/model.pt``.
"""

from __future__ import annotations

import os
import time

import torch

from dataset import CharacterTokenizer, get_batch, load_data
from model import GPT

# --- Architecture (compact laptop-scale Mini-LLM) ---
EMBEDDING_DIM: int = 128
NUM_HEADS: int = 4
NUM_LAYERS: int = 3
BLOCK_SIZE: int = 128
DROPOUT: float = 0.1

# --- Optimization ---
BATCH_SIZE: int = 32
MAX_ITERS: int = 500
EVAL_INTERVAL: int = 100
EVAL_BATCHES: int = 10
LEARNING_RATE: float = 3e-4
WARMUP_ITERS: int = 100

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH: str = "checkpoints/model.pt"


def get_learning_rate(iteration: int) -> float:
    """Returns the LR for ``iteration`` using a linear warmup then a plateau.

    Early gradients from random weights are noisy. A large step then can
    destabilize training. Warmup ramps 0 → ``LEARNING_RATE`` over
    ``WARMUP_ITERS``, then holds constant. GPT papers often add a later
    decay stage; this script keeps warmup + constant for simplicity.

    Args:
        iteration: Zero-based training step.

    Returns:
        Scalar learning rate for this step.
    """
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    return LEARNING_RATE


@torch.no_grad()
def evaluate(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
) -> dict[str, float]:
    """Estimates mean cross-entropy on train and validation splits.

    Args:
        model: GPT in any mode; this function sets eval then restores train.
        train_data: 1-D training token ids.
        val_data: 1-D validation token ids.

    Returns:
        Mapping ``{"train": loss, "val": loss}`` averaged over
        ``EVAL_BATCHES`` random batches each.
    """
    model.eval()
    results: dict[str, float] = {}

    for name, data in (("train", train_data), ("val", val_data)):
        total_loss = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()
        results[name] = total_loss / EVAL_BATCHES

    model.train()
    return results


@torch.no_grad()
def generate_sample(model: GPT, tokenizer: CharacterTokenizer) -> str:
    """Generates a short continuation from the prompt ``ROMEO:``.

    Used as a qualitative check while loss is still dropping.

    Args:
        model: Trained or partially trained GPT.
        tokenizer: Tokenizer fitted on the same corpus as training.

    Returns:
        Decoded sample string including the prompt.
    """
    model.eval()
    prompt = "ROMEO:"
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        prompt_ids, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)
    output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8)
    model.train()
    return tokenizer.decode(output_ids[0])


def train() -> None:
    """Loads data, trains the Mini-GPT, and writes a checkpoint.

    Each step: sample a batch, forward, backward, clip gradients, AdamW
    step. Every ``EVAL_INTERVAL`` steps, prints losses and a text sample.
    """
    print("=" * 60)
    print("Shakespeare GPT Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    print("\nLoading data...")
    train_data, val_data, tokenizer = load_data(block_size=BLOCK_SIZE)
    vocab_size = tokenizer.vocab_size

    print("\nCreating model...")
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...")
    print(f"Max iterations: {MAX_ITERS}")
    print(f"Eval interval: {EVAL_INTERVAL}\n")

    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()

    for iteration in range(MAX_ITERS):
        lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        x, y = x.to(DEVICE), y.to(DEVICE)

        _logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()

        # TR: Derin ağlarda gradyan patlamasını 1.0 normu ile kes.
        # EN: Clip global grad norm so a single step cannot explode weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            losses = evaluate(model, train_data, val_data)
            elapsed = time.time() - start_time
            print(
                f"Iter {iteration:5d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.0f}s"
            )
            if iteration > 0:
                print("\n--- Sample ---")
                print(generate_sample(model, tokenizer)[:400])
                print("--------------\n")

    losses = evaluate(model, train_data, val_data)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": MAX_ITERS,
        "val_loss": losses["val"],
        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "block_size": BLOCK_SIZE,
        },
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final val loss: {losses['val']:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
