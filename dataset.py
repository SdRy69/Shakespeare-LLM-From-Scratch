"""Tiny Shakespeare data pipeline and character-level tokenizer.

Loads (and if needed downloads) the corpus, maps each unique character to
an integer id, splits train/validation tensors, and yields next-character
batches for language-model training.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Union

import torch

# Tiny Shakespeare (Karpathy-style corpus, public-domain plays).
SHAKESPEARE_URL: str = (
    "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/"
    "refs/heads/main/input.txt"
)
DATA_PATH: str = "data/shakespeare.txt"

TokenIds = Union[list[int], torch.Tensor]


def download_shakespeare() -> None:
    """Downloads the corpus to ``DATA_PATH`` if the file is missing.

    Creates the ``data/`` directory as needed. Subsequent calls are no-ops
    when the file already exists.
    """
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return

    print("Downloading Shakespeare dataset...")
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)
    print(f"Downloaded to {DATA_PATH}")


class CharacterTokenizer:
    """Bidirectional map between characters and integer ids.

    Example::

        encode("hello")  -> [7, 4, 11, 11, 14]
        decode([...])    -> "hello"

    Character-level tokenization keeps the vocabulary tiny (~65 symbols on
    Tiny Shakespeare) and the pipeline easy to inspect. Production LLMs use
    subword tokenizers (BPE, WordPiece) to shorten sequences.

    Integer ids are arbitrary indices. The network never "reads" the integer
    as meaning; ``nn.Embedding`` looks it up and learns a vector. Characters
    that appear in similar contexts get similar vectors (Word2Vec idea)::

        char → id → embedding lookup → learned vector
         'h' →  7 → table[7]         → [0.23, -0.45, ...]

    Attributes:
        characters: Sorted unique characters in the corpus.
        vocab_size: Number of distinct characters.
        char_to_id: Character → integer map.
        id_to_char: Integer → character map.
    """

    def __init__(self, text: str) -> None:
        """Builds vocab tables from ``text``.

        Args:
            text: Full corpus used to discover the character set.
        """
        # TR: Vocab = metindeki benzersiz karakterler (harf, noktalama, \\n).
        # EN: One id per distinct character; no BPE merges.
        self.characters: list[str] = sorted(list(set(text)))
        self.vocab_size: int = len(self.characters)
        self.char_to_id: dict[str, int] = {
            char: index for index, char in enumerate(self.characters)
        }
        self.id_to_char: dict[int, str] = {
            index: char for index, char in enumerate(self.characters)
        }

        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Characters: {repr(''.join(self.characters[:50]))}...")

    def encode(self, text: str) -> list[int]:
        """Converts a string to a list of character ids.

        Args:
            text: Raw text.

        Returns:
            Integer ids, one per character, in order.
        """
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: TokenIds) -> str:
        """Converts character ids back to a string.

        Args:
            ids: Sequence of integer ids, or a 1-D ``torch.Tensor``.

        Returns:
            Decoded text.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        return "".join(self.id_to_char[token_id] for token_id in ids)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples random next-character windows from a 1-D token tensor.

    Language-model targets are the inputs shifted by one position. For
    ``block_size=5`` on ``"To be ..."``::

        x = ['T', 'o', ' ', 'b', 'e']
        y = ['o', ' ', 'b', 'e', ' ']   # x shifted by 1

    The model then learns: given ``T`` predict ``o``, given ``To`` predict
    space, and so on, for ``batch_size`` random starts at once.

    Args:
        data: 1-D tensor of token ids (train or val split).
        block_size: Context length of each sequence.
        batch_size: Number of sequences in the batch.

    Returns:
        ``x`` and ``y``, each of shape ``(batch_size, block_size)``.
    """
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    x_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for pos in positions:
        start = int(pos)
        x_list.append(data[start: start + block_size])
        y_list.append(data[start + 1: start + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x, y


def load_data(
    block_size: int = 256,
    train_split: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, CharacterTokenizer]:
    """Downloads the corpus, tokenizes it, and splits train/validation.

    The train/val cut is a contiguous split of the 1-D id tensor. Context
    length is applied later in :func:`get_batch`, not here.

    Args:
        block_size: Context length used by the caller (batches, not the split).
        train_split: Fraction of tokens used for training.

    Returns:
        Training ids, validation ids, and the fitted tokenizer.
    """
    download_shakespeare()

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"\nDataset size: {len(text):,} characters")
    print(f"Sample text:\n{text[:200]}")
    print("..." + "-" * 50)

    tokenizer = CharacterTokenizer(text)
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    print(f"\nTrain size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    return train_data, val_data, tokenizer


if __name__ == "__main__":
    train_data, val_data, tokenizer = load_data(block_size=128)
    x, y = get_batch(train_data, block_size=128, batch_size=4)

    print(f"\nSample batch:")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")

    print("\n--- Demonstrating input-target pairs ---")
    print(f"Input (x[0]):  {tokenizer.decode(x[0][:20])}...")
    print(f"Target (y[0]): {tokenizer.decode(y[0][:20])}...")
    print("Notice: target is input shifted by 1 character!")
