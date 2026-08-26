"""Decoder-only GPT for character-level Shakespeare generation.

This module implements a Mini-LLM in the GPT family: token and position
embeddings, a stack of Pre-LN decoder blocks with causal multi-head
attention, and a linear head that predicts the next character.

The original Transformer (Vaswani et al., 2017) is encoder-decoder with
three attention types (encoder self-attention, decoder masked attention,
cross-attention). GPT keeps only masked self-attention + feed-forward:

    Original decoder block          This GPT block
    +-------------------------+     +-------------------------+
    | Masked Self-Attention   | --> | Masked Self-Attention   |
    | Cross-Attention         |  X  |                         |
    | Feed-Forward            | --> | Feed-Forward            |
    +-------------------------+     +-------------------------+

Typical families: encoder-decoder (T5), decoder-only (GPT, LLaMA),
encoder-only (BERT).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Single decoder block: causal attention + MLP with Pre-LN residuals.

    Each sublayer is wrapped as ``x + sublayer(LayerNorm(x))`` (Pre-LN).
    Residual addition keeps a gradient highway; LayerNorm keeps activations
    on a stable scale before the expensive ops.

    Attributes:
        ln1: LayerNorm applied before multi-head attention.
        attention: Batched multi-head self-attention (Q=K=V).
        ln2: LayerNorm applied before the feed-forward network.
        mlp: Position-wise FFN with 4x inner width and GELU.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.1,
    ) -> None:
        """Builds one decoder block.

        Defaults (384 dim, 6 heads) target ~10M-parameter educational models.
        GPT-2 Small uses 768 dim / 12 heads; ``head_dim = embedding_dim /
        num_heads`` is typically 64 there. ``embedding_dim`` must be divisible
        by ``num_heads``.

        Args:
            embedding_dim: Residual stream width (d_model).
            num_heads: Parallel attention heads.
            dropout: Drop probability inside attention and the MLP.
        """
        super().__init__()

        # TR: Pre-LN — her alt katmandan ÖNCE normalize et (GPT-2/GPT-3).
        # EN: Pre-LN applies LayerNorm before the sublayer, not after.
        #     Post-LN (2017):  x → Attn → Add → LayerNorm
        #     Pre-LN (here):   x → LayerNorm → Attn → Add
        #     Pre-LN yields more stable gradients (Xiong et al., 2020).
        self.ln1 = nn.LayerNorm(embedding_dim)

        # TR: Q, K, V projeksiyonları ve scaled-dot-product PyTorch'ta.
        # EN: nn.MultiheadAttention owns Q/K/V projections and the scores.
        #     Self-attention: the same sequence is query, key, and value.
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (batch, sequence, embedding)
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        # TR: FFN 4× genişler, GELU, sonra d_model'e döner (Vaswani §3.3).
        # EN: Inner width is 4× d_model (512→2048 in the original paper).
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Applies masked self-attention and the MLP, each with a residual.

        Args:
            x: Hidden states of shape ``(batch, seq_len, embedding_dim)``.
            causal_mask: Boolean mask of shape ``(seq_len, seq_len)`` where
                ``True`` blocks a query-key pair (future positions).

        Returns:
            Tensor of the same shape as ``x``.
        """
        # TR: Residual (x + Attn) — sinyal ve gradyan kısa yoldan akar.
        # EN: Residual add is the skip connection around attention.
        x_norm = self.ln1(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask,
            is_causal=False,  # mask is supplied explicitly
        )
        x = x + attn_output

        # TR: İkinci residual: x + MLP(LayerNorm(x)).
        # EN: Same Pre-LN + residual pattern around the feed-forward net.
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Character-level decoder-only Transformer (Mini-GPT).

    Given a sequence of character ids, predicts the next character at every
    position. Training uses teacher forcing over the full context; generation
    samples one character at a time (autoregressive).

    Attributes:
        block_size: Maximum context length in characters.
        token_embedding: Learned lookup from character id to vector.
        position_embedding: Learned lookup from absolute position to vector.
        dropout: Dropout on the sum of token and position embeddings.
        blocks: Stack of :class:`TransformerBlock`.
        ln_final: LayerNorm before the vocabulary projection.
        output_proj: Linear map from d_model to vocab size.
        loss_fn: Token-level cross-entropy.
        causal_mask: Upper-triangular future-token mask (buffer, not a param).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """Constructs embeddings, decoder stack, head, and the causal mask.

        Args:
            vocab_size: Number of distinct character ids.
            embedding_dim: Residual stream width (d_model).
            num_heads: Attention heads per block.
            num_layers: Number of decoder blocks.
            block_size: Context window; also the position-embedding table size.
            dropout: Drop probability on embeddings, attention, and the MLP.
        """
        super().__init__()
        self.block_size = block_size

        # TR: Karakter ID → öğrenilen vektör. ID'ler keyfi; anlam embedding'te.
        # EN: nn.Embedding is a learnable lookup (same idea as nn.Parameter
        #     indexing). Characters that share context get similar vectors.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # TR: Mutlak konum gömme (sinüzoidal değil; GPT-2 tarzı öğrenilen tablo).
        # EN: Learned absolute positions, one vector per index in [0, block_size).
        self.position_embedding = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        # TR: Causal mask — eğitimde tüm dizi bir kerede verilir ama model
        #     geleceği göremez; aksi halde next-token "kopyalanır".
        # EN: True cells are blocked. Position i may attend to j only if j <= i.
        #
        #     Example "To b" (4 tokens):
        #                  to →  T    o    _    b
        #     from T:           ok  MASK MASK MASK
        #     from o:           ok   ok  MASK MASK
        #     from _:           ok   ok   ok  MASK
        #     from b:           ok   ok   ok   ok
        #
        #     Vaswani et al. §3.1: mask leftward flow to keep autoregression.
        causal_mask = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool),
            diagonal=1,
        )
        # Buffer: stored with the module, moved to GPU, not optimized.
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model created with {total_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes Linear and Embedding weights as N(0, 0.02).

        PyTorch defaults (Kaiming / unit Gaussian embeddings) are too large
        for deep transformers. GPT used N(0, 0.02) (Radford et al., 2018).

        Args:
            module: Child module visited by ``Module.apply``.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Runs the decoder stack and optionally computes next-token loss.

        Args:
            input_ids: Character ids of shape ``(batch, seq_len)``.
            targets: Next-character ids of the same shape. If omitted, loss
                is not computed (inference / generation).

        Returns:
            A pair ``(logits, loss)``. ``logits`` has shape
            ``(batch, seq_len, vocab_size)``. ``loss`` is a scalar tensor
            during training and ``None`` during generation.
        """
        _batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_emb = self.token_embedding(input_ids)

        # TR: Konum, token'ın dizideki yerini taşır ("ne" + "nerede").
        # EN: token_emb = what the character is; pos_emb = where it sits.
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        # Dropout on embedding sums: Vaswani et al. §5.4.
        x = self.dropout(token_emb + pos_emb)

        mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)

        # TR: Eğitimde target var → CrossEntropy. Üretimde target yok → loss=None.
        # EN: Flatten (B, T, V) → (B*T, V) because CrossEntropyLoss is 2D.
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            _batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(_batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(_batch_size * seq_len)
            loss = self.loss_fn(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Samples characters one by one and appends them to the prompt.

        Example: ``"ROMEO:"`` → ``'O'`` → ``"ROMEO:O"`` → ``' '`` → ...

        If the sequence exceeds ``block_size``, only the last ``block_size``
        tokens are fed in (position embeddings do not exist beyond that).

        Args:
            input_ids: Prompt ids of shape ``(batch, seq_len)``.
            max_new_tokens: Number of new characters to sample.
            temperature: Softmax temperature. Lower is greedier; higher is
                more random (``0.5`` / ``1.0`` / ``1.5``).

        Returns:
            Ids of shape ``(batch, seq_len + max_new_tokens)``.
        """
        for _ in range(max_new_tokens):
            if input_ids.size(1) <= self.block_size:
                current_input = input_ids
            else:
                current_input = input_ids[:, -self.block_size:]

            logits, _ = self.forward(current_input)

            # TR: Yalnızca son konumun logits'i "sıradaki karakter"dir.
            # EN: Autoregression reads the last time step only.
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)

            # TR: argmax tekrarlara yol açar; multinomial dağılımdan örnekler.
            # EN: Multinomial keeps diversity; argmax often loops ("the the").
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT(
        vocab_size=65,
        embedding_dim=384,
        num_heads=6,
        num_layers=6,
        block_size=256,
    ).to(device)

    batch_size = 4
    seq_len = 64
    dummy_input = torch.randint(0, 65, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, 65, (batch_size, seq_len)).to(device)

    logits, loss = model(dummy_input, dummy_targets)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    generated = model.generate(dummy_input[:1, :10], max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
