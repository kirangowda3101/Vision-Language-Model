import torch
import torch.nn as nn

# Reusing TransformerBlock from the image encoder: a transformer block just
# does self-attention + an MLP over a sequence of embed_dim-dimensional
# vectors -- it has no idea whether those vectors came from image patches
# or word/subword tokens. As long as the input is (B, seq_len, embed_dim),
# the exact same block architecture works for text.
from model_vit import TransformerBlock


class TextEncoder(nn.Module):
    """
    Text encoder ("text tower") for a CLIP-style model: turns a batch of
    token ID sequences into a single output_dim-dimensional vector per
    sequence, in the same shared embedding space the image tower's
    VisionTransformer projects into.
    """

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        num_heads,
        depth,
        mlp_ratio=4,
        output_dim=256,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Token embedding table: a learnable lookup table mapping each of
        # the vocab_size possible token IDs to an embed_dim-dimensional
        # vector. This is the text equivalent of the image side's Conv2d
        # patch projection -- both turn a discrete/raw input into a
        # learned embed_dim vector per token/patch.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable position embeddings, one per possible sequence position
        # up to max_seq_len. Like the ViT's position embeddings, these are
        # needed because self-attention has no inherent notion of token
        # order -- without them, "dog bites man" and "man bites dog" would
        # look identical to the attention layers.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        # Stack of `depth` transformer blocks, identical in architecture to
        # the ones used in VisionTransformer.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # Final LayerNorm, applied after all blocks (needed because the
        # blocks use pre-norm, so the last block's raw output is never
        # itself normalized).
        self.final_norm = nn.LayerNorm(embed_dim)

        # Projection head mapping embed_dim -> output_dim, into the same
        # shared image-text embedding space the VisionTransformer's
        # projection head produces. This is what lets a text embedding and
        # an image embedding be compared directly (e.g. cosine similarity).
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, token_ids):
        # token_ids: (B, seq_len), integer token IDs.
        B, seq_len = token_ids.shape

        # 1. Look up each token ID's embedding vector:
        # (B, seq_len) -> (B, seq_len, embed_dim)
        x = self.token_embedding(token_ids)

        # 2. Add position embeddings, sliced down to the actual seq_len
        # (which may be shorter than max_seq_len). Broadcasts over the
        # batch dimension.
        x = x + self.pos_embedding[:, :seq_len, :]

        # 3. Pass through each transformer block in turn, same as the
        # image encoder -- every token attends to every other token and
        # its representation gets updated, but the shape is preserved.
        for block in self.blocks:
            x = block(x)

        # 4. Normalize the final representations.
        x = self.final_norm(x)

        # 5. Take the LAST token's output as the sequence summary:
        # (B, seq_len, embed_dim) -> (B, embed_dim).
        # This mirrors GPT-style causal text encoders, where the last
        # token is the only one that (in a causal-attention setup) has
        # attended to every earlier token in the sequence, making it a
        # natural running summary of the whole input. (Note: this
        # implementation uses full/bidirectional attention like the image
        # side, so every token already sees every other token either way --
        # but keeping "last token" as the summary position matches common
        # practice and keeps the text/image towers structurally parallel:
        # image uses a dedicated prepended CLS token at index 0, text uses
        # a dedicated summary position at index -1.)
        summary = x[:, -1]

        # 6. Project into the shared image-text embedding space.
        return self.projection_head(summary)


if __name__ == "__main__":
    # Quick sanity check: random token IDs in, (B, output_dim) embedding out.
    text_encoder = TextEncoder(
        vocab_size=50257,
        max_seq_len=32,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    dummy_tokens = torch.randint(0, 50257, (2, 16))  # (B=2, seq_len=16)
    output = text_encoder(dummy_tokens)

    print(output.shape)  # expected: torch.Size([2, 256])
