import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects
    each patch into an embedding vector.
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        # Number of patches along one side is (image_size // patch_size).
        # Squaring that gives the total number of patches in the image.
        self.num_patches = (image_size // patch_size) ** 2

        # The "patchify + linear projection" trick:
        #
        # Conceptually, ViT patch embedding works like this:
        #   1. Cut the image into a grid of (patch_size x patch_size) patches.
        #   2. Flatten each patch into a vector of length
        #      (patch_size * patch_size * in_channels).
        #   3. Multiply that vector by a learned weight matrix (a linear
        #      layer) to project it into an embed_dim-dimensional vector.
        #
        # A Conv2d with kernel_size=patch_size and stride=patch_size does
        # exactly this in one shot:
        #   - Because stride == kernel_size, each patch_size x patch_size
        #     window the kernel slides over is non-overlapping, so the
        #     conv naturally "cuts" the image into a grid of patches.
        #   - At each patch location, a conv computes a dot product between
        #     its (in_channels, patch_size, patch_size) kernel weights and
        #     the patch's pixels -- which is mathematically identical to
        #     flattening the patch into a vector and multiplying it by a
        #     weight matrix (i.e. a linear layer).
        #   - Using out_channels=embed_dim gives us embed_dim independent
        #     kernels, i.e. embed_dim independent "linear projections" of
        #     each patch, computed in parallel.
        #
        # So this single Conv2d simultaneously does the patch splitting
        # and the per-patch linear projection.
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)

        # After conv: (B, embed_dim, H/patch_size, W/patch_size)
        # Each spatial position in this output corresponds to one patch,
        # and its embed_dim values are that patch's embedding vector.
        x = self.projection(x)

        # Flatten the two spatial dimensions into a single "num_patches"
        # dimension: (B, embed_dim, H/patch_size, W/patch_size) -> (B, embed_dim, num_patches)
        x = x.flatten(2)

        # Rearrange to put num_patches before embed_dim, matching the
        # (B, num_patches, embed_dim) shape expected by a transformer:
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class PatchEmbeddingWithTokens(nn.Module):
    """
    Wraps PatchEmbedding and adds:
      1. A learnable [CLS] token prepended to the patch sequence.
      2. Learnable position embeddings added to every token.

    This matches the input representation used at the start of a ViT,
    right before the transformer encoder blocks.
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()

        # Reuse PatchEmbedding to turn the image into (B, num_patches, embed_dim).
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # The [CLS] token is a single learnable vector, shared across all
        # images. It has shape (1, 1, embed_dim) so it can later be
        # "expanded" (broadcast-copied) to (B, 1, embed_dim) for any batch
        # size B. Its purpose is to act as a summary token: after the
        # transformer encoder, this token's output embedding is typically
        # used as the whole image's representation (e.g. for classification).
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Position embeddings tell the model where each token is located,
        # since (unlike a conv/RNN) a transformer's attention has no
        # built-in notion of order or position -- it treats the input as an
        # unordered set of tokens unless we inject position information.
        #
        # Shape is (1, num_patches + 1, embed_dim): one position vector for
        # the CLS token plus one for each patch. Like cls_token, the leading
        # 1 lets this broadcast across the batch dimension.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: (B, in_channels, H, W)
        batch_size = x.shape[0]

        # (B, num_patches, embed_dim)
        x = self.patch_embedding(x)

        # Expand the single cls_token (1, 1, embed_dim) into a copy for
        # every image in the batch: (B, 1, embed_dim). This doesn't
        # allocate new memory for each copy -- it's a broadcasted view --
        # but from here on it behaves like a per-image tensor.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Prepend the CLS token to the patch sequence along the sequence
        # dimension (dim=1): (B, 1, embed_dim) + (B, num_patches, embed_dim)
        # -> (B, num_patches + 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings to every token (broadcasts over the
        # batch dimension since pos_embedding has shape (1, num_patches+1, embed_dim)).
        x = x + self.pos_embedding

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention, implemented from scratch (no
    nn.MultiheadAttention), following the standard "Attention is All You
    Need" formulation:

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(head_dim)) @ V

    computed independently for `num_heads` heads and then concatenated.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Each head only gets to see a slice of embed_dim, so embed_dim
        # must split evenly across heads.
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Combined Q, K, V projection: instead of three separate
        # nn.Linear(embed_dim, embed_dim) layers, we use a single
        # nn.Linear(embed_dim, embed_dim * 3) and split its output into
        # three chunks afterward. This is mathematically identical to three
        # separate projections, but is one matrix multiply instead of
        # three, which is faster on GPU.
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

        # After concatenating all heads' outputs back together, this
        # projection lets the model mix information across heads before
        # returning the result.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        B, seq_len, embed_dim = x.shape

        # 1. Project x to Q, K, V simultaneously: (B, seq_len, embed_dim * 3)
        qkv = self.qkv_proj(x)

        # Split the last dimension into three equal chunks of size embed_dim:
        # q, k, v each (B, seq_len, embed_dim).
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape each of q, k, v so the embed_dim axis is split into
        # (num_heads, head_dim), giving every head its own independent
        # slice of the embedding to attend with:
        # (B, seq_len, embed_dim) -> (B, seq_len, num_heads, head_dim)
        # Then move num_heads before seq_len so that matrix multiplication
        # (which operates on the last two dims) is done independently per
        # head, per batch element:
        # (B, seq_len, num_heads, head_dim) -> (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Attention scores: for every pair of tokens (i, j), how much
        # should token i attend to token j? Q @ K^T computes a dot product
        # between every query and every key:
        # (B, num_heads, seq_len, head_dim) @ (B, num_heads, head_dim, seq_len)
        # -> (B, num_heads, seq_len, seq_len)
        #
        # We divide by sqrt(head_dim) (the "scaled" in scaled dot-product
        # attention) because as head_dim grows, dot products tend to grow
        # in magnitude too, pushing softmax into regions with extremely
        # small gradients (very peaked/saturated outputs). Scaling by
        # sqrt(head_dim) keeps the variance of the scores roughly constant
        # regardless of head_dim, which keeps training stable.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Softmax over the last dimension (the "key" dimension) so that,
        # for each query token, the attention weights over all key tokens
        # sum to 1.
        attn_weights = attn_scores.softmax(dim=-1)

        # 4. Use the attention weights to take a weighted average of the
        # value vectors: (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, head_dim)
        # -> (B, num_heads, seq_len, head_dim)
        attn_output = attn_weights @ v

        # 5. Undo the head split: move num_heads back next to head_dim,
        # then merge them back into a single embed_dim axis.
        # (B, num_heads, seq_len, head_dim) -> (B, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        # .contiguous() is needed because .transpose() only changes strides
        # (creates a non-contiguous view), and .view() requires contiguous
        # memory to reinterpret the shape.
        # (B, seq_len, num_heads, head_dim) -> (B, seq_len, embed_dim)
        attn_output = attn_output.contiguous().view(B, seq_len, embed_dim)

        # Final linear projection to mix information across heads.
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    One transformer encoder block, using the pre-norm pattern:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-norm (normalizing BEFORE attention/MLP, rather than after) means
    the residual path (the "x +" part) is a clean, unnormalized running sum
    of the input and every sublayer's output. This keeps gradients flowing
    smoothly through many stacked blocks during backpropagation, which in
    the original (post-norm) transformer design tended to make deep stacks
    unstable/harder to train without careful learning-rate warmup.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        # Residual connection #1: add the attention sublayer's output back
        # to its input. Residual connections let gradients (and
        # information) skip directly around the attention computation,
        # which is essential for training deep stacks of these blocks.
        x = x + self.attention(self.norm1(x))

        # Residual connection #2: same idea, around the MLP sublayer. The
        # MLP expands to a wider hidden dimension (embed_dim * mlp_ratio)
        # and projects back down, giving the model extra per-token
        # (non-attention) capacity to transform each token's representation.
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Full Vision Transformer image encoder: patch + CLS + position
    embedding, a stack of transformer blocks, a final LayerNorm, and a
    projection head mapping the CLS token to the output embedding space.

    This is the "image tower" half of a CLIP-style model -- it turns an
    image into a single output_dim-dimensional vector that can be compared
    (e.g. via cosine similarity) against a text tower's output vector.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        num_heads,
        depth,
        output_dim,
        mlp_ratio=4,
    ):
        super().__init__()

        # Turns (B, in_channels, H, W) into (B, num_patches + 1, embed_dim),
        # i.e. a sequence of patch tokens plus a prepended CLS token, all
        # with position embeddings already added.
        self.patch_embedding = PatchEmbeddingWithTokens(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Stack of `depth` transformer blocks. nn.ModuleList (rather than a
        # plain Python list) is required so PyTorch registers each block's
        # parameters -- a plain list would make them invisible to
        # .parameters()/.to(device)/optimizers.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        # A final LayerNorm applied after all transformer blocks. Since we
        # use pre-norm blocks (each block normalizes its own inputs, not
        # its outputs), the very last block's output is never normalized
        # before we read out the CLS token -- this final norm handles that.
        self.final_norm = nn.LayerNorm(embed_dim)

        # Projection head: maps the embed_dim-dimensional CLS token into
        # output_dim, the shared embedding space CLIP uses to compare
        # images and text. embed_dim is an internal size chosen for model
        # capacity, and it generally won't match the text tower's internal
        # size either -- this linear layer adapts both towers' outputs into
        # one common space where dot products/cosine similarity are
        # meaningful.
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x: (B, in_channels, H, W)

        # (B, num_patches + 1, embed_dim)
        x = self.patch_embedding(x)

        # Pass the token sequence through each transformer block in turn.
        # Every block attends across all tokens (including CLS) and
        # updates each token's representation, but preserves the shape.
        for block in self.blocks:
            x = block(x)

        # Normalize the final representations.
        x = self.final_norm(x)

        # Take only the CLS token (sequence index 0) as the image's overall
        # representation: (B, num_patches + 1, embed_dim) -> (B, embed_dim).
        # We use the CLS token (rather than, say, averaging all patch
        # tokens) because self-attention lets it freely gather information
        # from every patch token across every block -- by design, it's
        # trained to accumulate a summary of the whole image, which is
        # exactly the "one vector per image" representation CLIP needs.
        cls_output = x[:, 0]

        # Project into the shared image-text embedding space.
        return self.projection_head(cls_output)


if __name__ == "__main__":
    # Quick sanity check: a full VisionTransformer should turn a batch of
    # 224x224 RGB images directly into (B, output_dim) embeddings.
    vision_transformer = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    dummy_images = torch.randn(2, 3, 224, 224)  # (B=2, C=3, H=224, W=224)
    output = vision_transformer(dummy_images)

    print(output.shape)  # expected: torch.Size([2, 256])
