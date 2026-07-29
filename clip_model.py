import torch
import torch.nn as nn
import torch.nn.functional as F

from model_vit import VisionTransformer
from text_encoder import TextEncoder


class CLIPModel(nn.Module):
    """
    Ties an image encoder and a text encoder together into a CLIP-style
    model: both encoders project into the same output_dim-dimensional
    space, and this class computes a similarity grid between a batch of
    images and a batch of text sequences.
    """

    def __init__(self, image_encoder, text_encoder):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Learnable temperature, used to scale similarities before the
        # softmax in the contrastive loss. Sharper (larger) scaling makes
        # the model more confident/peaked; softer (smaller) scaling makes
        # it less confident.
        #
        # We store its LOG (log(1/0.07)) rather than the raw value, and
        # exponentiate it wherever it's used. Two reasons:
        #   1. Positivity: a temperature/scale must be positive, but an
        #      nn.Parameter is just a plain float with no constraint --
        #      unconstrained gradient updates could push a raw value
        #      negative or to zero, which would break or invert the
        #      similarity scaling. exp() of any real number is always
        #      positive, so optimizing the log value guarantees the actual
        #      scale used stays positive automatically.
        #   2. Stability: gradients w.r.t. a log-scale parameter correspond
        #      to *multiplicative* updates to the actual scale, which tends
        #      to be a better-conditioned/more stable optimization
        #      landscape than additive updates to the raw scale directly.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

    def forward(self, images, token_ids):
        # 1. Encode images: (B, 3, H, W) -> (B, output_dim)
        image_features = self.image_encoder(images)

        # 2. Encode text: (B, seq_len) -> (B, output_dim)
        text_features = self.text_encoder(token_ids)

        # 3. L2-normalize each embedding to unit length along the last
        # dimension. After this, the dot product between any two
        # embeddings equals their cosine similarity (a value in [-1, 1]
        # measuring how aligned their *directions* are), independent of
        # each vector's raw magnitude. This is what CLIP is designed to
        # compare -- we want "does this image point in the same direction
        # in embedding space as this caption", not "which vector is
        # longer".
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 4. Similarity grid: (B, output_dim) @ (output_dim, B) -> (B, B).
        # Entry [i, j] is the cosine similarity between image i and text
        # caption j, scaled by the learned temperature. If the batch is
        # constructed as (image_i, caption_i) matched pairs, then the
        # DIAGONAL entries [i, i] are the correct image-caption matches,
        # and every off-diagonal entry [i, j] (i != j) is an incorrect
        # pairing -- exactly the structure the contrastive loss below
        # exploits.
        logits = self.logit_scale.exp() * (image_features @ text_features.T)

        return logits


def clip_contrastive_loss(logits):
    """
    CLIP's symmetric contrastive loss: treats matching an image to its
    caption (and vice versa) as a classification problem, where the
    "classes" are the other items in the batch.
    """
    batch_size = logits.shape[0]

    # For row i (image i) and column i (caption i), the diagonal is always
    # the correct match. So the "label" (correct class index) for row i is
    # simply i itself -- labels = [0, 1, 2, ..., batch_size - 1].
    labels = torch.arange(batch_size, device=logits.device)

    # Image-to-text direction: treat each ROW of `logits` (image i's
    # similarity to every caption) as a set of class scores over
    # batch_size possible captions, and classify which caption is correct.
    # cross_entropy applies softmax over the last dim internally, so this
    # is "softmax over captions, pick out the true caption's probability".
    loss_i2t = F.cross_entropy(logits, labels)

    # Text-to-image direction: same idea but transposed, so each COLUMN of
    # the original grid becomes a row -- i.e. treat each caption's
    # similarity to every image as class scores, and classify which image
    # is correct.
    loss_t2i = F.cross_entropy(logits.T, labels)

    # Average both directions. We need both because the loss is not
    # symmetric on its own: loss_i2t only pushes each image to prefer its
    # own caption over other captions in the batch, while loss_t2i only
    # pushes each caption to prefer its own image over other images.
    # Averaging trains both encoders to agree in both directions.
    return (loss_i2t + loss_t2i) / 2


if __name__ == "__main__":
    image_encoder = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    text_encoder = TextEncoder(
        vocab_size=50257,
        max_seq_len=32,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    clip_model = CLIPModel(image_encoder, text_encoder)

    dummy_images = torch.randn(4, 3, 224, 224)  # (B=4, C=3, H=224, W=224)
    dummy_tokens = torch.randint(0, 50257, (4, 16))  # (B=4, seq_len=16)

    logits = clip_model(dummy_images, dummy_tokens)
    loss = clip_contrastive_loss(logits)

    print(logits.shape)  # expected: torch.Size([4, 4])
    print(loss.item())  # a single scalar loss value
