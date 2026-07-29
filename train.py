import argparse

import torch
from torch.optim import AdamW

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel, clip_contrastive_loss
from data_cifar import get_cifar_dataloader


def train(epochs=1, batch_size=32, lr=1e-4):
    # 1. Pick the fastest available device. Training on GPU (cuda) is
    # dramatically faster than CPU for a transformer this size, but the
    # code should still run (just slower) if no GPU is present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Image encoder ("image tower"): turns a batch of images into
    # (B, 256) embeddings.
    image_encoder = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    # 3. Text encoder ("text tower"): turns a batch of token ID sequences
    # into (B, 256) embeddings, in the same shared space as the image
    # tower's output.
    text_encoder = TextEncoder(
        vocab_size=50257,
        max_seq_len=32,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )

    # 4. Combine both towers into one CLIPModel and move ALL of its
    # parameters (both encoders + the logit_scale temperature) onto the
    # target device in one call.
    model = CLIPModel(image_encoder, text_encoder).to(device)

    # 5. CIFAR-10-based CLIP dataloader: yields (image_batch, token_ids_batch)
    # pairs, where each image is captioned with its class name (e.g.
    # "a photo of a dog"). max_seq_len=32 matches the TextEncoder's
    # max_seq_len above.
    dataloader = get_cifar_dataloader(
        batch_size=batch_size, max_seq_len=32, train=True
    )

    # 6. AdamW optimizer over every learnable parameter in the model
    # (both encoders' weights plus logit_scale). weight_decay=0.1 is a
    # relatively strong L2-style regularization commonly used for
    # transformer pretraining (as in CLIP/GPT), which helps prevent the
    # (many) parameters from growing unboundedly large.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    # 7. Training loop.
    step = 0
    for epoch in range(epochs):
        for images, token_ids in dataloader:
            # Move this batch's tensors onto the same device as the model.
            # (The model and its inputs must live on the same device for
            # any operation between them to work.)
            images = images.to(device)
            token_ids = token_ids.to(device)

            # Clear gradients from the PREVIOUS step before computing new
            # ones. PyTorch accumulates (sums) gradients into .grad by
            # default rather than overwriting them -- this is useful for
            # things like gradient accumulation across multiple
            # mini-batches, but for a normal step-by-step loop it means
            # that without zero_grad(), each step's gradients would keep
            # adding on top of every previous step's gradients, corrupting
            # the update. So we must always reset gradients to zero right
            # before computing a fresh set.
            optimizer.zero_grad()

            # Forward pass: encode this batch of images and captions and
            # compute the (B, B) similarity grid between them.
            logits = model(images, token_ids)

            # Contrastive loss: how well does each image's similarity
            # ranking pick out its own caption (and vice versa)?
            loss = clip_contrastive_loss(logits)

            # Backward pass: compute d(loss)/d(parameter) for every
            # learnable parameter in the model, via backpropagation. This
            # populates each parameter's .grad attribute. This step must
            # come AFTER zero_grad() (so we start from clean gradients)
            # and AFTER the forward pass (autograd needs the computation
            # graph built during the forward pass to know how to
            # backpropagate).
            loss.backward()

            # Parameter update: using the gradients just computed in
            # .grad, AdamW updates every parameter a small step in the
            # direction that reduces the loss (adjusted by AdamW's
            # per-parameter adaptive learning rates and weight decay).
            # This must come AFTER backward() -- there's nothing to apply
            # until gradients exist.
            #
            # In short, the strict order each step is:
            #   zero_grad() -> clear stale gradients from last step
            #   forward + loss -> compute what we got wrong
            #   backward()  -> compute how each parameter contributed to that
            #   step()      -> nudge each parameter to reduce that error
            optimizer.step()

            # Lightweight progress logging every 10 steps, so we can watch
            # the loss trend downward without flooding the console every
            # single step.
            if step % 10 == 0:
                print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")

            step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-style model on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
