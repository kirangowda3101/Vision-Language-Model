import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel
from data_cifar import CIFAR10_CLASSES, get_class_captions, image_transform


def build_model(checkpoint_path):
    """
    Reconstructs the exact same CLIPModel architecture used in training.py
    (same VisionTransformer/TextEncoder hyperparameters), loads the trained
    weights from a checkpoint saved by train.py's save_checkpoint(), and
    puts the model in eval mode on the fastest available device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = CLIPModel(image_encoder, text_encoder)

    # train.py's checkpoints are a dict with keys "model", "optimizer",
    # "epoch", "global_step" (see save_checkpoint in train.py) -- for
    # inference we only need the model weights, under the "model" key.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)

    # eval() disables training-only behavior (e.g. dropout, if any were
    # added later) so inference is deterministic and doesn't update any
    # running statistics.
    model.eval()

    return model, device


def main():
    parser = argparse.ArgumentParser(description="Zero-shot CIFAR-10 classification with a trained CLIP model")
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser("~/checkpoints/clip_flickr/best.pt"),
    )
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    model, device = build_model(args.checkpoint)
    print(f"Using device: {device}")

    # --- Encode the 10 candidate class captions ONCE, up front -----------
    # Zero-shot classification flips CLIP's usual (image, caption) pairing
    # around: instead of one caption per image, we build a fixed "menu" of
    # 10 candidate captions (one per CIFAR-10 class, e.g. "a photo of a
    # dog") and ask, for every test image, "which of these 10 captions is
    # this image most similar to?" Since that menu never changes across
    # images, we only run it through the text encoder ONE time here and
    # reuse the resulting (10, 256) label matrix for every image/batch --
    # there's no reason to re-encode "a photo of a dog" hundreds of times
    # just because hundreds of different images are being classified.
    label_token_ids = get_class_captions(max_seq_len=32).to(device)  # (10, 32)

    with torch.no_grad():
        label_features = model.text_encoder(label_token_ids)  # (10, 256)

        # L2-normalize so that comparing embeddings via a dot product gives
        # cosine similarity (how aligned two vectors' DIRECTIONS are, in
        # [-1, 1]) rather than being skewed by whichever embedding happens
        # to have a larger raw magnitude. This is the same normalization
        # CLIPModel.forward() applies internally during training.
        label_features = F.normalize(label_features, dim=-1)

    # --- Load CIFAR-10 test images ----------------------------------------
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=image_transform)

    num_images = min(args.num_images, len(test_dataset))
    # Subset restricts iteration to just the first `num_images` examples,
    # so we don't have to run inference over the entire 10,000-image test
    # set every time this demo is run.
    subset = Subset(test_dataset, range(num_images))
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    correct = 0
    total = 0
    example_predictions = []  # first ~10 (true_class_idx, predicted_class_idx) pairs

    # No gradients needed for inference -- this avoids building the
    # autograd graph and reduces memory usage.
    with torch.no_grad():
        for images, true_labels in dataloader:
            images = images.to(device)
            true_labels = true_labels.to(device)

            # Encode this batch of images: (B, 3, 224, 224) -> (B, 256).
            image_features = model.image_encoder(images)
            image_features = F.normalize(image_features, dim=-1)

            # Compare every image in this batch against all 10
            # pre-computed label embeddings in one matrix multiply:
            # (B, 256) @ (256, 10) -> (B, 10) cosine similarities, where
            # column j is each image's similarity to class j's caption.
            similarity = image_features @ label_features.T

            # The predicted class for each image is whichever of the 10
            # captions it's most similar to.
            predicted_labels = similarity.argmax(dim=-1)  # (B,)

            correct += (predicted_labels == true_labels).sum().item()
            total += true_labels.size(0)

            if len(example_predictions) < 10:
                for true_idx, pred_idx in zip(true_labels.tolist(), predicted_labels.tolist()):
                    if len(example_predictions) >= 10:
                        break
                    example_predictions.append((true_idx, pred_idx))

    print("\nExample predictions:")
    for true_idx, pred_idx in example_predictions:
        print(f"true: {CIFAR10_CLASSES[true_idx]} | predicted: {CIFAR10_CLASSES[pred_idx]}")

    accuracy = 100.0 * correct / total
    print(
        f"\nZero-shot accuracy on {total} CIFAR-10 images: {accuracy:.1f}% "
        f"(random baseline: 10%)"
    )


if __name__ == "__main__":
    main()
