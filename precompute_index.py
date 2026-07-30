import argparse
import os

# Same reasoning as data_flickr.py / search_demo.py / app.py: redirect
# HuggingFace's cache to scratch space before importing `datasets`, so
# downloading Flickr30k doesn't blow through the home directory quota.
# This must happen before `import datasets`/`load_dataset`.
_HF_CACHE_DIR = "/scratch/ramanagarajayaram.k/hf_cache"
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = _HF_CACHE_DIR

import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel

MAX_SEQ_LEN = 32

# Preprocessing pipeline fed to the model -- same as data_flickr.py /
# search_demo.py / app.py: resize to the ViT's expected input size,
# convert to a float tensor in [0, 1], then normalize to roughly [-1, 1]
# per channel.
model_image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

# Separate, much smaller resize used ONLY for the copy of each image we
# save for display in the Gradio app later. The app never needs to feed
# these display images back into the model, so there's no reason to save
# them at full/224px resolution -- keeping them small (256x256) keeps the
# index file's size (and the app's memory/bandwidth) down.
DISPLAY_IMAGE_SIZE = (256, 256)


def get_caption(example):
    """Best-effort real caption for a Flickr30k row (same fallback as data_flickr.py)."""
    caption_list = example.get("original_alt_text")
    if caption_list:
        return caption_list[0]
    return example.get("alt_text", "<no caption available>")


def build_model(checkpoint_path):
    """
    Reconstructs the exact same CLIPModel architecture used in train.py,
    loads the trained weights, and puts the model in eval mode on the
    fastest available device. Identical to app.py's / zero_shot.py's
    build_model().
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
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=512,
        num_heads=8,
        depth=6,
        output_dim=256,
    )
    model = CLIPModel(image_encoder, text_encoder)

    # train.py's checkpoints are a dict with keys "model", "optimizer",
    # "epoch", "global_step" -- for inference we only need the weights,
    # under the "model" key.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    return model, device


def main():
    parser = argparse.ArgumentParser(
        description="Precompute a Flickr30k image search index (image embeddings + display images + captions)"
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser("~/checkpoints/clip_flickr_v2/best.pt"),
    )
    parser.add_argument("--num_images", type=int, default=300)
    parser.add_argument("--output", default="index.pt")
    args = parser.parse_args()

    model, device = build_model(args.checkpoint)
    print(f"Using device: {device}")

    print(f"Loading {args.num_images} Flickr30k images...")
    dataset = load_dataset(
        "Mozilla/flickr30k-transformed-captions", split="test", cache_dir=_HF_CACHE_DIR
    )
    num_images = min(args.num_images, len(dataset))
    dataset = dataset.select(range(num_images))

    # Parallel lists, indexed identically to each other and to the rows of
    # `image_features` below: index i always refers to the same image
    # across all three.
    captions = []
    display_images = []
    model_input_tensors = []

    for i in range(num_images):
        example = dataset[i]
        image = example["image"].convert("RGB")

        captions.append(get_caption(example))
        # A small, display-only copy for the Gradio app's gallery -- kept
        # separate from the (224, 224) tensor actually fed to the model.
        display_images.append(image.resize(DISPLAY_IMAGE_SIZE))
        model_input_tensors.append(model_image_transform(image))

        if (i + 1) % 50 == 0 or (i + 1) == num_images:
            print(f"  processed {i + 1}/{num_images} images")

    # Encode every image in one batched forward pass: (N, 3, 224, 224) -> (N, 256).
    image_batch = torch.stack(model_input_tensors, dim=0).to(device)

    with torch.no_grad():
        image_features = model.image_encoder(image_batch)
        # L2-normalize now, once, at index-build time -- so every future
        # search just does a plain dot product against these vectors to
        # get cosine similarity, without needing to renormalize anything.
        image_features = F.normalize(image_features, dim=-1)

    # Move back to CPU before saving: the index file should be loadable
    # (e.g. by the Gradio app) on a machine without a GPU.
    image_features = image_features.cpu()

    index = {
        "image_features": image_features,  # (N, 256)
        "captions": captions,  # list[str], length N
        "images": display_images,  # list[PIL.Image], length N
    }

    torch.save(index, args.output)

    output_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved index with {num_images} images to {args.output} ({output_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
