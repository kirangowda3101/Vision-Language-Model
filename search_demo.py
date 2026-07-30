import argparse
import os

# Same reasoning as data_flickr.py: redirect HuggingFace's cache to scratch
# space before importing `datasets`, so downloading/caching Flickr30k
# images doesn't blow through the home directory quota. This must happen
# before `import datasets`/`load_dataset`.
_HF_CACHE_DIR = "/scratch/ramanagarajayaram.k/hf_cache"
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = _HF_CACHE_DIR

import tiktoken
import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel

# Shared GPT-2 tokenizer for encoding search queries.
_tokenizer = tiktoken.get_encoding("gpt2")

MAX_SEQ_LEN = 32

# Same preprocessing pipeline used throughout the project (data_cifar.py,
# data_flickr.py): resize to the ViT's expected input size, convert to a
# float tensor in [0, 1], then normalize to roughly [-1, 1] per channel.
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def build_model(checkpoint_path):
    """
    Reconstructs the exact same CLIPModel architecture used in train.py,
    loads the trained weights, and puts the model in eval mode on the
    fastest available device. Identical to zero_shot.py's build_model().
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


def _tokenize_query(text):
    """Encodes a search query string into a fixed-length (1, MAX_SEQ_LEN) tensor."""
    token_ids = _tokenizer.encode(text)
    token_ids = token_ids[:MAX_SEQ_LEN]
    token_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long)  # (1, MAX_SEQ_LEN)


def _get_caption(example):
    """Best-effort real caption for a Flickr30k row, for sanity-checking matches."""
    caption_list = example.get("original_alt_text")
    if caption_list:
        return caption_list[0]
    return example.get("alt_text", "<no caption available>")


def build_image_index(model, device, num_images):
    """
    Builds a searchable image index ONCE, up front: encodes every candidate
    image into the shared CLIP embedding space, L2-normalizes each vector,
    and stacks them into a single (N, 256) matrix.

    We do this only once (not per-query) because the images themselves
    never change between searches -- only the text query changes. Encoding
    every image up front means each subsequent search() call just needs to
    encode ONE short text query and compare it against this
    already-computed matrix, instead of re-running the (expensive) image
    encoder for every image on every single query.
    """
    print(f"Loading {num_images} Flickr30k images for the search index...")
    dataset = load_dataset(
        "Mozilla/flickr30k-transformed-captions", split="test", cache_dir=_HF_CACHE_DIR
    )
    num_images = min(num_images, len(dataset))
    dataset = dataset.select(range(num_images))

    # Keep the raw PIL images (and their captions) around in parallel with
    # their embeddings, indexed identically, so a search result's index
    # into `image_features` also tells us which raw image/caption it is.
    images = []
    captions = []
    image_tensors = []

    for i in range(num_images):
        example = dataset[i]
        image = example["image"].convert("RGB")
        images.append(image)
        captions.append(_get_caption(example))
        image_tensors.append(image_transform(image))

        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"  indexed {i + 1}/{num_images} images")

    # Stack every image into one big batch and encode all at once:
    # (N, 3, 224, 224) -> (N, 256).
    image_batch = torch.stack(image_tensors, dim=0).to(device)

    with torch.no_grad():
        image_features = model.image_encoder(image_batch)
        # L2-normalize so a dot product against a normalized text vector
        # gives cosine similarity (pure directional alignment between the
        # image and text embeddings), not something skewed by whichever
        # vector happens to have larger raw magnitude.
        image_features = F.normalize(image_features, dim=-1)

    print("Index built.")
    return image_features, images, captions


def search(query_text, model, device, image_features, top_k):
    """
    Text-to-image search: embeds `query_text` with the text encoder,
    L2-normalizes it, and compares it against every pre-computed,
    L2-normalized image embedding via a single dot product -- since both
    vectors are unit-length, this dot product IS their cosine similarity,
    a score in [-1, 1] measuring how aligned the query's and each image's
    "direction" in embedding space are (not influenced by either vector's
    magnitude). Returns the top_k highest-scoring (image_index, score)
    pairs.
    """
    token_ids = _tokenize_query(query_text).to(device)  # (1, MAX_SEQ_LEN)

    with torch.no_grad():
        query_features = model.text_encoder(token_ids)  # (1, 256)
        query_features = F.normalize(query_features, dim=-1)

        # (1, 256) @ (256, N) -> (1, N) -> (N,): similarity of this one
        # query against every indexed image.
        similarity = (query_features @ image_features.T).squeeze(0)

    # Get the top_k highest-similarity images and their scores.
    top_scores, top_indices = similarity.topk(top_k)

    return list(zip(top_indices.tolist(), top_scores.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-image search demo with a trained CLIP model")
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser("~/checkpoints/clip_flickr/best.pt"),
    )
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model, device = build_model(args.checkpoint)
    print(f"Using device: {device}")

    # Build the image index ONCE before entering the interactive loop --
    # every query below reuses this same `image_features` matrix.
    image_features, images, captions = build_image_index(model, device, args.num_images)

    print("\nCLIP text-to-image search demo. Type a query, or 'quit' to exit.")
    while True:
        query_text = input("\nEnter a search query (or 'quit'): ").strip()
        if query_text.lower() == "quit":
            break
        if not query_text:
            continue

        results = search(query_text, model, device, image_features, args.top_k)

        for rank, (image_index, score) in enumerate(results, start=1):
            # No GUI yet -- just print the index/score/caption for now; a
            # web UI can later render `images[image_index]` directly.
            print(
                f"Rank {rank}: image #{image_index}  score {score:.4f}  "
                f"caption: {captions[image_index]}"
            )
