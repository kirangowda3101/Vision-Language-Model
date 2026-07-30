import os

# Same reasoning as data_flickr.py / search_demo.py: redirect HuggingFace's
# cache to scratch space before importing `datasets`, so it doesn't fill up
# the home directory quota on the cluster. This is a no-op (just points the
# cache somewhere writable) anywhere else, e.g. running this app locally.
_HF_CACHE_DIR = "/scratch/ramanagarajayaram.k/hf_cache"
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = _HF_CACHE_DIR

import gradio as gr
import tiktoken
import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel

MAX_SEQ_LEN = 32
NUM_INDEX_IMAGES = 300
TOP_K_SEARCH = 5

# Fixed candidate labels for the zero-shot classification tab. Unlike
# CIFAR-10 (10 fixed dataset classes), this is just a reasonable general
# vocabulary of "things a photo might be of", to demonstrate that CLIP can
# classify against ANY set of text labels you hand it -- not just labels
# it was trained with.
ZERO_SHOT_LABELS = [
    "a dog",
    "a cat",
    "a bird",
    "a horse",
    "a car",
    "an airplane",
    "a ship",
    "a person",
    "a building",
    "food",
]

_tokenizer = tiktoken.get_encoding("gpt2")

# Same preprocessing pipeline used throughout the project (data_cifar.py,
# data_flickr.py, search_demo.py): resize to the ViT's expected input size,
# convert to a float tensor in [0, 1], then normalize to roughly [-1, 1]
# per channel.
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def tokenize_text(text, max_seq_len=MAX_SEQ_LEN):
    """Encodes a string into a fixed-length (1, max_seq_len) GPT-2 token ID tensor."""
    token_ids = _tokenizer.encode(text)
    token_ids = token_ids[:max_seq_len]
    token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long)  # (1, max_seq_len)


def get_caption(example):
    """Best-effort real caption for a Flickr30k row, for display alongside search results."""
    caption_list = example.get("original_alt_text")
    if caption_list:
        return caption_list[0]
    return example.get("alt_text", "<no caption available>")


def build_model(checkpoint_path="best.pt"):
    """
    Reconstructs the exact same CLIPModel architecture used in train.py,
    loads the trained weights from `checkpoint_path`, and puts the model
    in eval mode on the fastest available device. Identical in spirit to
    zero_shot.py's / search_demo.py's build_model().
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


def build_image_index(model, device, num_images=NUM_INDEX_IMAGES):
    """
    Builds the text-to-image search index ONCE at startup: encodes
    `num_images` Flickr30k images into the shared CLIP embedding space,
    L2-normalizes each vector, and stacks them into a single (N, 256)
    matrix. Every subsequent search in the Gradio app just encodes ONE
    short text query and compares it against this already-computed
    matrix, instead of re-running the (expensive) image encoder on every
    search.
    """
    print(f"Building search index from {num_images} Flickr30k images...")
    dataset = load_dataset(
        "Mozilla/flickr30k-transformed-captions", split="test", cache_dir=_HF_CACHE_DIR
    )
    num_images = min(num_images, len(dataset))
    dataset = dataset.select(range(num_images))

    # Keep the raw PIL images and captions in lists parallel to the
    # embedding matrix's rows, so index i in `image_features` always
    # corresponds to `images[i]` / `captions[i]`.
    images = []
    captions = []
    image_tensors = []

    for i in range(num_images):
        example = dataset[i]
        image = example["image"].convert("RGB")
        images.append(image)
        captions.append(get_caption(example))
        image_tensors.append(image_transform(image))

    image_batch = torch.stack(image_tensors, dim=0).to(device)

    with torch.no_grad():
        image_features = model.image_encoder(image_batch)  # (N, 256)
        # L2-normalize so that a dot product against a normalized text
        # vector gives cosine similarity -- pure directional alignment
        # between the image and text embeddings, unaffected by either
        # vector's raw magnitude.
        image_features = F.normalize(image_features, dim=-1)

    print("Search index built.")
    return image_features, images, captions


def build_label_index(model, device, labels=ZERO_SHOT_LABELS):
    """
    Encodes the fixed zero-shot candidate labels ONCE at startup, for the
    same reason as build_image_index: the label set never changes between
    classification requests, so there's no reason to re-run the text
    encoder on it every time a user uploads a new image.
    """
    token_ids = torch.cat([tokenize_text(label) for label in labels], dim=0).to(device)  # (num_labels, MAX_SEQ_LEN)

    with torch.no_grad():
        label_features = model.text_encoder(token_ids)  # (num_labels, 256)
        label_features = F.normalize(label_features, dim=-1)

    return label_features


# --- One-time startup: build the model and both indexes ------------------
model, device = build_model()
print(f"Using device: {device}")

image_features_index, indexed_images, indexed_captions = build_image_index(model, device)
label_features_index = build_label_index(model, device)


def search_images(query_text):
    """
    Text-to-image search: embeds `query_text` with the text encoder,
    L2-normalizes it, and compares it against every pre-computed,
    L2-normalized image embedding via a single dot product -- since both
    vectors are unit-length, this dot product IS their cosine similarity.
    Returns the top-K images as (image, caption_with_score) pairs for a
    gr.Gallery.
    """
    if not query_text or not query_text.strip():
        return []

    token_ids = tokenize_text(query_text).to(device)

    with torch.no_grad():
        query_features = model.text_encoder(token_ids)  # (1, 256)
        query_features = F.normalize(query_features, dim=-1)

        # (1, 256) @ (256, N) -> (1, N) -> (N,): similarity of this one
        # query against every indexed image.
        similarity = (query_features @ image_features_index.T).squeeze(0)

    top_scores, top_indices = similarity.topk(min(TOP_K_SEARCH, similarity.shape[0]))

    gallery_items = []
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        label = f"{indexed_captions[idx]}  (score: {score:.3f})"
        gallery_items.append((indexed_images[idx], label))

    return gallery_items


def classify_image(uploaded_image):
    """
    Zero-shot image classification: embeds the uploaded image, compares it
    against the fixed set of pre-encoded ZERO_SHOT_LABELS via cosine
    similarity, and converts those similarity scores into a probability
    distribution via softmax so gr.Label can render confidence bars.
    """
    if uploaded_image is None:
        return {}

    # gr.Image (type="pil") hands us a PIL image directly.
    image_tensor = image_transform(uploaded_image.convert("RGB")).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        image_feature = model.image_encoder(image_tensor)  # (1, 256)
        image_feature = F.normalize(image_feature, dim=-1)

        # (1, 256) @ (256, num_labels) -> (1, num_labels) -> (num_labels,)
        similarity = (image_feature @ label_features_index.T).squeeze(0)

        # Softmax turns raw cosine-similarity scores (in [-1, 1], not
        # probabilities) into a proper probability distribution over the
        # candidate labels that sums to 1, which is what gr.Label expects
        # in order to draw its confidence bars.
        probabilities = similarity.softmax(dim=-1)

    return {
        label: probability.item()
        for label, probability in zip(ZERO_SHOT_LABELS, probabilities)
    }


# --- Gradio UI -------------------------------------------------------------
with gr.Blocks(title="CLIP Demo") as demo:
    gr.Markdown("# CLIP-from-scratch demo\nTrained on Flickr30k image-caption pairs.")

    with gr.Tab("Text-to-image search"):
        gr.Markdown("Enter a text query and find the closest matching images from a 300-image Flickr30k index.")
        query_input = gr.Textbox(label="Search query", placeholder="e.g. a dog running on the beach")
        search_button = gr.Button("Search")
        results_gallery = gr.Gallery(label="Top matches", columns=5)

        search_button.click(fn=search_images, inputs=query_input, outputs=results_gallery)
        query_input.submit(fn=search_images, inputs=query_input, outputs=results_gallery)

    with gr.Tab("Zero-shot classification"):
        gr.Markdown("Upload an image and see how it scores against a fixed set of candidate labels.")
        image_input = gr.Image(label="Upload an image", type="pil")
        classify_button = gr.Button("Classify")
        label_output = gr.Label(label="Predictions", num_top_classes=5)

        classify_button.click(fn=classify_image, inputs=image_input, outputs=label_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
