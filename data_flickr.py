import os

# HuggingFace's `datasets` library defaults to caching downloaded
# datasets/images under the user's home directory (~/.cache/huggingface).
# Flickr30k's images add up to several GB, which can blow through a home
# directory's disk quota on shared/cluster machines. Setting these env
# vars BEFORE importing `datasets` redirects all of its caching to scratch
# space instead. This must happen before `import datasets`/`load_dataset`,
# since the library reads these env vars once at import/init time.
_HF_CACHE_DIR = "/scratch/ramanagarajayaram.k/hf_cache"
os.makedirs(_HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = _HF_CACHE_DIR

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# Shared GPT-2 byte-pair encoding tokenizer, same as data_cifar.py -- built
# once at module load time since it's stateless and reused for every item.
_tokenizer = tiktoken.get_encoding("gpt2")

# Same preprocessing pipeline as the CIFAR-10 dataloader, so both datasets
# feed the VisionTransformer identically-shaped, identically-normalized
# inputs:
#   1. Resize to the 224x224 input size the VisionTransformer expects.
#   2. ToTensor: PIL image (H, W, C) in [0, 255] -> float tensor (C, H, W)
#      in [0, 1].
#   3. Normalize to roughly [-1, 1] per channel.
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def _tokenize_caption(caption, max_seq_len):
    """
    Encodes a caption string into a fixed-length tensor of GPT-2 token IDs,
    truncating or right-padding (with 0) to exactly max_seq_len. Same
    logic as data_cifar.py's _tokenize_caption, duplicated here so this
    module can be used standalone.
    """
    token_ids = _tokenizer.encode(caption)
    token_ids = token_ids[:max_seq_len]
    num_pad_tokens = max_seq_len - len(token_ids)
    token_ids = token_ids + [0] * num_pad_tokens
    return torch.tensor(token_ids, dtype=torch.long)


class Flickr30kClipDataset(Dataset):
    """
    Wraps a HuggingFace Flickr30k dataset split to produce (image,
    token_ids) pairs suitable for CLIP-style contrastive training.

    Unlike CIFAR-10 (one caption template per class), each Flickr30k image
    comes with a LIST of ~5 different human-written captions describing
    it. For simplicity/determinism we always use the first caption in the
    list here; using a different caption (or a randomly chosen one) per
    epoch is a valid data-augmentation alternative, since it exposes the
    model to more caption phrasings for the same image.
    """

    def __init__(self, hf_dataset, max_seq_len=32):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        # The "image" field decodes to a PIL image. Flickr30k images are
        # nominally RGB, but a handful of source JPEGs are grayscale (mode
        # "L") or CMYK -- feeding those directly into our transform would
        # produce a tensor with the wrong number of channels (e.g. 1
        # instead of 3) and break batching/the model's Conv2d, which
        # expects exactly in_channels=3. .convert("RGB") normalizes every
        # image to 3 channels up front, regardless of its original mode.
        image = example["image"].convert("RGB")
        image_tensor = image_transform(image)  # (3, 224, 224)

        # "original_alt_text" is a list of the 5 original human-written
        # Flickr30k captions for this image. We deterministically take the
        # first one so that __getitem__(idx) always returns the same
        # tokens for the same idx (useful for reproducibility/debugging).
        # A valid alternative would be to randomly sample among the 5
        # (e.g. `caption_list[random.randrange(len(caption_list))]`) each
        # time an image is drawn, which acts as text-side data
        # augmentation by exposing the model to more phrasings of the same
        # image.
        #
        # A small number of rows may have this field missing/empty (e.g.
        # due to upstream scraping/annotation gaps), so we fall back to
        # the single "alt_text" field in that case.
        caption_list = example.get("original_alt_text")
        if caption_list:
            caption = caption_list[0]
        else:
            caption = example["alt_text"]

        token_ids = _tokenize_caption(caption, self.max_seq_len)

        return image_tensor, token_ids


def get_flickr_dataloader(
    batch_size,
    max_seq_len=32,
    split="train",
    train_frac=0.9,
    root_cache=_HF_CACHE_DIR,
):
    """
    Builds a DataLoader over Flickr30k, yielding (image_batch, token_ids_batch)
    pairs:
      - image_batch: (B, 3, 224, 224)
      - token_ids_batch: (B, max_seq_len)

    Flickr30k (as distributed on the HuggingFace Hub under
    "Mozilla/flickr30k-transformed-captions" -- stored as parquet, no
    dataset loading script required, so it works with the current
    `datasets` library) only ships a single ~31,000-row split,
    confusingly named "test". Since we still want a train/val distinction
    for our own training loop, we manually carve that single split into
    two contiguous index ranges: the first `train_frac` fraction of rows
    for "train", and the remaining rows for "val".
    """
    # `root_cache` is accepted for interface symmetry with get_cifar_dataloader
    # (which takes a `root` dir) and to let callers point at a different
    # cache location if needed; the module-level env vars set above already
    # redirect the default HF cache, so this is only used if explicitly
    # overridden.
    full_dataset = load_dataset(
        "Mozilla/flickr30k-transformed-captions", split="test", cache_dir=root_cache
    )

    num_rows = len(full_dataset)
    split_index = int(num_rows * train_frac)

    if split == "train":
        # First train_frac fraction of rows.
        hf_split = full_dataset.select(range(0, split_index))
    else:
        # Remaining rows.
        hf_split = full_dataset.select(range(split_index, num_rows))

    dataset = Flickr30kClipDataset(hf_split, max_seq_len=max_seq_len)

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


if __name__ == "__main__":
    dataloader = get_flickr_dataloader(batch_size=4, split="train")

    # Grab a single batch to sanity-check shapes.
    image_batch, token_ids_batch = next(iter(dataloader))

    print(image_batch.shape)  # expected: torch.Size([4, 3, 224, 224])
    print(token_ids_batch.shape)  # expected: torch.Size([4, 32])

    # Decode the first caption in the batch to confirm we're reading real
    # Flickr30k captions (not just class-name templates like CIFAR-10).
    decoded_caption = _tokenizer.decode(token_ids_batch[0].tolist())
    print(decoded_caption)

