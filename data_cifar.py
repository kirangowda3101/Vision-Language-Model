import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# GPT-2's byte-pair encoding tokenizer -- the same tokenizer used by the
# speech LM. We create one shared instance at module load time since
# building/loading it has some overhead and it's stateless/reusable across
# every dataset item.
_tokenizer = tiktoken.get_encoding("gpt2")

# CIFAR-10's 10 classes, in the exact index order the dataset uses (i.e.
# CIFAR10's integer label `i` corresponds to CIFAR10_CLASSES[i]).
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Turn each class name into a natural-language caption. CLIP-style models
# are trained on (image, caption) pairs, so we synthesize a caption per
# class using a fixed template -- this is the standard "prompt template"
# trick also used for CLIP zero-shot classification.
CIFAR10_CAPTIONS = [f"a photo of a {class_name}" for class_name in CIFAR10_CLASSES]

# Standard image preprocessing pipeline:
#   1. Resize every image up to 224x224 (CIFAR-10 images are natively only
#      32x32, but our VisionTransformer was built for 224x224 inputs with
#      16x16 patches -> 196 patches).
#   2. ToTensor converts a PIL image (H, W, C) with pixel values in
#      [0, 255] into a float tensor (C, H, W) with values in [0, 1].
#   3. Normalize rescales each channel from [0, 1] to roughly [-1, 1] using
#      mean=0.5, std=0.5 per channel: (x - 0.5) / 0.5.
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def _tokenize_caption(caption, max_seq_len):
    """
    Encodes a caption string into a fixed-length tensor of GPT-2 token IDs.

    Since sequences in a batch must all be the same length (to stack into
    one tensor), we pad short sequences with 0 up to max_seq_len, and
    truncate anything longer than max_seq_len.
    """
    # Encode the raw string into a variable-length list of GPT-2 token IDs.
    token_ids = _tokenizer.encode(caption)

    # Truncate: keep only the first max_seq_len tokens if it's too long.
    token_ids = token_ids[:max_seq_len]

    # Pad: if the sequence is shorter than max_seq_len, right-pad with 0s
    # until it reaches max_seq_len. (Token ID 0 is used here purely as a
    # placeholder pad value, not a special "[PAD]" token from GPT-2's
    # vocabulary -- these captions are short and fixed, so the model will
    # just learn to treat trailing 0s as padding.)
    num_pad_tokens = max_seq_len - len(token_ids)
    token_ids = token_ids + [0] * num_pad_tokens

    # Return as a torch.long tensor, the dtype nn.Embedding expects for
    # indices.
    return torch.tensor(token_ids, dtype=torch.long)


class CIFARClipDataset(Dataset):
    """
    Wraps torchvision's CIFAR10 dataset to produce (image, token_ids)
    pairs suitable for CLIP-style contrastive training: each image is
    paired with the tokenized caption for its class.
    """

    def __init__(self, root="./data", train=True, max_seq_len=32, download=True):
        super().__init__()

        self.max_seq_len = max_seq_len

        # The underlying CIFAR10 dataset. It already applies image_transform
        # to each image and returns (image_tensor, class_index) pairs.
        self.cifar = CIFAR10(
            root=root, train=train, download=download, transform=image_transform
        )

        # Pre-tokenize all 10 class captions once, up front, instead of
        # re-tokenizing the same caption string every time __getitem__ is
        # called for an image of that class -- tokenization is deterministic
        # per class, so there's no reason to repeat the work per-sample.
        self.class_token_ids = [
            _tokenize_caption(caption, max_seq_len) for caption in CIFAR10_CAPTIONS
        ]

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, index):
        # image_tensor: (3, 224, 224) after image_transform.
        # class_index: integer in [0, 9] identifying the image's class.
        image_tensor, class_index = self.cifar[index]

        # Look up that class's pre-tokenized caption: (max_seq_len,)
        token_ids = self.class_token_ids[class_index]

        return image_tensor, token_ids


def get_cifar_dataloader(batch_size, max_seq_len=32, train=True, root="./data"):
    """
    Builds a DataLoader that yields (image_batch, token_ids_batch) pairs:
      - image_batch: (B, 3, 224, 224)
      - token_ids_batch: (B, max_seq_len)
    Downloads CIFAR-10 into `root` automatically if it isn't already there.
    """
    dataset = CIFARClipDataset(
        root=root, train=train, max_seq_len=max_seq_len, download=True
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_class_captions(max_seq_len=32):
    """
    Returns the 10 CIFAR-10 class captions, tokenized, stacked into a
    single (10, max_seq_len) tensor. Useful for zero-shot classification:
    encode this once with the text encoder to get one embedding per class,
    then compare an image embedding against all 10 to predict its class.
    """
    token_ids_list = [
        _tokenize_caption(caption, max_seq_len) for caption in CIFAR10_CAPTIONS
    ]
    return torch.stack(token_ids_list, dim=0)  # (10, max_seq_len)


if __name__ == "__main__":
    dataloader = get_cifar_dataloader(batch_size=4)

    # Grab a single batch to sanity-check shapes.
    image_batch, token_ids_batch = next(iter(dataloader))

    print(image_batch.shape)  # expected: torch.Size([4, 3, 224, 224])
    print(token_ids_batch.shape)  # expected: torch.Size([4, 32])

    # Decode the first caption in the batch back to text to confirm
    # tokenization (and padding) round-trips sensibly. decode() on the
    # trailing 0-padding will just produce whatever string token ID 0 maps
    # to, repeated -- included here so it's visible what padding looks
    # like when decoded.
    decoded_caption = _tokenizer.decode(token_ids_batch[0].tolist())
    print(decoded_caption)
