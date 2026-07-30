# Vision Language Model

A CLIP-style vision-language model built from scratch: a Vision Transformer image encoder and a text encoder, trained together with a contrastive loss on Flickr30k image-caption pairs.

No pretrained CLIP weights. No fine-tuning on top of an existing model. The patch embedding, the attention, the transformer blocks, the contrastive loss, the training loop — all of it was written from scratch.

I'd previously built a [speech-language model](https://github.com/kirangowda3101/Speech-Language-Model) from scratch, which meant implementing transformers (RoPE, SwiGLU, attention, the training loop) for audio and text. This project is the vision counterpart to that one — the same from-scratch approach, applied to images and text instead of speech and text. Together they're meant to show the same underlying skill (building transformers from the ground up) across two different modalities.

**Code:** [GitHub](https://github.com/kirangowda3101/Vision-Language-Model)

---

## The core idea

An image and a sentence describing it don't look anything alike as raw data — one is a grid of pixels, the other is a sequence of words. CLIP's trick is to train two separate encoders, one for each, and force both of them to land in the same vector space. A picture of a dog and the caption "a photo of a dog" should end up as two vectors that point in almost the same direction.

Training does this with a contrastive loss: for every batch of (image, caption) pairs, pull the correct pairs' embeddings together and push every other combination in the batch apart. Do that enough times and the model learns a shared space where "similar meaning" means "close together," regardless of which side (image or text) a vector came from.

Once you have that space, two things fall out for free. Text-to-image search: embed a query like "a dog on a beach," compare it against a set of pre-embedded images, and return whichever ones are closest. Zero-shot classification: embed a handful of candidate labels ("a cat," "a car," "a building"), embed an image, and see which label it's closest to — without ever training on those specific labels.

---

## Architecture

**Image side.** The image is cut into non-overlapping 16x16 patches and each patch is linearly projected into the embedding space — done in one shot with a single `Conv2d(kernel_size=16, stride=16)`, which is mathematically the same operation as flattening each patch and multiplying by a weight matrix. A learnable CLS token is prepended to the sequence of patch embeddings, and learnable position embeddings are added so the model knows where each patch sits in the image. The sequence passes through a stack of pre-norm transformer blocks with multi-head self-attention written from scratch (no `nn.MultiheadAttention`). After the last block, the CLS token's output is taken as the image's summary and projected into the shared 256-dim space.

**Text side.** A caption is tokenized with GPT-2's byte-pair tokenizer, embedded, given the same kind of learnable position embeddings, and run through the identical transformer block architecture used on the image side — attention doesn't care whether the tokens it's mixing came from patches or words. The last token's output is taken as the caption's summary and projected into the same 256-dim space.

| Setting | Value |
| --- | --- |
| Image size | 224 x 224 |
| Patch size | 16 x 16 |
| Embedding dim | 512 |
| Attention heads | 8 |
| Transformer depth | 6 blocks (each side) |
| Output (shared) dim | 256 |
| Text vocabulary | 50,257 (GPT-2 BPE) |
| Max caption length | 32 tokens |

---

## Training

Trained on Flickr30k — about 31,000 images, each with 5 human-written captions — loaded through HuggingFace `datasets`. One caption per image is used per epoch; the original loader (`nlphuji/flickr30k`) stopped working with newer versions of the `datasets` library, so training switched to a parquet-based mirror (`Mozilla/flickr30k-transformed-captions`) that doesn't need a custom dataset script.

**Optimizer:** AdamW, learning rate 1e-4, weight decay 0.1. Standard image/text contrastive setup: each GPU (or the single process, in the non-distributed case) computes the loss over its own batch, treating the diagonal of the image-caption similarity grid as the correct matches and everything off-diagonal as negatives.

**Infrastructure:** a single V100 on Northeastern's Explorer HPC cluster, managed with SLURM. Multi-GPU training via DDP was implemented and validated on 2 GPUs, though the actual training runs used a single GPU. A few things were necessary to make this workable on a cluster with an 8-hour job limit:

- **Self-chaining.** A SLURM job pre-submits its own successor (as a dependency) before it starts training, then cancels that pending job afterward if training turned out to be fully complete. This lets a run continue unattended across as many jobs as it needs.
- **Checkpoint resume.** Checkpoints save the full training state — model weights, optimizer state, epoch, and global step — not just weights. Restoring only weights and starting AdamW's momentum from zero on every resume would produce a burst of bad updates after every job boundary; saving optimizer state avoids that.
- **Atomic checkpoint writes.** Checkpoints are written to a temp file and then renamed into place, so a job killed mid-write (which is the normal way these jobs end, at the time limit) can't leave a corrupted checkpoint behind.

---

## The batch size experiment

The first full training run used batch size 16, and the results were weak. Zero-shot accuracy on CIFAR-10 came out to 16.2% (random guessing is 10%, so the model had learned something, but not much). Running the text-to-image search demo made the problem visible directly: similarity scores across different images were all bunched together, with no real gap between good and bad matches, and the top results for a query like "a dog" were a mix of dogs and things that clearly weren't dogs.

The cause is how contrastive learning actually works. Each image in a batch is only ever contrasted against the *other captions in that same batch* — those are its negatives. Batch size 16 means 15 negatives per image. That's not enough variety for the loss to force the model to draw sharp distinctions; with so few things to tell apart, "close enough" gets rewarded too easily.

Retraining with batch size 64 (63 negatives per image, same everything else) changed this substantially:

| | Batch 16 | Batch 64 |
| --- | --- | --- |
| Negatives per image | 15 | 63 |
| Zero-shot CIFAR-10 accuracy | 16.2% | 23.6% |
| "a dog" search results | mixed, several wrong | all 5 results were dogs |
| Similarity scores | bunched, no clear gap | confident matches ~0.66, uncertain ones ~0.49 |

The batch-64 model didn't just score higher — it started behaving the way a contrastive model is supposed to: confident when it's right, and visibly less confident when it's not, instead of returning everything with roughly the same score.

---

## Results

Text-to-image search (batch-64 model) is noticeably better on concepts that show up a lot in Flickr30k — dogs, people doing things outdoors, groups of people in a scene — and weaker on things that are rarer in that dataset, like bicycles. Two real queries side by side make this concrete: one it's good at, one it isn't.

Query: `"a dog"` — top 5 results, all of them actually dogs:

```
0.656  A little tan dog with large ears running through the grass.
0.638  Two brown dogs are creating large splashes as they run in a river.
0.637  A white dog is resting its head on a tiled floor with its eyes open.
0.564  A light brown dog runs down a path happily.
0.537  A yellow lab standing next to a man.
```

Query: `"a man on a bicycle"` — top result:

```
0.491  A young gymnast jumps high in the air while performing on a balance beam.
```

No bicycle, no man — a wrong match. But look at the score: 0.491, well below the 0.656–0.537 range for the dog query. That gap is the model telling you it doesn't have a good match, not confidently getting it wrong. A low, flat score when the model is unsure is the correct behavior for a contrastive model, not a bug.

Zero-shot on CIFAR-10 test images (never seen during training, and CIFAR-10 itself was never used for training — only for evaluation): **23.6%** against a 10% random baseline across the 10 classes. This is a real, if modest, out-of-domain transfer result: the model was trained entirely on Flickr30k photos and captions, and is being asked to classify a completely different, much lower-resolution dataset it has never encountered.

---

## What it can't do

This is a small-scale project by design, and the results should be read that way. Flickr30k is about 31,000 image-caption pairs; the original CLIP paper trained on roughly 400 million. The gap between 23.6% and something closer to CLIP's reported zero-shot numbers is a data and compute limitation, not evidence that the architecture or training loop is broken — the batch size experiment above is a pretty direct demonstration that the training mechanics work correctly and respond to the things that should improve them.

Concretely:

- It's strong at matching within the distribution it was trained on (photo-style Flickr images with natural-language captions) and weaker at generalizing to genuinely out-of-domain images.
- Padding tokens in captions are not masked out of attention. Every caption is padded to a fixed 32 tokens, and the model attends over the padding along with the real tokens. This is a deliberate simplification, not an oversight — it costs some efficiency and precision, but at this sequence length and scale it didn't stop the model from learning.
- Rare concepts in the training data are underrepresented in the model's vocabulary of "things it recognizes well," which shows up directly in weaker search results for them.

---

## Repository structure

```
model_vit.py          Vision Transformer image encoder, written from scratch
                       (patch embedding via strided conv, CLS token, learnable
                       positions, multi-head self-attention, pre-norm blocks)
text_encoder.py        text encoder reusing the same transformer block as the
                       image side, GPT-2 tokenization, last-token summary
clip_model.py          ties both encoders together, L2-normalizes, computes
                       the symmetric contrastive loss
data_cifar.py          CIFAR-10 dataloader, used for zero-shot evaluation
data_flickr.py         Flickr30k dataloader (parquet mirror), caption
                       tokenization and padding
train.py               training loop: DDP support, checkpoint resume,
                       atomic checkpoint saves, self-chaining support
train.slurm            SLURM job script that self-chains across the
                       cluster's time limit
zero_shot.py           zero-shot CIFAR-10 classification using a trained
                       checkpoint
search_demo.py         command-line text-to-image search demo
precompute_index.py    builds and saves an image search index once, offline
app.py                 Gradio demo for the cluster (builds the search index
                       live at startup)
app_space.py           CPU-only variant for HuggingFace Spaces, loads the
                       precomputed index instead of building it live
```

---

## Running it

Set up the environment. This was developed on Python 3.11 with PyTorch. On the GPU cluster, a plain `pip install torch` silently pulled a CPU-only or mismatched-CUDA build that didn't actually use the GPU — installing the CUDA 12.1 build explicitly fixed it:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken datasets gradio
```

The right `--index-url` depends on the machine's actual CUDA driver version, so check that before copying this verbatim on a different machine. The SLURM details below (partition names, paths, module versions) are specific to Northeastern's Explorer cluster and will need adjusting elsewhere.

Train on the cluster (self-chains automatically until it reaches the configured number of epochs):

```
sbatch train.slurm
```

Run zero-shot classification on CIFAR-10 with a trained checkpoint:

```
python zero_shot.py --checkpoint ~/checkpoints/clip_flickr_v2/best.pt --num_images 500
```

Run the text-to-image search demo:

```
python search_demo.py --checkpoint ~/checkpoints/clip_flickr_v2/best.pt --num_images 500 --top_k 5
```

---

Kiran Gowda [LinkedIn](https://www.linkedin.com/in/kirangowda3101/) | [GitHub](https://github.com/kirangowda3101)
