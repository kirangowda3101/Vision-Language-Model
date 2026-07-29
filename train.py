import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model_vit import VisionTransformer
from text_encoder import TextEncoder
from clip_model import CLIPModel, clip_contrastive_loss
from data_cifar import get_cifar_dataloader
from data_flickr import get_flickr_dataloader


def init_distributed():
    """
    Sets up (or skips) PyTorch's distributed process group, depending on
    whether this process was launched by `torchrun`.

    When you launch a script with:
        torchrun --nproc_per_node=<N> train.py ...
    torchrun spawns N processes (one per GPU) and sets the environment
    variables RANK, LOCAL_RANK, and WORLD_SIZE in each one before running
    the script:
      - RANK: this process's global rank across ALL GPUs/machines (0..world_size-1)
      - LOCAL_RANK: this process's rank on its OWN machine (0..num_gpus_on_this_machine-1)
      - WORLD_SIZE: total number of processes/GPUs participating

    If those env vars are present, we're running under torchrun -> set up
    real multi-GPU distributed training. If they're absent (plain
    `python3 train.py`), we're running as a single ordinary process -> skip
    all distributed setup entirely and behave like the original
    single-device script.
    """
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # "nccl" is NVIDIA's collective-communications backend, the
        # standard (and fastest) choice for GPU-to-GPU communication
        # (gradient all-reduce, etc.) -- it's the backend basically every
        # multi-GPU PyTorch training script uses.
        dist.init_process_group(backend="nccl")

        # Pin this process to its own GPU. Each of the N processes torchrun
        # launches must use a DIFFERENT GPU; local_rank is exactly the
        # index for that, since it's assigned 0..num_gpus_on_this_machine-1.
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size, True

    # Not launched via torchrun -> single-process, non-distributed mode.
    return 0, 0, 1, False


def is_master(rank):
    # By convention, rank 0 is the "master"/"main" process. In distributed
    # training, N processes are all doing (nearly) identical work in
    # lockstep, so anything that should only happen ONCE per step (logging,
    # checkpoint saving) should be gated on `is_master(rank)` -- otherwise
    # every GPU would print duplicate logs / write the same checkpoint file
    # N times, wasting I/O and racing each other.
    return rank == 0


def cleanup():
    # Tear down the process group cleanly at the end of the script. Safe to
    # call unconditionally: dist.is_initialized() is False in the
    # non-distributed (no torchrun) case, so this is a no-op there.
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, save_path, is_distributed):
    # When wrapped in DDP, `model` is a DistributedDataParallel object, not
    # the underlying CLIPModel itself -- the real model (and its
    # state_dict) lives at `model.module`. Saving `model.state_dict()`
    # directly would work too, but its keys would all be prefixed with
    # "module." which makes the checkpoint awkward to load back into a
    # plain (non-DDP) model later. Unwrapping to `model.module` here keeps
    # the checkpoint format identical regardless of how many GPUs were used
    # to produce it.
    state_dict = model.module.state_dict() if is_distributed else model.state_dict()
    torch.save(state_dict, save_path)
    print(f"Saved checkpoint to {save_path}")


def build_dataloader(dataset_name, batch_size, is_distributed, rank, world_size):
    # Reuse the existing single-process dataloader builders to construct
    # the underlying Dataset (get_cifar_dataloader / get_flickr_dataloader
    # already handle download/caching/tokenization); we then discard the
    # DataLoader they return and rebuild our own around the same
    # `.dataset`, so we can plug in a DistributedSampler when needed.
    if dataset_name == "cifar":
        base_loader = get_cifar_dataloader(batch_size=batch_size, max_seq_len=32, train=True)
    else:
        base_loader = get_flickr_dataloader(batch_size=batch_size, max_seq_len=32, split="train")
    dataset = base_loader.dataset

    if is_distributed:
        # DistributedSampler splits the dataset into `world_size`
        # non-overlapping shards (one per GPU) based on `rank`, so every
        # GPU sees a different slice of the data each epoch instead of all
        # GPUs redundantly processing the exact same batches.
        # shuffle=True here makes it reshuffle indices each epoch (using
        # the epoch number set via sampler.set_epoch() as the random seed)
        # -- important so shards aren't a frozen, identical split every
        # epoch.
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        # NOTE: when a sampler is provided, DataLoader forbids also passing
        # shuffle=True -- the sampler is now solely responsible for
        # ordering/subsetting the data.
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, sampler


def train(args):
    # Must be called before touching CUDA/creating the model, since it
    # determines which device this process should use.
    rank, local_rank, world_size, is_distributed = init_distributed()

    if is_distributed:
        # Each process gets exactly one GPU, identified by local_rank.
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_master(rank):
        print(
            f"Using device: {device} | distributed: {is_distributed} | world_size: {world_size}"
        )

    # Image encoder ("image tower"): turns a batch of images into
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

    # Text encoder ("text tower"): turns a batch of token ID sequences
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

    # Combine both towers into one CLIPModel and move ALL of its
    # parameters (both encoders + the logit_scale temperature) onto this
    # process's device.
    model = CLIPModel(image_encoder, text_encoder).to(device)

    if is_distributed:
        # DDP wraps the model so that, after each backward() call, it
        # automatically all-reduces (averages) gradients across every GPU
        # -- so even though each GPU only ever sees its own shard of data,
        # every GPU ends up with the exact same averaged gradient and
        # therefore stays in sync, producing identical model weights after
        # every optimizer.step().
        #
        # find_unused_parameters=True: DDP normally expects EVERY
        # parameter to receive a gradient on EVERY forward/backward pass,
        # and raises an error otherwise (it uses this to build a static
        # backward graph for efficiency). Our model doesn't quite satisfy
        # that: CLIPModel's `logit_scale` is used in every step here, but
        # in general "some parameters not touched every step" is a common
        # pattern (e.g. optional heads, layers used only in certain
        # branches) and this class of bug can silently appear as this
        # model architecture evolves. find_unused_parameters=True tells
        # DDP to tolerate that instead of erroring, at the cost of a bit
        # of extra bookkeeping overhead per step to detect which
        # parameters were actually used.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    dataloader, sampler = build_dataloader(
        args.dataset, args.batch_size, is_distributed, rank, world_size
    )

    # AdamW optimizer over every learnable parameter in the model (both
    # encoders' weights plus logit_scale). Calling model.parameters() on a
    # DDP-wrapped model transparently returns the underlying model's
    # parameters, so this line is identical in both distributed and
    # non-distributed modes.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    step = 0
    stop_training = False
    for epoch in range(args.epochs):
        if sampler is not None:
            # Must be called at the start of every epoch. DistributedSampler
            # uses the epoch number as part of its shuffling seed; without
            # this call, every epoch would use the SAME seed and therefore
            # produce the exact same shuffled order/shard split every time,
            # defeating the point of reshuffling between epochs.
            sampler.set_epoch(epoch)

        for images, token_ids in dataloader:
            images = images.to(device)
            token_ids = token_ids.to(device)

            # zero_grad() -> clear stale gradients from the previous step
            # (PyTorch accumulates gradients into .grad by default, so old
            # ones must be reset before computing new ones).
            optimizer.zero_grad()

            # Forward pass: encode this batch of images and captions and
            # compute the (B_local, B_local) similarity grid, where
            # B_local is the batch size on THIS GPU alone.
            logits = model(images, token_ids)

            # Contrastive loss. NOTE: with DDP, each GPU computes this loss
            # independently, over only its own local shard of the batch --
            # so the "negatives" each image is contrasted against are only
            # the other B_local - 1 items on the SAME GPU, not the full
            # effective batch across all GPUs. A more advanced setup would
            # all-gather every GPU's image/text features before computing
            # the similarity grid, so every image is contrasted against
            # negatives from the whole global batch (more negatives
            # generally means a harder, more informative contrastive
            # loss). We use the simpler per-shard loss here -- it's
            # standard practice and perfectly fine at this scale/dataset
            # size; all-gathering features is an optimization worth adding
            # later if training larger batches becomes important.
            loss = clip_contrastive_loss(logits)

            # backward() -> compute gradients via backpropagation. Under
            # DDP, this line ALSO triggers the cross-GPU gradient
            # all-reduce described above, once all parameters' gradients
            # for this step have been computed.
            loss.backward()

            # step() -> apply the (now-synced, identical-across-GPUs)
            # gradient update to this process's copy of the parameters.
            optimizer.step()

            # Only the master process logs, to avoid N duplicate log lines
            # per step when running on N GPUs.
            if is_master(rank) and step % 10 == 0:
                print(f"epoch {epoch} | step {step} | loss {loss.item():.4f}")

            if is_master(rank) and step > 0 and step % args.save_every == 0:
                save_checkpoint(model, args.save_path, is_distributed)

            step += 1
            if args.steps is not None and step >= args.steps:
                # Allow capping total steps (e.g. for a quick smoke test)
                # regardless of how many epochs/batches remain.
                stop_training = True
                break

        if stop_training:
            break

    if is_distributed:
        # Make sure every GPU has finished its last training step before
        # the master process reads out final weights for the checkpoint --
        # otherwise the master might save while other ranks are still
        # mid-step.
        dist.barrier()

    if is_master(rank):
        save_checkpoint(model, args.save_path, is_distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-style model (single-GPU or DDP)")
    parser.add_argument("--dataset", choices=["cifar", "flickr"], default="flickr")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=None, help="Cap total training steps (for quick test runs)")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="clip_flickr.pt")
    args = parser.parse_args()

    train(args)
    cleanup()
