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


def save_checkpoint(model, optimizer, epoch, global_step, save_path, is_distributed):
    """
    Saves FULL training state (not just model weights) so training can
    resume exactly where it left off, e.g. across a chain of separate
    SLURM job submissions:
      - "model": the underlying model's weights.
      - "optimizer": AdamW's internal per-parameter state (running
        averages of the gradient and its square -- "momentum" and
        "variance"). Without this, every resumed job would restart Adam's
        momentum from zero, causing a burst of unstable, incorrectly
        scaled updates right after every resume.
      - "epoch" / "global_step": where training was, so the next run knows
        where to continue counting from instead of starting over at 0.
    """
    # When wrapped in DDP, `model` is a DistributedDataParallel object, not
    # the underlying CLIPModel itself -- the real model (and its
    # state_dict) lives at `model.module`. Unwrapping here keeps the
    # checkpoint format identical regardless of how many GPUs were used to
    # produce it (no "module." prefix on every key).
    model_state = model.module.state_dict() if is_distributed else model.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    # Atomic save: write to a temporary file first, then rename it into
    # place. torch.save() writing directly to `save_path` is NOT atomic --
    # if the process is killed mid-write (e.g. SLURM hits its walltime
    # limit, which is exactly when we most need to checkpoint), `save_path`
    # would be left as a truncated/corrupted file, and the next chained job
    # would fail to resume from it. os.replace() (a rename) is atomic on
    # POSIX filesystems: at every instant, `save_path` either shows the
    # previous complete checkpoint or the new complete checkpoint, never a
    # half-written one.
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"Saved checkpoint to {save_path} (epoch {epoch}, step {global_step})")


def load_checkpoint(model, optimizer, save_path, is_distributed, device, rank):
    """
    Restores training state from `save_path`, if it exists, for --resume.

    All ranks call this (not just rank 0): on the shared filesystem SLURM
    jobs normally use, every rank reads the identical checkpoint file, so
    every rank's model weights AND optimizer state start out identical and
    stay in sync. Relying only on DDP's internal parameter broadcast at
    construction time would restore model weights but NOT optimizer
    state, which DDP does not manage -- each rank must load that itself.

    Note: we only record the epoch a checkpoint was saved at, not the
    exact batch/step within that epoch, so resuming restarts from the
    beginning of that epoch rather than the exact mid-epoch batch. This is
    a deliberate simplicity trade-off -- fine at this scale, since redoing
    part of one epoch is cheap compared to a full training run.
    """
    if not os.path.exists(save_path):
        if is_master(rank):
            print("No checkpoint found, starting fresh")
        return 0, 0

    checkpoint = torch.load(save_path, map_location=device)

    target_model = model.module if is_distributed else model
    target_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    if is_master(rank):
        print(f"Resuming from {save_path} at epoch {start_epoch}, step {global_step}")

    return start_epoch, global_step


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

    # Resume logic: only attempt to load a checkpoint if --resume was
    # explicitly passed. This keeps a plain (non-resuming) invocation
    # behaving exactly as before -- it ignores any leftover checkpoint
    # file at save_path and always starts fresh from epoch 0, step 0.
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            model, optimizer, args.save_path, is_distributed, device, rank
        )
    else:
        start_epoch, global_step = 0, 0

    stop_training = False
    completed_all_epochs = False
    epoch = start_epoch

    # Loop bound is --max_epochs (not the old --epochs), since in a
    # self-chaining SLURM setup this single invocation is only ONE link in
    # a chain of jobs collectively training up to max_epochs -- the real
    # stopping condition for the whole run is "epoch reached max_epochs",
    # not "this particular job did N epochs".
    for epoch in range(start_epoch, args.max_epochs):
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

            global_step += 1

            # Only the master process logs, to avoid N duplicate log lines
            # per step when running on N GPUs.
            if is_master(rank) and global_step % 10 == 0:
                print(f"epoch {epoch} | step {global_step} | loss {loss.item():.4f}")

            if is_master(rank) and global_step % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, epoch, global_step, args.save_path, is_distributed
                )

            if args.steps is not None and global_step >= args.steps:
                # Allow capping total steps (e.g. for a quick smoke test)
                # regardless of how many epochs/batches remain. This is a
                # deliberate early stop, not "training complete" -- it does
                # NOT trigger the TRAINING_COMPLETE marker below.
                stop_training = True
                break

        if stop_training:
            break
    else:
        # This `else` belongs to the `for epoch in ...` loop above, and
        # only runs if that loop finished normally -- i.e. it iterated
        # through every epoch up to max_epochs without ever hitting the
        # `break` from the --steps cap. That means training is genuinely,
        # fully complete, as opposed to merely paused/checkpointed for the
        # next link in the SLURM chain.
        completed_all_epochs = True

    if is_distributed:
        # Make sure every GPU has finished its last training step before
        # the master process reads out final weights for the checkpoint --
        # otherwise the master might save while other ranks are still
        # mid-step.
        dist.barrier()

    if is_master(rank):
        # If we finished all epochs, record the checkpoint's epoch as
        # max_epochs (rather than the last loop value, max_epochs - 1) so
        # that if this checkpoint is ever resumed, range(start_epoch,
        # max_epochs) is empty and immediately falls through to the
        # "completed" state again, instead of quietly re-running the final
        # epoch.
        final_epoch = args.max_epochs if completed_all_epochs else epoch
        save_checkpoint(
            model, optimizer, final_epoch, global_step, args.save_path, is_distributed
        )

        if completed_all_epochs:
            # Drop an empty marker file next to the checkpoint so the
            # SLURM chaining script can check for its existence and stop
            # resubmitting further jobs once training has genuinely
            # finished.
            save_dir = os.path.dirname(os.path.abspath(args.save_path))
            marker_path = os.path.join(save_dir, "TRAINING_COMPLETE")
            open(marker_path, "a").close()
            print(f"Reached max_epochs ({args.max_epochs}); wrote {marker_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP-style model (single-GPU or DDP)")
    parser.add_argument("--dataset", choices=["cifar", "flickr"], default="flickr")
    parser.add_argument("--max_epochs", type=int, default=30, help="Total epochs to train to, across all chained runs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=None, help="Cap total steps this run (for quick test runs)")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="clip_flickr.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from --save_path if it exists")
    args = parser.parse_args()

    train(args)
    cleanup()
