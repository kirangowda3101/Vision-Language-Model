"""
Drift-detection experiment for the CLIP model on CIFAR-10.

1. Splits CIFAR-10 test indices 500-9999 into four disjoint pools
   (indices 0-499 are the README's zero-shot evaluation subset, used here
   only to confirm the checkpoint reproduces that accuracy):
     reference   -> the clean reference window
     calibration -> clean batches used to set alert thresholds
     heldout     -> separate clean batches used to measure false alarms
     shift       -> images that get corrupted
2. Records per-image input stats, embeddings and zero-shot probabilities.
3. Scores clean and corrupted batches against the reference, calibrates
   each signal's threshold to a 5% false-alarm rate on calibration
   batches, and measures false alarms on held-out clean batches and
   detection on corrupted batches.
4. Writes results/drift/drift_results.json, plots, and the reference
   artifact (drift_reference.npz) that drift_monitor.py loads at serving
   time.

Usage:
    python drift_experiment.py --checkpoint weights/best.pt
"""

import argparse
import json
import os
import platform
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import CIFAR10

import drift_metrics as dm
from corruptions import CORRUPTIONS, SEVERITIES, corrupt_batch
from data_cifar import CIFAR10_CLASSES, get_class_captions, image_transform
from zero_shot import build_model

EVAL_SUBSET_SIZE = 500  # zero_shot.py's default --num_images


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_images(model, device, images, batch_size=256):
    """(N, 32, 32, 3) uint8 -> (N, 256) L2-normalized image embeddings."""
    out = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = torch.stack([image_transform(Image.fromarray(img)) for img in chunk]).to(device)
        out.append(F.normalize(model.image_encoder(tensors), dim=-1).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def features(model, device, images, label_features, logit_scale):
    """Everything the drift signals need for a set of images."""
    embeddings = encode_images(model, device, images)
    return {
        "stats": dm.image_stats(images),
        "embeddings": embeddings,
        "probs": dm.zero_shot_probs(embeddings, label_features, logit_scale),
    }


def score_batches(reference, feats, batches, n_permutations, rng, labels=None):
    """Scores each batch; returns {signal: array}, mmd p-values, and per-batch accuracy."""
    scores = {s: [] for s in dm.SIGNALS}
    pvalues, accuracies = [], []
    for idx in batches:
        s, details = reference.score(
            feats["stats"][idx], feats["embeddings"][idx], feats["probs"][idx],
            n_permutations=n_permutations, rng=rng,
        )
        for name in dm.SIGNALS:
            scores[name].append(s[name])
        pvalues.append(details["mmd_pvalue"])
        if labels is not None:
            accuracies.append(float((feats["probs"][idx].argmax(1) == labels[idx]).mean()))
    return {k: np.array(v) for k, v in scores.items()}, np.array(pvalues), np.array(accuracies)


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean())


def plot_results(results, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Reference categorical palette, fixed slot order: one hue per corruption.
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    corruptions = list(CORRUPTIONS)
    severities = [0, *SEVERITIES]
    by_key = {(c["corruption"], c["severity"]): c for c in results["conditions"]}

    def series(corruption, getter):
        # Severity 0 is the uncorrupted shift pool, shared by all corruptions.
        return [getter(by_key[("none", 0)] if s == 0 else by_key[(corruption, s)]) for s in severities]

    def style(ax, title, ylabel):
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("severity (0 = clean)", fontsize=9, color="#52514e")
        ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
        ax.set_xticks(severities)
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")

    signal_titles = {
        "input_stats": "Input statistics (max |z|)",
        "embedding_cosine": "Embedding: cosine distance of means",
        "embedding_mmd": "Embedding: MMD² (RBF)",
        "confidence_top_prob": "Confidence: top-class prob (|z|)",
        "confidence_entropy": "Confidence: entropy (|z|)",
    }

    # 1. Drift score vs severity, one panel per signal. Scores live on very
    # different scales, so each is shown relative to its own calibrated
    # threshold: above 1.0 means the average batch would raise an alert.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#fcfcfb")
    for ax, signal in zip(axes.flat, dm.SIGNALS):
        threshold = results["thresholds"][signal]
        for corruption, color in zip(corruptions, colors):
            y = series(corruption, lambda c: c["signals"][signal]["mean_score"] / threshold)
            ax.plot(severities, y, color=color, linewidth=2, marker="o", markersize=5, label=corruption)
        ax.axhline(1.0, color="#52514e", linewidth=1, linestyle="--")
        # Unbiased MMD^2 can be negative on clean batches (its expectation
        # under no drift is 0), which a log axis would silently drop, so
        # that panel uses symlog; every other score is non-negative.
        if signal == "embedding_mmd":
            ax.set_yscale("symlog", linthresh=0.1)
            style(ax, signal_titles[signal], "mean score / threshold (symlog)")
        else:
            ax.set_yscale("log")
            style(ax, signal_titles[signal], "mean score / threshold (log)")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10,
                         title="corruption  (dashed = alert threshold)", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drift_score_vs_severity.png"), dpi=150)
    plt.close(fig)

    # 2. Detection rate vs severity, same layout.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#fcfcfb")
    for ax, signal in zip(axes.flat, dm.SIGNALS):
        for corruption, color in zip(corruptions, colors):
            y = series(corruption, lambda c: c["signals"][signal]["detection_rate"])
            ax.plot(severities, y, color=color, linewidth=2, marker="o", markersize=5, label=corruption)
        ax.axhline(results["config"]["target_false_alarm_rate"], color="#52514e", linewidth=1, linestyle="--")
        ax.set_ylim(-0.03, 1.03)
        style(ax, signal_titles[signal], "fraction of batches flagged")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10,
                         title="corruption  (dashed = 5% target false-alarm rate)", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "detection_rate_vs_severity.png"), dpi=150)
    plt.close(fig)

    # 3. Zero-shot accuracy vs severity.
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    for corruption, color in zip(corruptions, colors):
        y = series(corruption, lambda c: 100 * c["accuracy"])
        ax.plot(severities, y, color=color, linewidth=2, marker="o", markersize=5, label=corruption)
    ax.axhline(10, color="#52514e", linewidth=1, linestyle="--")
    ax.text(severities[-1], 10.3, "chance (10%)", ha="right", va="bottom", fontsize=8, color="#52514e")
    style(ax, f"Zero-shot CIFAR-10 accuracy (shift pool, n={results['config']['pool_sizes']['shift']})",
          "accuracy (%)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_severity.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Drift detection experiment for the CLIP model on CIFAR-10")
    parser.add_argument("--checkpoint", default="weights/best.pt")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--out_dir", default="results/drift")
    parser.add_argument("--reference_out", default="drift_reference.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference_size", type=int, default=2000)
    parser.add_argument("--pool_size", type=int, default=2500, help="size of calibration / heldout / shift pools")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_clean_batches", type=int, default=500, help="per calibration and heldout pool")
    parser.add_argument("--n_shift_batches", type=int, default=100, help="per corruption x severity")
    parser.add_argument("--n_permutations", type=int, default=500)
    parser.add_argument("--false_alarm_rate", type=float, default=0.05)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    model, _ = build_model(args.checkpoint)
    device = pick_device()
    model = model.to(device)
    logit_scale = float(model.logit_scale.exp().item())
    print(f"Using device: {device}, logit_scale = {logit_scale:.4f}")

    with torch.no_grad():
        label_features = F.normalize(model.text_encoder(get_class_captions(32).to(device)), dim=-1)
    label_features = label_features.float().cpu().numpy().astype(np.float64)

    test = CIFAR10(root=args.data_root, train=False, download=True)
    images = test.data  # (10000, 32, 32, 3) uint8, same order zero_shot.py iterates
    labels = np.array(test.targets)

    # --- Sanity check: reproduce zero_shot.py's 500-image accuracy -----------
    eval_feats = features(model, device, images[:EVAL_SUBSET_SIZE], label_features, logit_scale)
    eval_acc = accuracy(eval_feats["probs"], labels[:EVAL_SUBSET_SIZE])
    print(f"Eval subset (indices 0-{EVAL_SUBSET_SIZE - 1}) zero-shot accuracy: {100 * eval_acc:.1f}%")

    # --- Disjoint pools from indices 500..9999 ---------------------------------
    split_rng = np.random.default_rng([args.seed, 0])
    available = split_rng.permutation(np.arange(EVAL_SUBSET_SIZE, len(images)))
    sizes = [args.reference_size, args.pool_size, args.pool_size, args.pool_size]
    assert sum(sizes) <= len(available), "pools don't fit in the 9,500 non-eval test images"
    bounds = np.cumsum([0, *sizes])
    pools = {
        name: np.sort(available[bounds[i] : bounds[i + 1]])
        for i, name in enumerate(["reference", "calibration", "heldout", "shift"])
    }

    print("Encoding clean pools...")
    clean = {name: features(model, device, images[idx], label_features, logit_scale) for name, idx in pools.items()}
    pool_labels = {name: labels[idx] for name, idx in pools.items()}

    reference = dm.DriftReference(
        clean["reference"]["stats"], clean["reference"]["embeddings"], clean["reference"]["probs"], seed=args.seed
    )
    print(f"RBF bandwidth (median heuristic on reference): {reference.bandwidth:.4f}")

    # --- Calibrate thresholds, then measure false alarms on held-out clean ---
    perm_rng = np.random.default_rng([args.seed, 4])
    cal_batches = dm.sample_batches(args.pool_size, args.batch_size, args.n_clean_batches, np.random.default_rng([args.seed, 1]))
    held_batches = dm.sample_batches(args.pool_size, args.batch_size, args.n_clean_batches, np.random.default_rng([args.seed, 2]))
    shift_batches = dm.sample_batches(args.pool_size, args.batch_size, args.n_shift_batches, np.random.default_rng([args.seed, 3]))

    print("Scoring calibration and held-out clean batches...")
    cal_scores, cal_p, _ = score_batches(reference, clean["calibration"], cal_batches, args.n_permutations, perm_rng)
    held_scores, held_p, _ = score_batches(reference, clean["heldout"], held_batches, args.n_permutations, perm_rng)

    thresholds = {s: dm.calibrate_threshold(cal_scores[s], args.false_alarm_rate) for s in dm.SIGNALS}
    n_held = len(held_batches)
    false_alarms = {}
    for s in dm.SIGNALS:
        k = int(np.sum(held_scores[s] > thresholds[s]))
        false_alarms[s] = {
            "calibration_rate": dm.alarm_rate(cal_scores[s], thresholds[s]),
            "heldout_rate": k / n_held,
            "heldout_ci95": dm.wilson_interval(k, n_held),
            "heldout_alarms": k,
            "heldout_batches": n_held,
        }
    # The permutation test also gives an uncalibrated rule (alert if
    # p < alpha). Reported separately as a check on the test's validity.
    k = int(np.sum(held_p < args.false_alarm_rate))
    mmd_pvalue_rule = {
        "alpha": args.false_alarm_rate,
        "calibration_rate": float(np.mean(cal_p < args.false_alarm_rate)),
        "heldout_rate": k / n_held,
        "heldout_ci95": dm.wilson_interval(k, n_held),
    }
    for s in dm.SIGNALS:
        fa = false_alarms[s]
        print(f"  {s:22s} threshold={thresholds[s]:.6g}  FAR cal={fa['calibration_rate']:.3f} heldout={fa['heldout_rate']:.3f}")
    print(f"  mmd p<{args.false_alarm_rate} rule        FAR cal={mmd_pvalue_rule['calibration_rate']:.3f} heldout={mmd_pvalue_rule['heldout_rate']:.3f}")

    # --- Shift conditions ----------------------------------------------------
    # Every condition is scored on the SAME batch indices into the shift
    # pool (paired design), so differences between severities come from the
    # corruption, not from which images happened to be drawn.
    def evaluate_condition(name, severity, param, feats):
        scores, pvals, batch_acc = score_batches(
            reference, feats, shift_batches, args.n_permutations, perm_rng, labels=pool_labels["shift"]
        )
        stats = feats["stats"]
        conf = dm.confidence_stats(feats["probs"])
        return {
            "corruption": name,
            "severity": severity,
            "parameter": param,
            "accuracy": accuracy(feats["probs"], pool_labels["shift"]),
            "batch_accuracy_mean": float(batch_acc.mean()),
            "batch_accuracy_std": float(batch_acc.std()),
            "mean_brightness": float(stats[:, 0].mean()),
            "mean_contrast": float(stats[:, 1].mean()),
            "mean_sharpness": float(stats[:, 2].mean()),
            "mean_top_prob": float(conf[:, 0].mean()),
            "mean_entropy": float(conf[:, 1].mean()),
            "signals": {
                s: {
                    "mean_score": float(scores[s].mean()),
                    "std_score": float(scores[s].std()),
                    "detection_rate": dm.alarm_rate(scores[s], thresholds[s]),
                }
                for s in dm.SIGNALS
            },
            "mmd_pvalue_rule_detection_rate": float(np.mean(pvals < args.false_alarm_rate)),
            "_batch_scores": scores,
            "_batch_accuracy": batch_acc,
        }

    conditions = [evaluate_condition("none", 0, None, clean["shift"])]
    clean_shift_acc = conditions[0]["accuracy"]
    shift_images = images[pools["shift"]]
    for name, (_, params) in CORRUPTIONS.items():
        for severity in SEVERITIES:
            t0 = time.time()
            corrupted = corrupt_batch(shift_images, name, severity, seed=args.seed, indices=pools["shift"])
            feats = features(model, device, corrupted, label_features, logit_scale)
            cond = evaluate_condition(name, severity, params[severity - 1], feats)
            conditions.append(cond)
            print(
                f"  {name:22s} sev {severity}: acc={100 * cond['accuracy']:.1f}%  "
                + "  ".join(f"{s.split('_', 1)[1] if s != 'input_stats' else 'input'}={cond['signals'][s]['detection_rate']:.2f}" for s in dm.SIGNALS)
                + f"  ({time.time() - t0:.0f}s)"
            )
    for cond in conditions:
        cond["accuracy_drop"] = clean_shift_acc - cond["accuracy"]

    # --- How drift scores track the accuracy drop ----------------------------
    # Condition level: Spearman over (mean score, accuracy drop), across all
    # 21 conditions and within each corruption's 6 severity levels (0-5).
    # Batch level: Spearman over every individual shift batch's (score,
    # batch accuracy drop vs the clean shift-pool accuracy).
    correlation = {}
    for s in dm.SIGNALS:
        per_corruption = {}
        for name in CORRUPTIONS:
            subset = [conditions[0]] + [c for c in conditions if c["corruption"] == name]
            per_corruption[name] = dm.spearman(
                [c["signals"][s]["mean_score"] for c in subset], [c["accuracy_drop"] for c in subset]
            )
        batch_scores = np.concatenate([c["_batch_scores"][s] for c in conditions])
        batch_drop = clean_shift_acc - np.concatenate([c["_batch_accuracy"] for c in conditions])
        correlation[s] = {
            "spearman_condition_level_all": dm.spearman(
                [c["signals"][s]["mean_score"] for c in conditions], [c["accuracy_drop"] for c in conditions]
            ),
            "spearman_condition_level_per_corruption": per_corruption,
            "spearman_batch_level_all": dm.spearman(batch_scores, batch_drop),
            "n_conditions": len(conditions),
            "n_batches": int(len(batch_scores)),
        }
    for cond in conditions:
        del cond["_batch_scores"], cond["_batch_accuracy"]

    ref_conf = dm.confidence_stats(clean["reference"]["probs"])
    results = {
        "config": {
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "logit_scale": logit_scale,
            "prompts": [f"a photo of a {c}" for c in CIFAR10_CLASSES],
            "excluded_eval_subset": f"CIFAR-10 test indices 0-{EVAL_SUBSET_SIZE - 1}",
            "pool_sizes": {k: int(len(v)) for k, v in pools.items()},
            "batch_size": args.batch_size,
            "n_calibration_batches": len(cal_batches),
            "n_heldout_batches": len(held_batches),
            "n_shift_batches_per_condition": len(shift_batches),
            "batch_sampling": "each batch = batch_size distinct images drawn uniformly from its pool; batches may overlap",
            "n_permutations": args.n_permutations,
            "rbf_bandwidth": reference.bandwidth,
            "target_false_alarm_rate": args.false_alarm_rate,
            "corruptions": {k: list(v[1]) for k, v in CORRUPTIONS.items()},
        },
        "sanity_check": {
            "eval_subset_accuracy": eval_acc,
            "eval_subset_size": EVAL_SUBSET_SIZE,
        },
        "reference": {
            "n_images": int(len(pools["reference"])),
            "accuracy": accuracy(clean["reference"]["probs"], pool_labels["reference"]),
            # Norm of the mean of unit-length embeddings: near 0 would make the
            # cosine-of-means signal ill-conditioned; near 1 means a strong
            # shared direction.
            "mean_embedding_norm": float(np.linalg.norm(clean["reference"]["embeddings"].mean(axis=0))),
            "mean_brightness": float(clean["reference"]["stats"][:, 0].mean()),
            "mean_contrast": float(clean["reference"]["stats"][:, 1].mean()),
            "mean_sharpness": float(clean["reference"]["stats"][:, 2].mean()),
            "mean_top_prob": float(ref_conf[:, 0].mean()),
            "mean_entropy": float(ref_conf[:, 1].mean()),
        },
        "pool_accuracy": {k: accuracy(clean[k]["probs"], pool_labels[k]) for k in pools},
        "thresholds": thresholds,
        "false_alarm": false_alarms,
        "mmd_pvalue_rule": mmd_pvalue_rule,
        "conditions": conditions,
        "correlation_with_accuracy_drop": correlation,
        "runtime_seconds": time.time() - started,
    }

    json_path = os.path.join(args.out_dir, "drift_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    plot_results(results, args.out_dir)

    # Reference artifact for serving: raw per-image stats + embeddings of the
    # reference and calibration pools. Probabilities are NOT stored, because
    # the serving app uses a different label set; drift_monitor.py recomputes
    # them (and re-calibrates thresholds) for whatever labels it serves.
    np.savez_compressed(
        args.reference_out,
        reference_stats=clean["reference"]["stats"],
        reference_embeddings=clean["reference"]["embeddings"].astype(np.float32),
        calibration_stats=clean["calibration"]["stats"],
        calibration_embeddings=clean["calibration"]["embeddings"].astype(np.float32),
        stats_resolution=np.array([32, 32]),
        bandwidth=np.array(reference.bandwidth),
        seed=np.array(args.seed),
    )
    print(f"\nWrote {json_path}, plots in {args.out_dir}/, reference artifact {args.reference_out}")
    print(f"Total runtime: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
