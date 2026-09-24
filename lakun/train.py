"""Train LaKun's mmBERT and SigLIP towers on image/text decision groups.

Smoke test (10 train groups, 10 test groups, tiny random encoders):
    python train_lakun.py --smoke

Full pretrained run:
    python train_lakun.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import (BertConfig, BertModel, PreTrainedTokenizerFast,
                          SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel)

from lakun.vendor.laya.common import DecisionModel, proper_reward

from lakun.data import DecisionGroups
from lakun.encoding import MAX_CONTEXT, VISUAL_TOKENS, prepare_group
from lakun.inference import load_checkpoint_into
from lakun.model import LaKunModel
from lakun.model_card import write_model_card
from lakun.pretrained import initialize_base_components


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def decision_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    probabilities = F.softmax(logits.float(), dim=-1)
    return -proper_reward(probabilities, batch["target"], batch["qtype"],
                          batch["marker_mask"], w_sph=0.5, w_rps=1.0).mean()


def make_smoke_components(project_root: Path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(project_root / "tokenizer" / "tokenizer.json"),
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
    text_config = BertConfig(vocab_size=len(tokenizer), hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=256, max_position_embeddings=512)
    decision_model = DecisionModel(BertModel(text_config), head_layers=1, n_act=2)
    vision_config = SiglipVisionConfig(hidden_size=128, intermediate_size=256,
                                       num_hidden_layers=2, num_attention_heads=4,
                                       image_size=112, patch_size=16)
    vision_model = SiglipVisionModel(vision_config)
    processor = SiglipImageProcessor(size={"height": 112, "width": 112})
    return tokenizer, processor, LaKunModel(decision_model, vision_model, 16)


def make_pretrained_components(args):
    tokenizer, processor, model = initialize_base_components(args.text_model, args.vision_model)
    model.decision.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tokenizer, processor, model


def optimizer_for(model: LaKunModel, smoke: bool):
    text_lr = 1e-3 if smoke else 2e-5
    vision_lr = 1e-3 if smoke else 1e-5
    bridge_lr = 1e-3 if smoke else 1e-4
    head_parameters = (list(model.decision.head.parameters()) if model.decision.head is not None else [])
    head_parameters += list(model.decision.type_emb.parameters()) + list(model.decision.scorer.parameters())
    bridge_parameters = list(model.visual_projection.parameters())
    return torch.optim.AdamW([
        {"name": "text_encoder", "params": model.decision.encoder.parameters(),
         "lr": text_lr, "base_lr": text_lr},
        {"name": "decision_head", "params": head_parameters,
         "lr": bridge_lr, "base_lr": bridge_lr},
        {"name": "vision_encoder", "params": model.vision.parameters(),
         "lr": vision_lr, "base_lr": vision_lr},
        {"name": "visual_bridge", "params": bridge_parameters,
         "lr": bridge_lr, "base_lr": bridge_lr},
    ], weight_decay=0.01)


def shuffle_choice_options(group: dict, rng: random.Random) -> dict:
    """Randomize nominal choice positions while preserving semantic labels."""
    rows = []
    for row in group["rows"]:
        if row["type"] != "choice" or len(row["criteria"]) < 2:
            rows.append(row)
            continue
        order = list(range(len(row["criteria"])))
        rng.shuffle(order)
        inverse = {old: new for new, old in enumerate(order)}
        updated = dict(row)
        updated["criteria"] = [row["criteria"][old] for old in order]
        updated["target"] = [row["target"][old] for old in order]
        updated["selected_index"] = inverse[row["selected_index"]]
        rows.append(updated)
    return dict(group, rows=rows)


def run_epoch(model, loader, tokenizer, processor, device, optimizer=None, amp=False,
              epoch=1, log_every=100, rank=0, start_step=0, on_step=None,
              on_train_step=None, shuffle_choices=False, choice_rng=None) -> dict:
    training = optimizer is not None
    core_model = model.module if isinstance(model, DistributedDataParallel) else model
    model.train(training)
    groups = 0
    questions = 0
    losses = 0.0
    correct = Counter()
    total = Counter()
    modalities = Counter()
    joint_gradients = {"text": False, "vision": False}
    started = time.monotonic()
    stopped_early = False
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, group in enumerate(loader, 1):
            if training and shuffle_choices:
                group = shuffle_choice_options(group, choice_rng or random)
            if training and on_train_step is not None:
                on_train_step(start_step + step)
            batch, pixels = prepare_group(group, tokenizer, processor, device,
                                          core_model.visual_tokens)
            if training:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
            with autocast:
                logits = model(batch, pixels)
                loss = decision_loss(logits, batch)
            if training:
                loss.backward()
                joint_gradients["text"] |= any(p.grad is not None for p in core_model.decision.encoder.parameters())
                if pixels is not None:
                    joint_gradients["vision"] |= any(p.grad is not None for p in core_model.vision.parameters())
                torch.nn.utils.clip_grad_norm_(core_model.parameters(), 1.0)
                optimizer.step()
            prediction = logits.argmax(-1)
            for i, row in enumerate(group["rows"]):
                kind = row["type"]
                total[kind] += 1
                correct[kind] += int(prediction[i].item() == row["selected_index"])
            modalities[group["modality"]] += 1
            groups += 1
            questions += len(group["rows"])
            losses += loss.item() * len(group["rows"])
            if training and rank == 0 and log_every and step % log_every == 0:
                elapsed = time.monotonic() - started
                print(json.dumps({"epoch": epoch, "step": step, "steps_per_rank": len(loader),
                                  "local_questions": questions, "local_mean_loss": round(losses / questions, 6),
                                  "elapsed_seconds": round(elapsed, 1)}, ensure_ascii=False), flush=True)
            if training and on_step is not None and on_step(start_step + step):
                stopped_early = True
                break
    return {"groups": groups, "questions": questions, "loss_sum": losses,
            "correct": correct, "total": total, "modality_groups": modalities,
            "gradients_seen": joint_gradients, "steps": groups,
            "stopped_early": stopped_early}


def summarize_epoch(raw: dict, device: torch.device, training: bool) -> dict:
    types = ("choice", "score", "noul")
    numbers = [raw["groups"], raw["questions"], raw["loss_sum"]]
    numbers += [raw["correct"][kind] for kind in types]
    numbers += [raw["total"][kind] for kind in types]
    numbers += [raw["modality_groups"][kind] for kind in ("image", "text")]
    numbers += [int(raw["gradients_seen"][kind]) for kind in ("text", "vision")]
    stats = torch.tensor(numbers, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    values = stats.tolist()
    if not values[1]:
        raise ValueError("epoch processed no questions")
    result = {"groups": int(values[0]), "questions": int(values[1]),
              "loss": round(values[2] / values[1], 6),
              "accuracy": {kind: round(values[3 + i] / values[6 + i], 4)
                           for i, kind in enumerate(types) if values[6 + i]},
              "modality_groups": {kind: int(values[9 + i]) for i, kind in enumerate(("image", "text"))
                                  if values[9 + i]}}
    if training:
        result["gradients_seen"] = {kind: bool(values[11 + i])
                                    for i, kind in enumerate(("text", "vision"))}
    return result


def save_checkpoint(model, tokenizer, processor, args, output: Path):
    from huggingface_hub import save_torch_model

    output.mkdir(parents=True, exist_ok=True)
    save_torch_model(model, output, filename_pattern="lakun{suffix}.safetensors",
                     max_shard_size="10GB")
    model.decision.encoder.config.save_pretrained(output / "text_encoder")
    model.vision.config.save_pretrained(output / "vision_encoder")
    tokenizer.save_pretrained(output / "tokenizer")
    processor.save_pretrained(output / "image_processor")
    base_models = ({"text": "tiny_random_BERT_fixture", "vision": "tiny_random_SigLIP_fixture"}
                   if args.smoke else
                   {"text": "jhu-clsp/mmBERT-base", "vision": "google/siglip-so400m-patch14-384"})
    config = {"model_type": "lakun", "architecture": "LaKunModel",
              "base_models": base_models,
              "visual_tokens": model.visual_tokens,
              "visual_resampler": model.visual_resampler,
              "context_tokens": MAX_CONTEXT,
              "decision_head_layers": len(model.decision.head.layers) if model.decision.head is not None else 0,
              "decision_n_act": model.decision.act_head[-1].out_features}
    (output / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Joint LaKun mmBERT + SigLIP training")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="root containing dataset/ and images/; defaults to project root")
    parser.add_argument("--smoke", action="store_true", help="run small train/val/test groups with tiny random encoders")
    parser.add_argument("--train-groups", type=int, default=None)
    parser.add_argument("--val-groups", type=int, default=None)
    parser.add_argument("--test-groups", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3, help="maximum epochs; early stopping may finish sooner")
    parser.add_argument("--eval-every", type=int, default=3000, help="optimizer steps between validation checks")
    parser.add_argument("--patience", type=int, default=3, help="validation checks without improvement before stopping")
    parser.add_argument("--min-delta", type=float, default=0.001, help="minimum validation loss improvement")
    parser.add_argument("--text-model", type=Path)
    parser.add_argument("--vision-model", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--freeze-backbones-steps", type=int, default=None,
                        help="explicit frozen update count; overrides --freeze-backbones-ratio")
    parser.add_argument("--freeze-backbones-ratio", type=float, default=0.05,
                        help="freeze text/vision backbones for this fraction of the first epoch")
    parser.add_argument("--shuffle-choice-options", action=argparse.BooleanOptionalAction,
                        default=True, help="shuffle nominal choice options online during training")
    args = parser.parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if args.device.startswith("cuda") else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://?use_libuv=0")
    project_root = args.project_root.resolve()
    preferred_data_root = project_root / "new_dataset"
    default_data_root = (preferred_data_root
                         if (preferred_data_root / "dataset").is_dir() else project_root)
    data_root = (args.data_root or default_data_root).resolve()
    args.text_model = args.text_model or project_root / "weights" / "mmbert-base"
    args.vision_model = args.vision_model or project_root / "weights" / "siglip-so400m-patch14-384"
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.freeze_backbones_steps is not None and args.freeze_backbones_steps < 0:
        raise ValueError("freeze-backbones-steps must be nonnegative")
    if not 0 <= args.freeze_backbones_ratio < 1:
        raise ValueError("freeze-backbones-ratio must be in [0, 1)")
    if args.eval_every < 1 or args.patience < 1 or args.min_delta < 0:
        raise ValueError("eval-every/patience must be positive and min-delta nonnegative")
    if args.smoke:
        args.train_groups = args.train_groups or 10
        args.val_groups = args.val_groups or 10
        args.test_groups = args.test_groups or 10
        if args.freeze_backbones_steps is None:
            args.freeze_backbones_steps = 0
    output = args.out or project_root / "runs" / ("lakun_smoke" if args.smoke else "lakun_full")
    output = output.resolve()
    random.seed(20260923)
    torch.manual_seed(20260923)
    device = torch.device(f"cuda:{local_rank}" if distributed and args.device.startswith("cuda") else args.device)
    train_data = DecisionGroups(data_root, "train", args.train_groups)
    val_data = DecisionGroups(data_root, "val", args.val_groups)
    test_data = DecisionGroups(data_root, "test", args.test_groups)
    if distributed and len(train_data) < world_size:
        raise ValueError(f"{len(train_data)} training groups cannot cover {world_size} ranks")
    train_sampler = (DistributedSampler(train_data, num_replicas=world_size, rank=rank,
                                        shuffle=True, seed=20260923, drop_last=True)
                     if distributed else None)
    train_loader = DataLoader(train_data, batch_size=None, sampler=train_sampler,
                              shuffle=train_sampler is None, num_workers=0)
    if args.freeze_backbones_steps is None:
        args.freeze_backbones_steps = max(1, round(len(train_loader) * args.freeze_backbones_ratio))
    val_subset = (Subset(val_data, range(rank, len(val_data), world_size))
                  if distributed else val_data)
    val_loader = DataLoader(val_subset, batch_size=None, shuffle=False, num_workers=0)
    test_subset = (Subset(test_data, range(rank, len(test_data), world_size))
                   if distributed else test_data)
    test_loader = DataLoader(test_subset, batch_size=None, shuffle=False, num_workers=0)
    if rank == 0:
        print(json.dumps({"data_root": str(data_root),
                          "train_groups": len(train_data),
                          "val_groups": len(val_data),
                          "test_groups": len(test_data),
                          "steps_per_rank": len(train_loader),
                          "world_size": world_size}, ensure_ascii=False), flush=True)
    tokenizer, processor, model = (make_smoke_components(project_root) if args.smoke
                                   else make_pretrained_components(args))
    model.to(device)
    optimizer = optimizer_for(model, args.smoke)
    backbone_frozen = None

    def set_backbone_stage(step: int) -> None:
        nonlocal backbone_frozen
        frozen = step <= args.freeze_backbones_steps
        if frozen == backbone_frozen:
            return
        backbone_frozen = frozen
        for group in optimizer.param_groups:
            if group.get("name") in {"text_encoder", "vision_encoder"}:
                group["lr"] = 0.0 if frozen else group["base_lr"]
                for parameter in group["params"]:
                    parameter.requires_grad_(not frozen)
        if rank == 0:
            print(json.dumps({"training_stage": "bridge_and_head" if frozen else "joint_finetune",
                              "step": step}, ensure_ascii=False), flush=True)

    training_model = (DistributedDataParallel(model,
                                              device_ids=[local_rank] if device.type == "cuda" else None,
                                              find_unused_parameters=True, broadcast_buffers=False)
                      if distributed else model)
    set_backbone_stage(1)
    use_amp = device.type == "cuda" and not args.smoke
    history = []
    validations = []
    best_val_loss = float("inf")
    best_step = None
    stale_checks = 0
    global_step = 0
    last_validation_step = -1
    best_dir = output / "best"

    def check_validation(step: int, epoch: int) -> bool:
        nonlocal best_val_loss, best_step, stale_checks, last_validation_step
        if step == last_validation_step:
            return stale_checks >= args.patience
        last_validation_step = step
        val_raw = run_epoch(model, val_loader, tokenizer, processor, device,
                            amp=use_amp, epoch=epoch, rank=rank)
        metrics = summarize_epoch(val_raw, device, training=False)
        improved = metrics["loss"] < best_val_loss - args.min_delta
        if improved:
            best_val_loss = metrics["loss"]
            best_step = step
            stale_checks = 0
            if rank == 0:
                save_checkpoint(model, tokenizer, processor, args, best_dir)
        else:
            stale_checks += 1
        if distributed:
            dist.barrier()
        result = {"epoch": epoch, "step": step, "val": metrics,
                  "improved": improved, "checks_without_improvement": stale_checks}
        validations.append(result)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / "validation_history.json").write_text(
                json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"validation": result}, ensure_ascii=False), flush=True)
        model.train()
        return stale_checks >= args.patience

    stopped_early = False
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        def after_step(step: int) -> bool:
            return check_validation(step, epoch + 1) if step % args.eval_every == 0 else False
        train_raw = run_epoch(training_model, train_loader, tokenizer, processor, device,
                              optimizer, use_amp, epoch + 1, args.log_every, rank,
                              start_step=global_step, on_step=after_step,
                              on_train_step=set_backbone_stage,
                              shuffle_choices=args.shuffle_choice_options,
                              choice_rng=random.Random(20260923 + epoch * world_size + rank))
        global_step += train_raw["steps"]
        train = summarize_epoch(train_raw, device, training=True)
        stopped_early = train_raw["stopped_early"]
        if not stopped_early:
            stopped_early = check_validation(global_step, epoch + 1)
        result = {"epoch": epoch + 1, "train": train}
        history.append(result)
        if rank == 0:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if stopped_early:
            break
    if best_step is None:
        raise RuntimeError("validation never selected a checkpoint")
    if distributed:
        dist.barrier()
    load_checkpoint_into(model, best_dir)
    test_raw = run_epoch(model, test_loader, tokenizer, processor, device,
                         amp=use_amp, epoch=len(history), rank=rank)
    test = summarize_epoch(test_raw, device, training=False)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        report = {"mode": "tiny_random_smoke" if args.smoke else "pretrained_joint_finetune",
                  "history": history, "validation_checks": validations,
                  "best_step": best_step, "best_val_loss": best_val_loss,
                  "stopped_early": stopped_early, "test": test,
                  "train_groups": len(train_data), "val_groups": len(val_data), "test_groups": len(test_data),
                  "distributed_world_size": world_size,
                  "data_root": str(data_root),
                  "visual_tokens": model.visual_tokens,
                  "visual_resampler": model.visual_resampler,
                  "freeze_backbones_steps": args.freeze_backbones_steps,
                  "freeze_backbones_ratio": args.freeze_backbones_ratio,
                  "shuffle_choice_options": args.shuffle_choice_options,
                  "dropped_train_groups_per_epoch": len(train_data) % world_size if distributed else 0,
                  "text_model": str(args.text_model) if not args.smoke else "tiny_random_BERT_fixture",
                  "vision_model": str(args.vision_model) if not args.smoke else "tiny_random_SigLIP_fixture"}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_model_card(best_dir, report)
        print(f"Best checkpoint directory: {best_dir}", flush=True)
        print(f"Report: {output / 'report.json'}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
