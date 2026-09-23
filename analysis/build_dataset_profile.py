"""Build a reproducible profile of LaKun's six final JSONL splits."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
SPLITS = ("train", "val", "test")
MODALITIES = ("image", "text")
TYPES = ("choice", "score", "noul")
COLORS = {"image": "#2563eb", "text": "#f97316"}

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def counter_dict(counter: Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def count_at(counter: dict, key: int) -> int:
    return counter.get(key, counter.get(str(key), 0))


def hist_bucket(value: float, bins: int = 5) -> int:
    return min(bins - 1, int(value * bins))


def has_han(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def collect() -> dict:
    result = {
        "splits": {},
        "by_modality": {},
        "question_types": {},
        "option_counts": {},
        "label_position_quintiles": {},
        "choice_k4_label_positions": {},
        "noul_labels": {},
        "questions_per_group": {},
        "image_sources": {},
        "text_domains": {},
        "question_contains_han": {},
        "token_budget": {},
        "integrity": {},
    }
    all_group_ids = {modality: set() for modality in MODALITIES}
    type_counts = defaultdict(Counter)
    option_counts = defaultdict(Counter)
    label_positions = defaultdict(Counter)
    choice_k4_positions = defaultdict(Counter)
    noul_labels = defaultdict(Counter)
    group_sizes = defaultdict(Counter)
    sources = Counter()
    domains = Counter()
    han = defaultdict(Counter)
    token_buckets = defaultdict(Counter)
    faults = Counter()

    for modality in MODALITIES:
        for split in SPLITS:
            path = ROOT / "dataset" / f"{modality}_{split}.jsonl"
            seen_ids = set()
            previous_group = None
            previous_count = 0
            rows = 0
            groups = 0
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    rows += 1
                    group_id = row["group_id"]
                    if row["split"] != split:
                        faults["wrong_split"] += 1
                    if group_id != previous_group:
                        if previous_group is not None:
                            group_sizes[modality][previous_count] += 1
                        if group_id in seen_ids:
                            faults["noncontiguous_group"] += 1
                        if group_id in all_group_ids[modality]:
                            faults["duplicate_group_across_splits"] += 1
                        seen_ids.add(group_id)
                        all_group_ids[modality].add(group_id)
                        previous_group = group_id
                        previous_count = 0
                        groups += 1
                        if modality == "image":
                            sources[row.get("source", "unknown")] += 1
                            if len(row.get("images", [])) != 1:
                                faults["wrong_image_count"] += 1
                        else:
                            domains[row.get("domain", "unspecified")] += 1
                            if row.get("images"):
                                faults["text_has_image"] += 1
                    previous_count += 1
                    kind = row["type"]
                    criteria = row["criteria"]
                    selected = row["selected_index"]
                    count = len(criteria)
                    type_counts[modality][kind] += 1
                    option_counts[(modality, kind)][count] += 1
                    label_positions[(modality, kind)][hist_bucket(selected / (count - 1))] += 1
                    if kind == "choice" and count == 4:
                        choice_k4_positions[modality][selected] += 1
                    if kind == "noul":
                        noul_labels[modality][criteria[selected]] += 1
                    han[modality]["contains_han" if has_han(row["question"]) else "no_han"] += 1
                    budget = row.get("token_budget", {}).get("total_budgeted_tokens")
                    if isinstance(budget, int):
                        token_buckets[modality][min(512, 50 * (budget // 50))] += 1
                        if budget > 512:
                            faults["stored_token_budget_over_512"] += 1
                    if not 0 <= selected < count:
                        faults["invalid_selected_index"] += 1
                    if len(row["target"]) != count:
                        faults["target_length_mismatch"] += 1
            if previous_group is not None:
                group_sizes[modality][previous_count] += 1
            result["splits"].setdefault(split, {})[modality] = {"groups": groups, "questions": rows}

    for modality in MODALITIES:
        result["by_modality"][modality] = {
            "groups": sum(result["splits"][split][modality]["groups"] for split in SPLITS),
            "questions": sum(result["splits"][split][modality]["questions"] for split in SPLITS),
        }
        result["question_types"][modality] = counter_dict(type_counts[modality])
        result["noul_labels"][modality] = counter_dict(noul_labels[modality])
        result["questions_per_group"][modality] = counter_dict(group_sizes[modality])
        result["question_contains_han"][modality] = counter_dict(han[modality])
        result["token_budget"][modality] = counter_dict(token_buckets[modality])
        result["option_counts"][modality] = {
            kind: counter_dict(option_counts[(modality, kind)]) for kind in TYPES
        }
        result["label_position_quintiles"][modality] = {
            kind: counter_dict(label_positions[(modality, kind)]) for kind in TYPES
        }
        result["choice_k4_label_positions"][modality] = counter_dict(choice_k4_positions[modality])
    result["image_sources"] = counter_dict(sources)
    result["text_domains"] = counter_dict(domains)
    result["integrity"] = counter_dict(faults)
    result["totals"] = {
        "groups": sum(value["groups"] for value in result["by_modality"].values()),
        "questions": sum(value["questions"] for value in result["by_modality"].values()),
    }
    return result


def finish_figure(fig, name: str) -> None:
    fig.tight_layout(pad=2.2)
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def overview(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LaKun 数据集画像：规模与任务组成", fontsize=19, fontweight="bold")
    positions = np.arange(3)
    for offset, modality in ((-0.19, "image"), (0.19, "text")):
        vals = [data["splits"][split][modality]["groups"] for split in SPLITS]
        axes[0, 0].bar(positions + offset, vals, 0.36, label="图像" if modality == "image" else "文本", color=COLORS[modality])
        for x, y in zip(positions + offset, vals):
            axes[0, 0].text(x, y + 400, f"{y:,}", ha="center", fontsize=9)
    axes[0, 0].set_xticks(positions, ["训练", "验证", "测试"])
    axes[0, 0].set_ylabel("组数")
    axes[0, 0].set_title("8:1:1 分组划分")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_ylim(0, 61000)

    for i, modality in enumerate(MODALITIES):
        vals = [data["question_types"][modality][kind] for kind in TYPES]
        left = 0
        for kind, val, color in zip(TYPES, vals, ("#2763a5", "#54a483", "#bd6b45")):
            axes[0, 1].barh(i, val, left=left, color=color, label=kind if i == 0 else None)
            label = f"{kind}\n{val:,}" if i == 0 else f"{val:,}"
            axes[0, 1].text(left + val / 2, i, label, ha="center", va="center", color="white", fontsize=9)
            left += val
    axes[0, 1].set_yticks([0, 1], ["图像", "文本"])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("问题数")
    axes[0, 1].set_title("选择 / 评分 / 判别任务")

    for modality in MODALITIES:
        counts = data["questions_per_group"][modality]
        xs = sorted(int(k) for k in counts)
        axes[1, 0].plot(xs, [count_at(counts, k) for k in xs], marker="o", label="图像" if modality == "image" else "文本", color=COLORS[modality])
    axes[1, 0].set_xticks(range(1, 8))
    axes[1, 0].set_xlim(0.8, 7.2)
    axes[1, 0].set_xlabel("每组问题数")
    axes[1, 0].set_ylabel("组数")
    axes[1, 0].set_title("一张图 / 一段文本对应多少问题")
    axes[1, 0].legend(frameon=False)

    positions = np.arange(3)
    for offset, modality in ((-0.19, "image"), (0.19, "text")):
        vals = [data["splits"][split][modality]["questions"] for split in SPLITS]
        axes[1, 1].bar(positions + offset, vals, 0.36,
                       label="图像" if modality == "image" else "文本", color=COLORS[modality])
        for x, y in zip(positions + offset, vals):
            axes[1, 1].text(x, y + 1300, f"{y:,}", ha="center", fontsize=9)
    axes[1, 1].set_xticks(positions, ["训练", "验证", "测试"])
    axes[1, 1].set_ylabel("问题数")
    axes[1, 1].set_title("各集合的问题数量")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_ylim(0, 180000)
    finish_figure(fig, "01-overview.png")


def options_and_labels(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("LaKun 数据集画像：选项与答案分布", fontsize=19, fontweight="bold")
    for axis, kind, title in ((axes[0, 0], "choice", "选择题候选项数量"),
                              (axes[0, 1], "score", "评分题档位数量")):
        counts = {modality: data["option_counts"][modality][kind] for modality in MODALITIES}
        ks = sorted({int(k) for values in counts.values() for k in values})
        x = np.arange(len(ks))
        for offset, modality in ((-0.19, "image"), (0.19, "text")):
            vals = [count_at(counts[modality], k) for k in ks]
            axis.bar(x + offset, vals, 0.36, label="图像" if modality == "image" else "文本", color=COLORS[modality])
        axis.set_xticks(x, ks)
        axis.set_xlabel("选项 / 档位个数")
        axis.set_ylabel("问题数")
        axis.set_title(title)
        axis.legend(frameon=False)

    x = np.arange(5)
    for offset, modality in ((-0.19, "image"), (0.19, "text")):
        counts = data["label_position_quintiles"][modality]["score"]
        total = sum(counts.values())
        vals = [100 * count_at(counts, i) / total for i in range(5)]
        axes[1, 0].bar(x + offset, vals, 0.36, label="图像" if modality == "image" else "文本", color=COLORS[modality])
    axes[1, 0].set_xticks(x, ["最低 20%", "20–40%", "40–60%", "60–80%", "最高 20%"])
    axes[1, 0].set_ylabel("评分题占比（%）")
    axes[1, 0].set_title("教师标记档位在各题候选范围中的相对位置")
    axes[1, 0].legend(frameon=False)

    for i, modality in enumerate(MODALITIES):
        counts = data["noul_labels"][modality]
        false, true = counts.get("false", 0), counts.get("true", 0)
        total = false + true
        axes[1, 1].barh(i, 100 * false / total, color="#6b7280", label="false" if i == 0 else None)
        axes[1, 1].barh(i, 100 * true / total, left=100 * false / total, color="#54a483", label="true" if i == 0 else None)
        axes[1, 1].text(100 * false / total / 2, i, f"false {false / total:.1%}", ha="center", va="center", color="white")
        axes[1, 1].text(100 * false / total + 50 * true / total, i, f"true {true / total:.1%}", ha="center", va="center", color="white")
    axes[1, 1].set_yticks([0, 1], ["图像", "文本"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("判别题占比（%）")
    axes[1, 1].set_title("判别题标签平衡")
    finish_figure(fig, "02-options-and-labels.png")


def coverage(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("LaKun 数据集画像：来源、主题与语言", fontsize=19, fontweight="bold")
    source_counts = Counter(data["image_sources"])
    names, vals = zip(*source_counts.most_common(10))
    axes[0, 0].barh(range(len(names)), vals, color=COLORS["image"])
    axes[0, 0].set_yticks(range(len(names)), names)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("图片组数")
    axes[0, 0].set_title("图片来源 Top 10")

    domain_counts = Counter(data["text_domains"])
    names, vals = zip(*domain_counts.most_common(15))
    axes[0, 1].barh(range(len(names)), vals, color=COLORS["text"])
    axes[0, 1].set_yticks(range(len(names)), names)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("文本组数")
    axes[0, 1].set_title(f"文本主题 Top 15（共 {len(domain_counts)} 类）")

    for i, modality in enumerate(MODALITIES):
        counts = data["question_contains_han"][modality]
        total = sum(counts.values())
        yes = 100 * counts.get("contains_han", 0) / total
        axes[1, 0].barh(i, yes, color=COLORS[modality])
        axes[1, 0].text(yes + 1, i, f"{yes:.1f}%", va="center")
    axes[1, 0].set_yticks([0, 1], ["图像问题", "文本问题"])
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, 108)
    axes[1, 0].set_xlabel("含汉字的问题比例（%）")
    axes[1, 0].set_title("语言提示：字符启发式，并非语种识别")

    for modality in MODALITIES:
        counts = data["token_budget"][modality]
        xs = sorted(int(k) for k in counts)
        ys = np.cumsum([count_at(counts, k) for k in xs]) / sum(counts.values()) * 100
        axes[1, 1].step(xs, ys, where="post", label="图像" if modality == "image" else "文本", color=COLORS[modality], linewidth=2)
    axes[1, 1].set_xlim(0, 525)
    axes[1, 1].set_ylim(0, 102)
    axes[1, 1].set_xlabel("构建时 token 预算（每 50 token 分箱）")
    axes[1, 1].set_ylabel("累计问题占比（%）")
    axes[1, 1].set_title("序列长度预算分布")
    axes[1, 1].legend(frameon=False)
    finish_figure(fig, "03-coverage.png")


def domains_and_choice_labels(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 19), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.suptitle("LaKun 数据集画像：50 个文本方向与四选一标签位置", fontsize=19, fontweight="bold")
    ordered = sorted(data["text_domains"].items(), key=lambda item: (-item[1], item[0]))
    names, vals = zip(*ordered)
    axes[0].barh(range(len(names)), vals, color=COLORS["text"])
    axes[0].set_yticks(range(len(names)), names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("文本组数")
    axes[0].set_xlim(0, 1450)
    axes[0].set_title("全部 50 个文本方向（每类约 1,333 组）")
    for y, val in enumerate(vals):
        axes[0].text(val + 9, y, str(val), va="center", fontsize=8)

    x = np.arange(4)
    for offset, modality in ((-0.19, "image"), (0.19, "text")):
        counts = data["choice_k4_label_positions"][modality]
        total = sum(counts.values())
        values = [100 * count_at(counts, i) / total for i in range(4)]
        axes[1].bar(x + offset, values, 0.36,
                    label="图像" if modality == "image" else "文本", color=COLORS[modality])
    axes[1].set_xticks(x, ["A / 位置 1", "B / 位置 2", "C / 位置 3", "D / 位置 4"])
    axes[1].set_ylabel("选择题占比（%）")
    axes[1].set_title("四选一题：教师标记答案的位置")
    axes[1].legend(frameon=False)
    finish_figure(fig, "04-domains-and-choice-labels.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    data = collect()
    (OUT / "stats.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    overview(data)
    options_and_labels(data)
    coverage(data)
    domains_and_choice_labels(data)
    print(json.dumps({"totals": data["totals"], "integrity": data["integrity"],
                      "figures": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
