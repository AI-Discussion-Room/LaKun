# LaKun 启动说明

LaKun 使用原始 mmBERT-base 文本权重
原始 SigLIP SO400M 图像权重，
以及受 Laya 启发的决策评分头，不加载 Laya 微调权重。
项目包名是 `lakun`


## 放到服务器并训练


保留 `lakun/`、
`train_lakun.py`、
`predict_lakun.py`、
`pyproject.toml`、
`tokenizer/`、
`images/`、
`dataset/` 中图文各三份 `*_train.jsonl`／`*_val.jsonl`／`*_test.jsonl`
以及 `weights/mmbert-base/` 与 `weights/siglip-so400m-patch14-384/` 中的正式权重及配置。
划分按图片／文本 group ID 完成，比例约 8:1:1，同组问题不跨集合。图片路径由数据行引用，必须保留所引用图片及相对目录结构。

在项目根目录运行：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available()"
python -m pip install -e .
python train_lakun.py --train-groups 12 --val-groups 6 --test-groups 6 --epochs 1 --out runs/lakun_pilot
python train_lakun.py
```

第一行确认当前向带CUDA的PyTorch 环境
第二行安装本目录的LaKun包和其他依赖；
如果环境中已有满足要求的 PyTorch，就不需要另行安装。
第三行是小样本真权重试跑，确认显存与速度；第四行是完整训练。默认最多 3 轮，每 3000 个优化步验证一次，并在每轮结束时验证；验证损失改善至少 0.001 才更新最佳权重，连续 3 次无改善就早停。
可用 `--epochs`、`--eval-every`、`--patience`、`--min-delta` 调整。训练期间不看测试集，最后只对最佳权重测一次。`train_lakun.py` 自动使用当前机器可见的全部 GPU，单卡直接运行
多卡自动使用 DDP；
`CUDA_VISIBLE_DEVICES`。正式训练采用 BF16；


## 权重保存与下载

全量训练默认写入服务器项目目录 `runs/lakun_full/`，其中最佳完整模型权重在 `runs/lakun_full/best/`。
当前设置单文件上限 10GB，正常会得到一个 `best/lakun.safetensors`；`best/` 内还含文本／图像配置、tokenizer、图像预处理器及模型卡。
训练报告 `report.json` 和验证轨迹 `validation_history.json` 在上级目录。
训练结束后，推理至少需要下载**整个 `runs/lakun_full/best/` 文件夹**，不要只下载某个 `.safetensors`；如需保留训练记录，再下载同级的 `report.json` 和 `validation_history.json`。
小样本试跑输出在 `runs/lakun_pilot/`，不是正式模型。
可以用 `--out` 指定其他输出目录。最佳检查点是包括两座编码器和决策头的完整权重，推理不再需要额外读取原始两份底座；原始底座建议留下作重新训练与复现实验。

目前保存的是最佳模型的完整推理权重，不含优化器状态；如果训练中断，最佳权重可以推理，但当前脚本不能从中断位置无损续训。重新启动默认从两份原始底座开始，务必使用新的 `--out`，避免覆盖之前的最佳权重。

载入正式检查点做一次测试：

```bash
python predict_lakun.py --checkpoint runs/lakun_full/best --input examples/text_input.json
python predict_lakun.py --checkpoint runs/lakun_full/best --modality image --group-index 0
```

第一条使用仓库内的纯文本输入示例，不需要本地测试集；第二条读取本地测试集与图片。若做自己的图片推理，可在输入 JSON 中添加 `image` 路径。

在自己的代码中推理：

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("runs/lakun_full/best")
result = model.predict(
    [{"type": "choice", "question": "图中是什么？", "criteria": ["猫", "狗", "汽车"]}],
    image="/path/to/your/image.jpg",
)
print(result["answers"])
```

### 训练样例

```json
{"id": "aokvqa__009c9461e876957167dcf50e:q0",
"group_id": "aokvqa__009c9461e876957167dcf50e",
"source": "aokvqa",
"split": "test",
"images": ["images/aokvqa__009c9461e876957167dcf50e.jpg"],
 "state": "", "type": "choice", "question": "图中男子佩戴的领带是什么颜色？",
"criteria": ["红色", "蓝色", "黑色", "黄色"], "selected_index": 1, "target": [0.0, 1.0, 0.0, 0.0],
"teacher_confidence": 1.0, "teacher_evidence": "图像中清晰可见一条深蓝色的领带位于衬衫外。",
"token_budget": {"text_tokens": 51, "head_tokens": 47, "state_tokens": 0, "visual_token_reserve": 64, "total_budgeted_tokens": 115,
 "tokenizer": "convaiinnovations/laya/tokenizer"},
"status": "pseudo_labeled", "label_source": {"kind": "vlm"}}


{"id": "text:5726192412ef1e1510121558:q2",
"group_id": "text:5726192412ef1e1510121558",
"source": "deepseek_synthetic", "split": "test",
"images": [],
"state": "调度员：仓储机器人电量低于20%时，是立即回充还是先完成当前拣货？
"路径Agent：立即回充。剩余电量支撑不到拣货完成，中途停摆会堵住通道。"
"调度员：但回充点只有两个，三台机器人同时去会排队。"
"路径Agent：可让电量最低的那台优先，其余两台继续拣货，等它充到50%再轮换。"
"调度员：这样整体吞吐量会降多少？\n路径Agent：模拟显示约降4%，但避免堵通道带来的延误约7%。"
"调度员：那就按你说的做。",
"type": "noul",
"question": "调度员最终否决了路径Agent的轮换方案。",
"criteria": ["false", "true"], "selected_index": 0, "target": [1.0, 0.0], "teacher_confidence": 0.95,
"teacher_evidence": "那就按你说的做", "teacher_evidence_is_quote": true,
"token_budget": {"text_tokens": 312, "head_tokens": 49, "state_tokens": 259, "visual_token_reserve": 0, "total_budgeted_tokens": 312,
"tokenizer": "convaiinnovations/laya/tokenizer"}, "status": "pseudo_labeled", "language": "简体中文", "domain": "多 Agent 协作", "label_source": {"kind": "llm" }}
```

