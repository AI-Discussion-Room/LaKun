# LaKun

[English](README.md) · [简体中文](README.zh-CN.md)

LaKun 是一个给候选选项打分的多模态模型，支持三种决策任务：**Choice**（选择）、**Score**（分档评分）和 **Noul**（`false`/`true` 判别）。输入可以是一段文本状态，也可以是一张图片加若干问题；输出是每道题各候选项的概率。它**不是**自由生成文本的聊天模型。

模型联合微调 [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) 文本编码器和 [SigLIP SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) 视觉编码器，通过视觉 token 桥接到受 Laya 启发的类型化决策头。项目不加载 Laya 微调权重；相关派生源码和许可证保留在 `lakun/vendor/laya/`。

> **当前状态：**首轮正式训练已结束，选用优化步 16,882 的检查点。留出测试集对**教师伪标签**的一致率分别为：选择题 83.11%、评分题 83.37%、判别题 80.92%；这不是人工真值准确率，也不能代表通用任务表现。一次探索性的外部垃圾评论分类测试显示跨领域迁移较弱，部署前请用自己的任务独立评测。

## 数据集概况

当前本地数据集共有 **126,663 组、380,424 道问题**：图像 59,996 组 / 180,424 题，文本 66,667 组 / 200,000 题。训练、验证、测试约按 8:1:1 **以组为单位**划分；同组的问题不会拆开。选择、评分和判别三类任务的题量接近。图像涵盖 7 个来源子集，合成文本覆盖 50 个方向。

答案属于模型生成的**伪标签**，不是人工核验的真值。[数据集统计与图表](analysis/DATASET_PROFILE.md)列出了具体频数、来源和已发现的答案位置偏斜。未做独立核验前，训练或评测成绩只能理解为与这些标签的一致程度。默认 Git 仓库不包含图片和数据文件；发布数据前还需核对原始数据集的许可条款。

## 项目结构

| 路径 | 用途 |
|---|---|
| `train_lakun.py` | 直接启动单卡或本机多卡训练 |
| `predict_lakun.py` | 用输入 JSON 推理，或查看一组留出集样本 |
| `lakun/` | 模型、数据读取、训练与推理代码 |
| `analysis/` | 仅包含数据集统计与图表 |
| `dataset/`、`images/` | 本地图文数据及引用的图片 |
| `weights/` | 启动训练所需的原始 mmBERT、SigLIP 权重 |
| `runs/lakun_full/best/` | 训练完成后的最佳完整推理检查点 |

## 安装与训练

需要 Python 3.10+ 和支持 CUDA 的 PyTorch。若服务器镜像已经装好 PyTorch，在项目根目录运行：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available()"
python -m pip install -r requirements-train.txt
python train_lakun.py --train-groups 12 --val-groups 6 --test-groups 6 --epochs 1 --out runs/lakun_pilot
python train_lakun.py
```

把原始模型放在 `weights/mmbert-base/` 和 `weights/siglip-so400m-patch14-384/`；保留六份 `dataset/{image,text}_{train,val,test}.jsonl` 及其引用图片的相对目录结构。`train_lakun.py` 默认使用本机全部可见 GPU，多卡时自动启动 PyTorch DDP。可用 `--gpus N` 指定卡数，或用 `CUDA_VISIBLE_DEVICES` 选择设备；不同服务器上的卡不能直接当作同一台机器的本地多卡。

全量训练默认最多 3 轮，每 3,000 个优化步做一次验证；验证损失连续 3 次没有达到改进阈值就早停。测试集只在选好模型后用于最终评估。默认输出目录为 `runs/lakun_full/`，其中 `runs/lakun_full/best/` 是**完整**的最佳推理检查点。推理时下载整个 `best/` 目录，不要只取 `.safetensors`。其中已经包含两座编码器和决策模块，推理不再单独读取原始底座权重。目前检查点不含优化器状态，训练中断后不能从原步数无损续训。

## 推理

拿到完整检查点后，可直接运行仓库里的纯文本示例：

```bash
python predict_lakun.py --checkpoint runs/lakun_full/best --input examples/text_input.json
```

修改 JSON 可输入自己的文本和问题；图片任务在 JSON 中增加 `image` 路径，相对路径按 JSON 文件所在目录解析。GitHub 代码仓库不包含检查点。

本地已有测试集时，可查看留出集样本：

```bash
python predict_lakun.py --checkpoint /path/to/your/checkpoint --modality image --group-index 0
```

也可以直接调用库，传自己的图片和问题（先替换图片路径）：

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("/path/to/your/checkpoint")
result = model.predict(
    [{"type": "choice", "question": "图中是什么？", "criteria": ["猫", "狗", "汽车"]}],
    image="/path/to/your/image.jpg",
)
print(result["answers"])
```

纯文本任务不传 `image`，改为提供 `state` 字符串。当前一组只支持**一张图片**、1–20 个问题；尚不支持多图输入。`noul` 判别题的候选项固定为 `criteria=["false", "true"]`。

## 发布与许可

[GitHub 源码仓库](https://github.com/AI-Discussion-Room/LaKun)、将来的 PyPI 安装包和模型权重仓库是三个独立的发布物。GitHub 仓库公开后，可用 `python -m pip install "git+https://github.com/AI-Discussion-Room/LaKun.git"` 安装代码；只有另行发布到 PyPI 后，`pip install lakun` 才会生效。安装包不包含权重和训练数据。

在魔塔创建模型仓库时，把 `runs/lakun_full/best/` **里面的内容**上传到仓库根目录，不要额外套一层 `best/` 或 `runs/`。根目录须有 `model_config.json`、`lakun.safetensors`、`text_encoder/`、`vision_encoder/`、`tokenizer/`、`image_processor/`。`lakun.safetensors` 文件名不要改，当前加载器会查找这个名字；公开仓库可命名为 `LaKun-0.76B-v1`。

魔塔下载需另外安装 `modelscope-hub`。当前 LaKun 加载器接受本地目录或 Hugging Face 仓库 ID，**不能**直接输入魔塔仓库 ID：

```python
from modelscope_hub import HubApi
from lakun import LaKunPredictor

checkpoint = HubApi().download_repo("你的魔塔账号/LaKun-0.76B-v1", "model")
model = LaKunPredictor.from_pretrained(checkpoint)
```

LaKun 源码采用 Apache-2.0，见根目录 `LICENSE`；复制的 Laya 组件保留其自己的 `lakun/vendor/laya/LICENSE`。以后公开权重时，还需在模型卡单独写明权重许可，并核对原始模型和数据条款。**不要**把整个项目、原始底座、数据集、图片、API Key 或外层 `runs/` 目录上传到模型仓库。这个自定义架构由 `lakun` 加载，暂不能直接用 Transformers 的 `AutoModel.from_pretrained()` 读取。

更详细的中文启动说明见 [START.md](START.md)。
