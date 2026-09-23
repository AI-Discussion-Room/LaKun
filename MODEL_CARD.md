---
tags:
- multimodal
- typed-decisions
- safetensors
base_model:
- jhu-clsp/mmBERT-base
- google/siglip-so400m-patch14-384
---

# LaKun-0.7B：文本与单图的类型化决策

我做 LaKun，是想把一段文本状态或一张图片，变成几个**事先定义好答案空间**的问题：选择、分档评分、二元判别。它对每题的候选项打分，返回最高分的候选项及整组 softmax 分数；它不生成自由文本，也不是聊天模型。三类问题可以放在一次 `predict()` 中一起问。

> **当前是私有预览。**这份权重已上传到 `hh108801/LaKun-0.7B`，但仓库尚未公开，只有获得访问权限的人能下载。我还没有发布 PyPI 包。源代码在私有仓库 [`AI-Discussion-Room/LaKun`](https://github.com/AI-Discussion-Room/LaKun)，采用 Apache-2.0；**权重许可尚未单独确定**，不应把源码许可证当成权重许可证。

## 模型档案

| 项目 | 本检查点 |
|---|---|
| 输入 | 文本状态，或单张图片；每组 1–20 道问题 |
| 类型 | `choice`（2–20 项）、`score`（3–20 档）、`noul`（`false/true`） |
| 输出 | 每题的 `criteria`、`probabilities` 和从 0 开始的 `predicted_index` |
| 文本/视觉主干 | [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) / [SigLIP SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) |
| 融合 | 64 个视觉查询 token + 类型化决策头，文本与视觉主干联合微调 |
| 参数量 | **756,409,158（约 0.756B）**；`0.7B` 是近似命名 |
| 文件大小 | `lakun.safetensors` **3,025,715,432 字节（2.82 GiB）**；完整推理目录约 **2.85 GiB** |
| 上下文 | 最多 512 token，图像预留 64 个视觉 token；超限报错 |
| 选择的训练步 | 16,882（按验证损失选出） |

我没有采用 Laya 已训练好的权重；只是复用了其部分决策头实现，并保留衍生代码的许可声明。这个检查点包含完整的文本编码器、视觉编码器、融合层与决策头；**不要只下载权重文件**，还需要同目录的 `model_config.json`、`tokenizer/`、`image_processor/`、`text_encoder/`、`vision_encoder/`。当前是 LaKun 自定义架构，不能直接用 Transformers 的 `AutoModel.from_pretrained()` 读取。

## 如何使用

我还没发布 `pip install lakun`。在有权限的环境里，先拉取源码并安装，然后下载完整魔塔仓库到本地：

```bash
git clone https://github.com/AI-Discussion-Room/LaKun.git
cd LaKun
python -m pip install -e .
python -m pip install modelscope-hub
ms-hub login
ms-hub download hh108801/LaKun-0.7B --local-dir LaKun-0.7B
```

私有 GitHub 仓库需要自己的访问权限。`ms-hub login` 在终端输入 Token，**不要把 Token 粘贴进代码或提交到仓库**。在项目目录里运行下面的示例：

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("LaKun-0.7B")
result = model.predict(
    [
        {"type": "choice", "question": "这是什么类型的请求？", "criteria": ["咨询", "投诉", "退款"]},
        {"type": "score", "question": "这件事有多紧急？", "criteria": ["不紧急", "一般", "紧急"]},
        {"type": "noul", "question": "用户是否要求退款？", "criteria": ["false", "true"]},
    ],
    state="用户说：我被重复扣费了，请尽快退款。",
)
for answer in result["answers"]:
    print(answer["type"], answer["predicted_index"], answer["probabilities"])
```

图片任务将 `state=` 换成 `image="/path/to/your/image.jpg"` 即可，目前**一组只支持一张图片**。`score` 返回的是分档索引，而不是连续回归数值。所有 `probabilities` 是 softmax 输出；我尚未验证校准程度，**不把它们宣传为真实正确概率**。

## 我做过的评测

我按组切分训练、验证、测试集；模型选择只使用验证损失。最佳验证损失为 `0.014464`。在 **12,703 组、38,141 题**的留出测试集上，最高分选项与 `qwen3.8-flash` 教师伪标签的逐题一致率是：

| 问题类型 | 留出集一致率 |
|---|---:|
| Choice（选择） | 83.11% |
| Score（分档） | 83.37% |
| Noul（二元） | 80.92% |

这**不是人工标注真值的准确率**，更不是通用能力或概率校准结论。第二轮训练损失继续下降时，验证损失反而升高，因此我最终保留了第一轮末的检查点，而没有用训练末尾的权重。

### 推理测速

我在**本机 Windows 11、RTX 3060 12GB、PyTorch 2.7.1+cu126、FP32**上对同一个完整检查点测速。每个场景预热 5 次、串行计时 30 次，CUDA 在每次计时前后同步；下表是整个 `predict()` 调用的墙钟时间，包含分词、图片读取与预处理、前向传播和输出整理，**不含加载权重、下载或网络请求**。图像是本地留出集的一张样本；不同机器、输入长度或并发下结果会变。

| 单次请求 | 题数 | 中位延迟 | p95 延迟 |
|---|---:|---:|---:|
| 文本 | 1 | **17.01 ms** | 19.23 ms |
| 文本 | 3 | **18.18 ms** | 18.65 ms |
| 单张图片 | 1 | **162.01 ms** | 168.69 ms |
| 单张图片 | 3 | **168.71 ms** | 171.59 ms |

可复测脚本与本次测量记录在[源码仓库的 `analysis/`](https://github.com/AI-Discussion-Room/LaKun/tree/main/analysis) 中（仓库当前为 Private）。这些是 RTX 3060 数字，**不是**先前训练使用的 4090 的速度，也不是吞吐量测试。

## 我的数据集：只公开统计，不公开样本

我把图像与文本都整理为“一组状态 + 多道类型化问题”。全部问题标签由 `qwen3.8-flash` 生成，是**伪标签**而不是人工核验答案。当前数据共有 **126,663 组 / 380,424 题**：图像 59,996 组 / 180,424 题，文本 66,667 组 / 200,000 题；按组约 8:1:1 划分，避免同组问题跨集合。

| 类型 | 图像题 | 文本题 | 总题数 |
|---|---:|---:|---:|
| Choice | 60,401 | 66,667 | 127,068 |
| Score | 60,015 | 66,667 | 126,682 |
| Noul | 60,008 | 66,666 | 126,674 |

![数据规模、题型与切分](assets/01-overview.png)

我还统计了候选项个数与答案位置。四选一题存在**明显位置偏差**：图像题的 B 项占 53.15%、D 项仅 1.74%；文本题 B 项占 46.48%、D 项仅 2.98%。因此模型有可能学到位置偏好。

![候选项数、评分档位和伪标签分布](assets/02-options-and-labels.png)

图像来自 7 个来源子集，合成文本涵盖 50 个方向。下面两张图记录来源、中文字符覆盖、长度预算和文本方向频数；图中文字为中文。

![图像来源、文本主题与覆盖](assets/03-coverage.png)

![50 个文本方向与四选一答案位置](assets/04-domains-and-choice-labels.png)

我没有在这里上传原始训练 JSONL、图片或之前检查过的 1,000 组测试明细。以上图表只有汇总统计；原始数据来源及再分发许可仍需逐项核查。

## 我目前看到的局限

LaKun 在**接近我训练数据分布**的选择、分档和二元题上已有可用表现。我在试用中感觉：带有明确数值线索、需要在给定选项之间做简单推理的题目，它往往更容易答好；**这是观察，不是已完成的独立分项基准测试**。

但我也观察到明显的外推边界：一次外部垃圾评论分类测试显示跨领域迁移较弱；第二轮训练损失继续下降而验证损失变差，让我担心过度专用化，甚至某种“微调坍塌”。目前**不能据此证明底座能力本身发生坍塌**；还需要与原始底座、不同训练检查点、人工标注域外集做严格对照。候选项位置偏斜、教师伪标签错误和未校准的 softmax 分数也都会放大误判风险。

因此我不会把本模型宣传成通用分类器、已验证的数值推理模型或强泛化视觉模型。它当前仅支持单图；在你自己的数据上完成评测、人工复核和概率校准之前，请勿把输出当作高风险决策依据。
