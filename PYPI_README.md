# LaKun：JEV 决策类型模型的多模态版本

灵感来源于JEV，感谢laya https://github.com/mizorewww/laya-mlx 提供的优化方式（RLCD），所以取名文LaJ(谐音垃圾)，因致敬前辈故LaKun诞生
LaKun是决策类型模型的多模态版本
面对一段文本状态或一张图片，它直接回答我指定的选择、档位评分和二元判断问题，
输出每个候选项的softmax分数与最高分选项，不生成自由文本。一次调用最多可提交20道问题，目前每次最多处理一张图片。
这个PyPI安装包提供推理代码，源码包另含训练脚本；两种发行文件都**不包含权重或数据集**。我的完整检查点有 756,409,158 个参数；`0.7B` 是模型仓库名中的近似称呼。

## 安装与使用

先安装适合本机的 PyTorch，再安装 LaKun：

```bash
python -m pip install lakun
```

模型权重独立存放在[我的魔塔模型仓库](https://modelscope.cn/models/hh108801/LaKun-0.7B)，不随PyPI包安装；
获取权重后，保留包含 `model_config.json`、权重文件、编码器配置、tokenizer 和图像处理器的**完整目录**。直接运行下面的命令，程序会先询问该目录的位置，再逐项询问文本状态或本地图片路径、题型和问题：
完整代码见[我的个人仓库](https://github.com/AI-Discussion-Room/LaKun)。


```bash
lakun
```

输入图片路径即可问单图问题；纯文本任务直接跳过图片路径。选择题的候选项用 `|` 分隔。`--checkpoint` 接受下载后的完整本地目录，也接受 `hh108801/LaKun-0.7B` 魔塔仓库 ID 并自动下载。需要在脚本中使用时：

```python
from lakun import LaKunPredictor

model = LaKunPredictor.from_pretrained("hh108801/LaKun-0.7B")
result = model.predict(
    [{"type": "choice", "question": "这是什么类型的请求？", "criteria": ["咨询", "投诉", "退款"]}],
    state="用户说：我被重复扣费了，请尽快退款。",
)
print(result["answers"][0])
```

`choice` 接受 2–20 个候选项，`score` 接受 3–20 个有序档位，`noul` 固定为 `false/true`。`predicted_index` 从 0 开始。上下文最多 512 token；输入超限时，我的实现优先保留问题和候选项，再保留状态开头。。
