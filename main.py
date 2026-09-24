"""Single-question inference: python main.py prompts for a model ID or checkpoint.

# 单图示例：运行 python main.py，输入 hh108801/LaKun-0.7B、图片路径、题型和问题。
# 纯文本示例：图片路径直接回车，然后输入文本状态。
"""

from lakun.cli import main


if __name__ == "__main__":
    main()
