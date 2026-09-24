"""Run the packaged single-question CLI from this project directory.

Text:  python main.py --state "这笔订单已退款" --type noul --question "是否已退款？"
Image: python main.py --image photo.jpg --type choice --question "图中是什么？" --criteria 猫 狗 汽车
Add --checkpoint /path/to/checkpoint when it is not at runs/lakun_full/best.
"""

from lakun.cli import main


if __name__ == "__main__":
    main()
