# LaKun 的 PyPI 发布清单

当前状态（2026-09-24）：**只做本地构建与验证，不上传 PyPI / TestPyPI。**PyPI 公开发布会把 wheel 和源码包向任何人开放；即使 GitHub 和魔塔仓库仍是 Private，别人也能从源码包获得其中的代码。模型权重不在安装包内。

## 第 1 步：检查项目

- 包名：`lakun`。2026-09-24 查询 `https://pypi.org/pypi/lakun/json` 返回 404，说明当时没有该项目；这不是永久保留，真正上传时仍可能发现名称被占用。
- 待发布版本：`0.1.0`，在 `pyproject.toml` 中配置。PyPI 不允许覆盖已经上传的同一版本文件；发布后的修正必须改为新版本。
- `pyproject.toml` 的许可是**源码的 Apache-2.0**，不授予模型权重的使用或再分发许可。
- `PYPI_README.md` 是 PyPI 页面说明，明确写了权重仍私有；GitHub 的中英文 README 是另一份文档。

## 第 2 步：本地构建与校验

在项目根目录执行：

```powershell
python -m pip install build twine
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/lakun-0.1.0-py3-none-any.whl dist/lakun-0.1.0.tar.gz
```

构建结果应仅包含源码和元数据：`dist/lakun-0.1.0-py3-none-any.whl`、`dist/lakun-0.1.0.tar.gz`。确认其中没有 `dataset/`、`images/`、`runs/`、`weights/`、密钥和本地绝对路径。当前 wheel 提供 `lakun` 命令，`main.py` 则供从源码运行。

## 第 3 步：像用户那样安装验证

在新的虚拟环境中安装本地 wheel，最好从项目目录外运行命令，避免误用源码目录中的 `lakun/`：

```powershell
python -m venv .venv-release
.venv-release\Scripts\python.exe -m pip install --no-deps dist/lakun-0.1.0-py3-none-any.whl
.venv-release\Scripts\lakun.exe --help
```

若要实际推理，虚拟环境还需要与本机匹配的 PyTorch 和本包依赖，以及完整的本地权重目录。可传 `--checkpoint` 指向权重。单独安装 wheel **不会**下载私有权重。

## 第 4 步：决定是否公开

在确认项目名、作者名称、公开代码范围、权重访问方式和权重许可后，才上传 PyPI。这一步需要项目拥有者的明确确认。PyPI 账号和 API Token 由拥有者在 [PyPI](https://pypi.org/) 创建；**不要把 Token 发到聊天中、写入仓库或截图公开**。

正式上传时使用官方推荐的 Twine：

```powershell
python -m twine upload dist/lakun-0.1.0-py3-none-any.whl dist/lakun-0.1.0.tar.gz
```

按终端提示输入用户名 `__token__` 和 PyPI API Token（隐藏输入）。只有看到上传成功，且 `https://pypi.org/project/lakun/0.1.0/` 可访问后，才能把 README 中“尚未发布”的说明改为 `pip install lakun`。TestPyPI 也是公开服务，不是私有预检空间。

上述构建与上传流程参考 [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)；不要在一次预检后跳过隔离安装验证。
