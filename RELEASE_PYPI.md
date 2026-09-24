# LaKun 的 PyPI 发布清单

我已在 2026-09-24 将 `lakun 0.1.0` 的 wheel 和源码包发布到 [PyPI](https://pypi.org/project/lakun/0.1.0/)。我核对了线上文件与本地文件的 SHA-256：wheel 为 `406452bc73a6dc6928425385f4fb6cb94b1a8446ecda9bae1035470ce90b923e`，源码包为 `84b182f2fef4921ea70b5c54531a0bc5dbf12e0459ca99299a73f669c69d6b6a`。我还从 PyPI 重新安装了 wheel，确认 `lakun --help` 正常，并用完整权重跑通一题文本推理。PyPI 向所有人开放代码发行文件，不取决于 GitHub 与魔塔仓库的访问设置；模型权重不在发行文件内。

## 第 1 步：检查项目

- 包名：`lakun`。2026-09-24 查询 `https://pypi.org/pypi/lakun/json` 返回 404，说明当时没有该项目；这不是永久保留，真正上传时仍可能发现名称被占用。
- 待发布版本：`0.1.0`，在 `pyproject.toml` 中配置。PyPI 不允许覆盖已经上传的同一版本文件；发布后的修正必须改为新版本。
- `pyproject.toml` 的许可是**源码的 Apache-2.0**，不授予模型权重的使用或再分发许可。
- `PYPI_README.md` 是我的 PyPI 页面说明，明确写了权重需从模型仓库另行获取；GitHub 的中英文 README 是另一份文档。

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

若要实际推理，虚拟环境还需要与本机匹配的 PyTorch 和本包依赖，以及完整的权重目录。交互运行 `lakun` 会询问该目录；单独安装 wheel **不会**下载权重。

## 第 4 步：上传与核对

我公开代码包，权重的获取与许可由模型仓库单独说明。上传前需要在 [PyPI](https://pypi.org/) 创建账号和 API Token。首次创建项目时，Token 需使用 Entire account 范围；项目创建后可改用仅限该项目的 Token。**我不会把 Token 发到聊天中、写入仓库或截图公开**。

正式上传时使用官方推荐的 Twine：

```powershell
python -m twine upload dist/lakun-0.1.0-py3-none-any.whl dist/lakun-0.1.0.tar.gz
```

按终端提示输入用户名 `__token__` 和 PyPI API Token（隐藏输入）。上传后打开 `https://pypi.org/project/lakun/0.1.0/`，并在一个新虚拟环境中试运行 `python -m pip install lakun` 与 `lakun --help`，确认发布文件可用。TestPyPI 也是公开服务，不是私有预检空间。

上述构建与上传流程参考 [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)；不要在一次预检后跳过隔离安装验证。
