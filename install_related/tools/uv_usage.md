

```shell
# uv 的理念是：所有项目都应该有自己独立、可复现的环境
# 也就是说，它会：
# 用你指定的 /User/.../python 作为基础解释器
# 然后在项目目录下建立一个 .venv
# 所有包都装进这个 .venv/lib/pythonX.X/site-packages
# 并生成 uv.lock 记录版本


# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 使用uv（从空项目开始

# 1. 初始化，下面的命令会生成 .python-version main.py pyproject.toml README.md等文件
uv init

# 2. (可选)修改基础环境，把 .python-version 里面内容进行修改
uv python list    # 可以查看有哪些python可以用（其中当前的conda环境的python可会显示）
# 给版本号的话，若没有会自动下载
# 给绝对路径（如conda的环境）的话会直接使用
# 也可以使用 uv python pin ${python_path} 这个命令进行修改

# 3. 查看python版本（之后使用uv run都是用的这个解释器）
uv run python -V        # 查看版本
uv run which python     # 查看具体python路径

# 4. 查看依赖
uv pip list            # 这个跟pip list是一样的，查看的是当前的conda的python的依赖
uv tree [--depth=1]    # 查看该项目的uv有哪些依赖

# 5. 添加依赖（写入pyproject.toml文件，并且会自己安装到uv虚拟环境中，不会污染conda的环境）
uv add fastapi    # 移除依赖用 uv remove fastapi

# 6. 生成锁文件uv.lock
uv lock

# 7. 同步安装依赖（根据锁文件）
uv sync

# 8. 运行
uv run main.py    # 如果使用python main.py会使用conda环境的依赖，用不到uv管理的依赖

# 9. 激活uv依赖
# 如果想要python main.py可以运行的话，可以手动激活uv的依赖
source ./.venv/bin/activate    # 这样就可以使用python直接运行了

# 10. 退出激活的话直接
deactivate
```