
# conda 下载与配置

**conda 下载与安装**

使用miniconda，可以到[清华大学镜像网站](https://mirrors.tuna.tsinghua.edu.cn/)找相应的 [miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 版本

```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p ~/miniconda3
```

**初始化 conda**

```shell
# 有时候找不到 conda 命令，是因为没有 conda init
# 需要查看 ～/.bashrc 或者 ～/.zshrc 中是否有：
. <path_to_conda_install>/etc/profile.d/conda.sh
# 然后再运行：
conda init [zsh | bash]

# 正常来说经过上面的步骤就不需要下面这个了
# echo "export PATH=~/miniconda3/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc
```

**创建环境与安装包**

```shell
# 创建环境
conda create -n <env_name> python=<python_version> -y [--dry-run]
    # -n: 环境名称
    # python=: 该环境里 python 的版本
    # --dry-run: 不真正运行，只查看结果
# 若需修改 python 版本
# conda install python=<another_version>
# 激活环境
conda activate <env_name>
# 安装 uv （后续用这玩意安装包很快）
pip install uv
# 安装包
uv pip install <pkg_name>
```

**查看环境**

```shell
conda info -e
```

**删除环境**

```shell
conda remove -n <env_name> --all
```

# 部分包的安装

```shell

```

# 环境迁移

## 跨机器迁移

```shell
# 在旧机器
conda activate <env_name>
conda env export --from-history > environment.yml    # --from-history: 只导出手动装过的包，不会锁死平台细节（不会导出 pip 安装的）
pip freeze > requirements.txt

# 在新机器
conda env create -f environment.yml
conda activate <env_name>
pip install uv
uv pip install -r requirements.txt   # 用 uv 安装比较快
```
