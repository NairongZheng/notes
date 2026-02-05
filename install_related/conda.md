- [conda 常用命令](#conda-常用命令)
- [conda 下载与配置](#conda-下载与配置)
- [环境迁移](#环境迁移)
  - [跨机器迁移](#跨机器迁移)

# conda 常用命令

```shell
# 创建环境
conda create -n <env_name> python=<python_version> -y [--dry-run]
# 克隆环境（-v 可以查看过程）
conda create -n <new_env_name> --clone <old_env_name/or/old_env_path> -v
# 查看环境
conda info -e
# 删除环境
conda remove -n <env_name> --all
```

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

# 正常来说经过上面的步骤就不需要下面这个了（这是老版的安装方式了）
# echo "export PATH=~/miniconda3/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc
```

**source 与 activate**

| 命令                                          | 作用                                                                       | 是否激活环境 | 是否改 PATH     | 是否推荐   | 备注 / 使用建议                |
| --------------------------------------------- | -------------------------------------------------------------------------- | ------------ | --------------- | ---------- | ------------------------------ |
| source activate ${CONDA_ROOT}/envs/<env_name> | 旧版激活。执行 activate 脚本激活指定 env                                   | ✅            | ✅               | ❌ 已废弃   | 老 Conda 写法，不兼容新机制    |
| source ${CONDA_ROOT}/bin/activate             | 初始化 + 激活（混合）。执行旧版 activate 脚本，直接修改 PATH、CONDA_PREFIX | ✅            | ✅               | ❌ 不推荐   | 老用法，历史遗留，容易污染环境 |
| source ${CONDA_ROOT}/etc/profile.d/conda.sh   | 初始化。向当前 shell 注入 conda 函数和 hook                                | ❌            | ❌（不指向 env） | ✅ 强烈推荐 | 现代 Conda 的第一步            |
| conda activate conda_env                      | 激活（现代）。通过 shell 函数切换 env，清理旧状态                          | ✅            | ✅               | ✅ 唯一推荐 | 必须先 source conda.sh         |

需要使用多个人多个环境的时候，建议直接指定 python 的路径，不要激活，因为同一个脚本多次切换 conda 环境的话， conda deactivate 并不彻底

```shell
# 不推荐
source /user1/conda_root/etc/profile.d/conda.sh
conda activate user1_env1
python script1
source /user2/conda_root/etc/profile.d/conda.sh
conda activate user2_env2
python script2
# 推荐
/user1/conda_root/envs/user1_env1/bin/python script1
/user2/conda_root/envs/user2_env2/bin/python script2
# 当然，使用上面表格中的其他方式更不推荐！
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
