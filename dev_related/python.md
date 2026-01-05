- [命令操作](#命令操作)
- [pip包管理](#pip包管理)
- [代码质量工具](#代码质量工具)
- [项目管理](#项目管理)

## 命令操作
1. 打包：`pyinstaller --onefile --noconsole <py_file_path>`
2. proto编译：`cd <proto_file_path>`&&`python -m grpc_tools.protoc -I./ --python_out=./ --grpc_python_out=./ ./<proto_file_name.proto>`（`pip install grpcio-tools==1.47.0`这个版本编译完有类型声明，新版本没有。新版本好像高效一点，但是可读性差一些）
3. 生成代码关系结构：`pip install pylint`&&`sudo apt-get install graphviz`&&`pyreverse -o png -p <pic_name> <folder_or_file_path>`

## pip包管理

**基础命令**

```bash
# 安装包
pip install <package_name>
pip install <package_name>==<version>              # 安装指定版本
pip install <package_name>>=<version>              # 安装大于等于指定版本
pip install -r requirements.txt                    # 从文件安装
pip install -e .                                   # 以可编辑模式安装当前项目

# 卸载包
pip uninstall <package_name>
pip uninstall -r requirements.txt                  # 批量卸载

# 查看包信息
pip show <package_name>                            # 查看包详细信息
pip list                                           # 列出所有已安装的包
pip list --outdated                                # 列出所有过时的包
pip freeze                                         # 输出已安装包列表（带版本号）
pip freeze > requirements.txt                      # 导出当前环境依赖

# 更新包
pip install --upgrade <package_name>               # 更新指定包
pip install --upgrade pip                          # 更新pip自身

# 搜索包
pip search <keyword>                               # 搜索包（注意：PyPI已禁用此功能）

# 检查依赖
pip check                                          # 检查已安装包的依赖是否满足
```

**配置国内镜像源**

临时使用：

```bash
pip install <package_name> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

永久配置（Linux/macOS）：

```bash
# 创建配置文件
mkdir -p ~/.pip
vim ~/.pip/pip.conf

# 添加以下内容
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

永久配置（Windows）：

```bash
# 创建配置文件：C:\Users\<用户名>\pip\pip.ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```

常用国内镜像源：

```bash
# 清华：https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里云：https://mirrors.aliyun.com/pypi/simple/
# 中科大：https://pypi.mirrors.ustc.edu.cn/simple/
# 豆瓣：http://pypi.douban.com/simple/
```

## 代码质量工具

**代码格式化**

```bash
# black - 自动格式化代码（推荐）
pip install black
black <file_or_directory>
black --check <file_or_directory>                  # 检查但不修改
black --diff <file_or_directory>                   # 显示会做的修改

# isort - 自动排序import语句
pip install isort
isort <file_or_directory>
isort --check-only <file_or_directory>             # 仅检查

# autopep8 - 自动修复PEP8问题
pip install autopep8
autopep8 --in-place --aggressive <file>
autopep8 --in-place --aggressive --recursive <directory>
```

**代码检查（Linting）**

```bash
# pylint - 代码静态分析
pip install pylint
pylint <file_or_module>
pylint --rcfile=.pylintrc <file>                   # 使用配置文件

# flake8 - 轻量级代码检查
pip install flake8
flake8 <file_or_directory>
flake8 --max-line-length=120 <file>                # 自定义行长度

# mypy - 静态类型检查
pip install mypy
mypy <file_or_directory>
mypy --ignore-missing-imports <file>               # 忽略缺失的类型提示
```

**代码复杂度分析**

```bash
# radon - 代码复杂度分析
pip install radon
radon cc <file_or_directory>                       # 圈复杂度
radon mi <file_or_directory>                       # 可维护性指数
radon raw <file_or_directory>                      # 原始度量
```

**测试工具**

```bash
# pytest - 测试框架（推荐）
pip install pytest
pytest                                             # 运行所有测试
pytest <test_file>                                 # 运行指定文件
pytest -v                                          # 详细输出
pytest -k <keyword>                                # 运行匹配关键字的测试
pytest --cov=<module>                              # 代码覆盖率（需安装pytest-cov）

# unittest - Python自带测试框架
python -m unittest discover                        # 自动发现并运行测试
python -m unittest <test_module>                   # 运行指定模块

# coverage - 测试覆盖率
pip install coverage
coverage run -m pytest                             # 运行测试并收集覆盖率
coverage report                                    # 显示覆盖率报告
coverage html                                      # 生成HTML报告
```

## 项目管理

**requirements.txt 管理**

```bash
# 生成requirements.txt
pip freeze > requirements.txt

# 自动生成（只包含项目直接依赖，推荐）
pip install pipreqs
pipreqs . --force                                  # 扫描项目生成requirements.txt

# 安装依赖
pip install -r requirements.txt

# 更新所有包到最新版本
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U
```

**pyproject.toml（现代Python项目配置）**

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "0.1.0"
description = "A sample Python package"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "requests>=2.28.0",
    "numpy>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "mypy>=0.950",
]

[tool.black]
line-length = 120
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 120

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**setup.py（传统方式，仍然常用）**

```python
from setuptools import setup, find_packages

setup(
    name="my_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
        ],
    },
    python_requires=">=3.8",
)
```

**常用项目结构**

```
my_project/
├── .gitignore
├── README.md
├── requirements.txt          # 或 pyproject.toml
├── setup.py                  # 可选
├── my_package/
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
├── tests/
│   ├── __init__.py
│   ├── test_module1.py
│   └── test_module2.py
└── docs/
    └── ...
```

**快速创建项目骨架**

```bash
# 使用cookiecutter创建项目模板
pip install cookiecutter
cookiecutter https://github.com/audreyr/cookiecutter-pypackage
```