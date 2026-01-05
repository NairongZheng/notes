- [git命令](#git命令)
  - [配置相关](#配置相关)
    - [换行符说明与配置](#换行符说明与配置)
  - [仓库相关](#仓库相关)
  - [分支相关](#分支相关)
  - [标签相关](#标签相关)
  - [子模块相关](#子模块相关)
  - [git lfs](#git-lfs)
  - [其他](#其他)
    - [commit规范](#commit规范)
    - [回退](#回退)
    - [merge和rebase](#merge和rebase)
    - [clone部分仓库](#clone部分仓库)
    - [git-filter-repo重写git历史](#git-filter-repo重写git历史)

# git命令

## 配置相关

**git log设置**

```shell
# vim ~/.bashrc，添加：
alias git-log='git log --pretty=oneline --all --graph --abbrev-commit
    # -all：显示所有分支
    # -pretty=oneline：将提交信息显示为一行
    # -abbrev-commit：是的输出的commit更简短
    # -graph：以图的形式显示
```

**文件权限与系统换行**


[忽略文件权限导致的diff](https://www.jianshu.com/p/3b8ba804c47b)：`git config [--global || --local] core.filemode false`

处理系统换行导致的diff：具体查看下一节[换行符说明与配置](#换行符说明与配置)

**配置查看与删除**

```shell
# 查看当前仓库配置
git config --local --list
# 查看全局配置
git config --global --list
# 查看某个特定配置
git config [--global || --local] <setting name such as core.fileMode>
# 查看所有配置（包括系统级、全局级和本地仓库级）
git config --list --show-origin
    # 这个命令会列出 所有级别的 Git 配置，并显示它们存储在哪个配置文件中。
    # 本地配置（仓库级）：存储在仓库中的 `.git/config` 文件中，优先级最高，会覆盖其他级别的设置。
    # 全局配置（用户级）：存储在用户主目录下的 `.gitconfig` 文件中，优先级低于本地配置，但高于系统配置。
    # 系统配置：存储在系统级别的配置文件中（例如 `/etc/gitconfig`），优先级最低。

# 删除某个配置项（单个最先匹配）
git config --local --unset <配置项>
# 删除某个配置项（所有）
git config --local --unset-all <配置项>
```

**配置免密登陆**

方法一：使用ssh的方式，可以参考[ssh的配置](./ssh.md)

方法二：使用token的方式：

```shell
# 都是在第一次使用的时候输入即可，后面就不用了
# token需要在对应网站的个人设置里面去获取，都是一次显示的，需要的话自己保存

# 用明文的方式存在 ~/.git-credentials （Windows、macOS、Linux通用）
git config --global credential.helper store

# 不用明文的话（使用明文存储就可以啦）
git config --global credential.helper osxkeychain    # macOS
git config --global credential.helper manager        # Windows
```


### 换行符说明与配置

最好是仓库中直接用`.gitattributes`文件来控制（该文件不止可以控制换行符，用处很广泛）

[参考链接1!!!](https://juejin.cn/post/6942320745494085669)

[参考链接2!!!](https://www.jianshu.com/p/fa4d5963b6c8)

[官方示例](https://github.com/gitattributes/gitattributes)

[huggingface示例](https://huggingface.co/datasets/damonzheng/AIR-MDSAR-Map/blob/main/.gitattributes)


```bash
* text=auto

# 确保脚本文件（如 Shell、Python、Perl 等）使用 LF
*.sh text eol=lf
*.py text eol=lf diff=python
*.cc text eol=lf
*.pl text eol=lf

# Git diff 语法高亮（也可以写在上面同一行，用空格分隔）
*.py diff=python
*.cc diff=cpp

# Windows 执行文件不转换换行符
*.bat text eol=crlf
*.cmd text eol=crlf

# 二进制文件保持原样，不转换换行符
*.jpg binary
*.png binary
*.zip binary
*.exe binary
*.dll binary
```


或者设置`core.autocrlf`：
1. 在`Linux/macOS`上：`git config [--global || --local] core.autocrlf input`
   1. 拉取代码时不修改文件中的换行符（保持仓库中的LF）。
   2. 提交时强制转换CRLF为LF，确保仓库内的文件始终为LF。
2. 在`Windows`上：`git config [--global || --local] core.autocrlf true`
   1. 拉取代码时将LF转换为CRLF（适应Windows）。
   2. 提交代码时自动转换CRLF为LF（保持仓库统一）。


## 仓库相关

**本地仓库**

```shell
# 初始化仓库。默认是`master`，现多用`main`
git init --initial-branch=<init_branch_name>
# 克隆仓库
git clone [-b branch_name] <remote_url> <file_name>
# 查看本地仓库所有记录
git reflog  # 记录了本地仓库中 HEAD 和分支的移动记录，包括提交、合并、分支创建和删除等操作
```

**远端仓库**

```shell
# 新建关联远端仓库。一般默认都叫`origin`
git remote add <another_remote_name> <another_remote_url>
# 重命名远端关联仓库。一般默认都叫`origin`
git remote rename <old_remote_name> <new_remote_name>
# 查看远端仓库
git remote -v
# 重新设置远端关联
git remote set-url <remote_name> <new_url>
# 删除远端仓库关联
git remote remove <remote_name>
```

## 分支相关

**查看分支与分支关联**

```shell
# 查看所有分支
git branch -a
# 查看远端所有分支
git branch -r
# 查看本地分支与远端分支的追踪关系（只能追踪一个，所以如果有多个远端仓库关联，也只显示一个）
git branch -vv
# 设置本地分支与远端分支关联追踪
git branch --set-upstream-to=<remote_name>/<remote_branch_name> <local_branch_name>
```

**获取分支**

```shell
# 拉取远端分支（会自动进行merge，建议先fetch）
git pull <remote_name> <remote-branch-name>:<local-branch-name>
# 更新追踪的远端分支信息
git fetch <remote_name> <remote_branch_name>
# 更新所有远端仓库的所有分支信息，并清理不再存在的分支
git fetch --all -p
```

**推送分支**

```shell
# 推送分支到指定远端仓库
git push <remote_name> <local_branch_name>
# 推送分支到所有远端仓库（好像不好使？）
git push --all
```

**删除分支**

```shell
# 删除本地分支
git branch -d <branch_name>
# 删除远端分支
git push <remote_name> --delete <branch_name>
```

**重命名分支**

```shell
# 重命名本地分支
git branch -m <old_branch_name> <new_branch_name>
# 重命名远端分支：先删除远端分支，重命名本地分支，再推送本地分支
```

**检出/切换分支**

```shell
# 检出到现有分支
git checkout <branch_name>
# 检出新的分支
git checkout -b <new_branch_name> <base_branch_or_tag_version>
```

**合并分支**

详细流程可以查看[rebase和merge应用示例](#rebase和merge应用示例)

```shell
git merge [--no-commit] <branch_name>
# 这里merge的时候，需要选择好分支，使用前面配置的`git-log`是一目了然的。
# 是合并更新下来的`origin/branch_name`（远端的）还是`branch_name`（本地的）。
# 如果多人开发的话，大概率远端跟本地是不同步的，且很有可能是有冲突的。具体怎么做还是看开发需求。
```

**恢复删除的分支**

```shell
# 查找要恢复的分支
git reflog
# 恢复分支
git checkout -b <branch-name> <reflog-hash>
```

## 标签相关

**创建附注标签**

```shell
# 从HEAD
git tag -a <tag_version> -m <"message">
# 从某个特定commit
git tag -a <tag_version> <commit_id> -m <"message">
```

**查看标签**

```shell
# 查看所有标签
git tag
# 查看标签信息
git show <tag_version>
```

**推送标签**

```shell
# 推送单个标签
git push <remote_name> <tag_version>
# 推送所有标签
git push <remote_name> --tags
```

**删除标签**

```shell
# 删除本地标签
git tag -d <tag_version>
# 删除远程标签（跟删除远端分支的命令一样，如果重名只会删除tag而不会删除分支，除非显式指定）
git push <remote_name> --delete <tag_version>
```

**检出标签**

```shell
# Git 标签不是分支，无法直接切换到标签进行开发，而是进入“分离 HEAD”状态
git checkout <tag_version>
# 基于某个标签创建新分支
git checkout -b <new_branch_name> <tag_version>
```

## 子模块相关

**添加子模块**

```bash
git submodule add <repository_url> [path]
    # repository_url：子模块的 Git 仓库地址。
    # path（可选）：子模块存放的目录，默认是和仓库同名的文件夹。
```

**初始化并拉取子模块**

```bash
# 1. 如果是第一次 clone 含有子模块的项目
git clone --recurse-submodules <repo_url>
# 2. 如果已经 clone 了项目，需要手动初始化和更新子模块
git submodule init  # 初始化 .git/config 中的子模块配置（只需一次）
git submodule update  # 检出父仓库记录的 commit（不会拉 remote 最新）
# 3. 建议统一用以下命令来确保递归拉取所有子模块及其嵌套依赖
git submodule update --init --recursive
```

**查看子模块状态**

```bash
git submodule status
# 会列出每个子模块当前所指向的 commit。
# 若子模块目录内容与父仓库记录不一致，会有 + 或 - 前缀，代表状态变动。
```

**提交并推送本地修改**

```bash
# step 1：到子模块文件夹修改并提交并推送
# step 2：到父仓库修改并提交并推送
# 若是多人协作开发应该先进行下面的《同步远端更新到本地》，再提交当前本地修改
```

**同步远端更新到本地**

情景1：本地子模块没有修改：

```bash
# 本地子模块有改动的话千万别用这种，不然比较麻烦。
# 1. fetch父仓库、merge父仓库
# 2. 处理完冲突之后执行：
git submodule update --init --recursive
    # 子模块没有本地修改，所以直接恢复到父仓库记录的 commit 就行；
    # 避免触发子模块合并逻辑，流程简单安全；
    # 可防止“误拉子模块远程 HEAD”导致和父仓库记录不一致。
```

情景2：本地子模块有修改：

```bash
# 1. fetch&merge 父仓库并解决冲突（先别提交，因为子模块还没有跟远端merge）
# 2. fetch 子模块（不会覆盖本地改动）
git submodule foreach git fetch <origin>
# 3. 进入有修改的子模块，手动 merge 并解决冲突
cd <path/to/submodule>
git merge <origin/main>
git commit -m <"Resolve submodule conflict">
git push
# 4. 回到父仓库，更新子模块引用
# 5. 推送所有更改（先子模块，再父仓库）
cd <path/to/submodule>
git push <origin> <main>
cd <path/to/repositories>
git push <origin> <your-branch>
```

情景3：管他三七二十一：

```bash
# 直接对父仓库跟所有子模块进行pull的merge
git pull --recurse-submodules
```

**删除子模块**

```bash
# Step 1: 取消关联
git submodule deinit -f <path/to/submodule>
# Step 2: 从 Git 中移除并删除文件
git rm -f <path/to/submodule>
# Step 3: 删除残留文件夹（如果还在）
rm -rf .git/modules/<path/to/submodule>
rm -rf <path/to/submodule>
# Step 4: 可选手动清理 .gitmodules 中的内容（若未自动移除）
vim .gitmodules
# Step 5: 提交更改
git commit -m "Remove submodule <name>"
```


## git lfs

注意！！！一旦用了 LFS，切换普通 Git 有点麻烦（要手动迁移）。
而且不同平台都有容量限制、带宽限制，需要收费。
超过这个额度后，就会收到错误提示，比如 403 Forbidden、This repository is over its data quota.

**安装及初始化**

```bash
# 安装
apt install git-lfs
# 初始化（进行一次即可在~/.gitconfig中添加配置）
git lfs install
# ...
```

**新git仓库**

新git仓库用lfs管理大文件，需要做以下设置

```bash
# 使用git lfs track命令配置需要用lfs管理的文件，如mp4（强烈建议用.gitattributes文件配置）
git lfs track "*.mp4"
```

**将原有仓库的大文件改用lfs管理**

原有仓库要改成lfs管理大文件比较麻烦，因为`.gitattributes`只会管理新增的文件，原来就有的大文件，需要手动修改记录

```bash
# 重写已经管理了的大文件，如jpg、png、npy、zip
git lfs migrate import --include="*.jpg,*.png,*.npy,*.zip"
# 查看是否修改成功（列出来的说明已被管理）
git lfs ls-files
```


## 其他

### commit规范

一般标准的 commit 信息分为三个部分：

```shell
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

1. type：类型。标识本次提交的类别。常用类型在下面列出来。
2. scope：范围（可选）。说明本次 commit 影响的模块或范围，如果没有特定的模块，可以省略。
3. subject：简短描述。尽量简短，不超过 50 个字符。
4. body：详细描述（可选）。分点说明修改的细节、逻辑或副作用。
5. footer：脚注（可选）。用于关联 issue、标记 BREAKING CHANGE 或添加 co-author。

示例：

```shell
feat(cart): 支持购物车批量删除

1. 用户可以一次性删除多个购物车商品
2. 同时修复了删除后数量显示错误的问题

Closes #234
```

**type类型**

| 类型     | 说明                                                   |
| -------- | ------------------------------------------------------ |
| feat     | 新功能（feature）                                      |
| fix      | 修复 bug                                               |
| docs     | 文档更新                                               |
| style    | 代码格式（空格、分号、缩进等，不影响功能）             |
| refactor | 重构（既不是新增功能，也不是修复 bug）                 |
| perf     | 性能优化                                               |
| test     | 测试相关                                               |
| build    | 构建系统或依赖更新                                     |
| ci       | 持续集成相关                                           |
| chore    | 杂务（不影响源代码的修改，例如修改配置文件、工具脚本） |
| revert   | 回滚某次提交                                           |

**Co-author（共同作者）**

当多人协作完成某个提交时，可以在 commit 信息的 footer 中添加共同作者：

```shell
feat(auth): 实现用户登录功能

1. 添加 JWT 认证
2. 实现登录接口
3. 添加单元测试

Co-authored-by: Name1 <email1@example.com>
Co-authored-by: Name2 <email2@example.com>
```

**命令行直接添加 Co-author：**

```shell
# 单个共同作者
git commit -m "feat: add new feature" -m "Co-authored-by: Name <email@example.com>"

# 多个共同作者
git commit -m "feat: add new feature" \
           -m "Co-authored-by: Name1 <email1@example.com>" \
           -m "Co-authored-by: Name2 <email2@example.com>"
```

**注意事项：**

1. Co-author 必须在 footer 部分，与 body 之间要有空行
2. 邮箱地址必须是该用户在 GitHub/GitLab 等平台注册的邮箱
3. GitHub 会将 Co-author 也计入该提交的贡献者
4. 格式必须严格遵守：`Co-authored-by: Name <email@example.com>`
   1. 也可以使用 github 提供的隐私邮箱格式：`Co-authored-by: username <username@users.noreply.github.com>`


### 回退

1. 放弃本地未提交的修改：
   1. 撤销**工作区**更改：`git checkout -- .`
   2. 撤销**工作区和暂存区**更改：`git reset --hard`
   3. 撤销某个文件的修改：`git checkout -- <filename>`
   4. 撤销**未追踪文件**：`git clean -fd`（不可恢复！可以先用`git chean -nd`查看有哪些会被删除）
2. 放弃`Git Merge`：
   1. 中止正在进行的合并：`git merge --abort`
   2. 如果`--abort`无效：`git reset --hard HEAD`
   3. 回退到`merge`之前的提交：`git reset --hard <之前的 commit ID>`

| 用法                        | 用途                                      | 是否保留代码改动 |
| --------------------------- | ----------------------------------------- | ---------------- |
| `git reset`                 | 取消 commit 或 add，但保留工作区修改      | ✅ 是             |
| `git reset --soft`          | 仅移动 HEAD，保留暂存区和工作区（最安全） | ✅ 是             |
| `git reset --mixed`（默认） | 移动 HEAD + 清除暂存区，但保留工作区      | ✅ 是             |
| `git reset --hard`          | 完全回退 HEAD + 暂存区 + 工作区（危险）   | ❌ 否             |


### merge和rebase

查看[git_usage](../install_related/git_usage.md)

**git diff**

| 命令                           | 对比范围          | 用途                                         |
| ------------------------------ | ----------------- | -------------------------------------------- |
| `git diff`                     | 工作区 ↔ 暂存区   | 查看还没 git add 的改动                      |
| `git diff --cached`            | 暂存区 ↔ 最新提交 | 查看准备提交的内容                           |
| `git diff HEAD`                | 工作区 ↔ 最新提交 | 查看当前所有未提交的改动（含未 add）         |
| `git diff <commit1> <commit2>` | 两个提交之间      | 对比任意两次提交                             |
| `git diff <branch1> <branch2>` | 两个分支之间      | 比较两个分支的差异                           |
| `git diff --stat`              | 显示变更统计      | 显示修改文件数量、增删行数（不显示具体代码） |

### clone部分仓库

有时候仓库太大，只想要其中的某个文件夹或者某个文件，可以参考下面的做法：

```shell
git clone --filter=blob:none --sparse <repo_rul>
cd <repo_name>
git sparse-checkout init [--cone || --no-cone]
git sparse-checkout set <folder_or_filepath> # 文件用 --no-cone, 文件夹用 --cone
```

当然如果只要看文件而不需要提交记录等，直接用 `curl` 或者 `wget` 是最简单的。比如：

```shell
# 具体的 url 可以查看该文件在 github 页面右上角的 "Raw" 按钮
curl -O https://raw.githubusercontent.com/...
```

### git-filter-repo重写git历史

！！！危险操作！！！

下面用删除某文件举例说明 git-filter-repo 的大概使用方法

假设我们需要删除一些文件及它们的记录，那么就需要对整个仓库的提交进行更改。可以参考（注意：commit id 都会变！）：

```shell
# 安装
pip install git-filter-repo
# 删除某文件
git filter-repo --path <relative_path1> --path <relative_path2> --invert-paths
    # --invert-paths：删除这些文件（从所有历史中彻底移除）
# 有时候需要多 check 两遍是否真的删除了，然后重新 add remote，再 push

# 由于 hash 都变了，所以别人同步的时候需要重新覆盖本地
git fetch origin
git reset --hard origin/main
```