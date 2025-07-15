- [git命令](#git命令)
  - [配置相关](#配置相关)
    - [换行符说明与配置](#换行符说明与配置)
  - [仓库相关](#仓库相关)
  - [分支相关](#分支相关)
  - [标签相关](#标签相关)
  - [子模块相关](#子模块相关)
  - [git lfs](#git-lfs)
  - [其他操作](#其他操作)

# git命令

## 配置相关

1. git log显示格式：在`~/.bashrc`中配置`git-log`：`alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'`
   1. `-all`：显示所有分支
   2. `-pretty=oneline`：将提交信息显示为一行
   3. `-abbrev-commit`：是的输出的commit更简短
   4. `-graph`：以图的形式显示
2. 配置：
   1. [忽略文件权限导致的diff](https://www.jianshu.com/p/3b8ba804c47b)：`git config [--global || --local] core.filemode false`
   2. [处理系统换行导致的diff](#换行符说明与配置)
3. 查看配置：
   1. 查看当前仓库配置：`git config --local --list`
   2. 查看全局配置：`git config --global --list`
   3. 查看某个特定配置：`git config [--global || --local] <setting name such as core.fileMode>`
   4. 查看所有配置（包括系统级、全局级和本地仓库级）：`git config --list --show-origin`
      1. 这个命令会列出 所有级别的 Git 配置，并显示它们存储在哪个配置文件中。
      2. 本地配置（仓库级）：存储在仓库中的 `.git/config` 文件中，优先级最高，会覆盖其他级别的设置。
      3. 全局配置（用户级）：存储在用户主目录下的 `.gitconfig` 文件中，优先级低于本地配置，但高于系统配置。
      4. 系统配置：存储在系统级别的配置文件中（例如 `/etc/gitconfig`），优先级最低。
4. 删除配置：
   1. 删除某个配置项（单个最先匹配）：`git config --local --unset <配置项>`
   2. 删除某个配置项（所有）：`git config --local --unset-all <配置项>`


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

1. 初始化仓库：`git init --initial-branch=<init_branch_name>`（默认是`master`，现多用`main`）
2. 新建关联远端仓库：`git remote add <another_remote_name> <another_remote_url>`（一般默认都叫`origin`）
3. 查看远端仓库：`git remote -v`
4. 重命名仓库：`git remote rename <old_remote_name> <new_remote_name>`（一般默认都叫`origin`）
5. 重新设置远端关联：`git remote set-url <remote_name> <new_url>`
6. 删除远端仓库关联：`git remote remove <remote_name>`
7. 查看本地仓库所有记录：`git reflog`（记录了本地仓库中 HEAD 和分支的移动记录，包括提交、合并、分支创建和删除等操作）
8. 克隆仓库：`git clone [-b branch_name] <remote_url> <file_name>`

## 分支相关

1. 查看分支：
   1. 查看所有分支：`git branch -a`
   2. 查看远端所有分支：`git branch -r`
   3. 查看本地分支与远端分支的追踪关系：`git branch -vv`（只能追踪一个，所以如果有多个远端仓库关联，也只显示一个）
2. 设置本地分支与远端分支关联追踪：`git branch --set-upstream-to=<remote_name>/<remote_branch_name> <local_branch_name>`
3. 推送分支：
   1. 推送分支到指定远端仓库：`git push <remote_name> <local_branch_name>`
   2. 推送分支到所有远端仓库：`git push --all`（好像不好使？）
4. 重命名分支：
   1. 重命名本地分支：`git branch -m <old_branch_name> <new_branch_name>`
   2. 重命名远端分支：先删除远端分支，重命名本地分支，再推送本地分支
5. 删除分支：
   1. 删除本地分支：`git branch -d <branch_name>`
   2. 删除远端分支：`git push <remote_name> --delete <branch_name>`
6. 检出/切换分支：
   1. 检出到现有分支：`git checkout <branch_name>`
   2. 检出新的分支：`git checkout -b <new_branch_name> <base_branch_or_tag_version>`
7. 获取分支：
   1. 拉取远端分支：`git pull <remote_name> <remote-branch-name>:<local-branch-name>`
   2. 更新追踪的远端分支信息：`git fetch <remote_name> <remote_branch_name>`
   3. 更新所有远端仓库的所有分支信息：`git fetch --all`
8. 合并分支：
   1. `git merge [--no-commit] <branch_name>`
   2. 这里merge的时候，需要选择好分支，使用前面配置的`git-log`是一目了然的。是合并更新下来的`origin/branch_name`（远端的）还是`branch_name`（本地的），如果多人开发的话，大概率远端跟本地是不同步的，且很有可能是有冲突的。具体怎么做还是看开发需求。
   3. 详细流程可以查看[这里](#rebase和merge应用示例)
9. 恢复删除的分支：
   1. 用`git reflog`查找要恢复的分支的hash
   2. 然后`git checkout -b <branch-name> <reflog-hash>`
10. 回退：
    1. 回退并保留更改：`git reset <hash>`
    2. 回退并清空暂存区和工作区：`git reset --hard <hash>`


## 标签相关

1. 创建附注标签：
   1. 从HEAD：`git tag -a <tag_version> -m <"message">`
   2. 从某个特定commit：`git tag -a <tag_version> <commit_id> -m <"message">`
2. 查看标签：
   1. 查看所有标签：`git tag`
   2. 查看标签信息：`git show <tag_version>`
3. 推送标签：
   1. 推送单个标签：`git push <remote_name> <tag_version>`
   2. 推送所有标签：`git push <remote_name> --tags`
4. 获取标签：
   1. 更新远端标签到本地：`git fetch --tags`
5. 删除标签：
   1. 删除本地标签：`git tag -d <tag_version>`
   2. 删除远程标签：`git push <remote_name> --delete <tag_version>`
6. 检出标签：
   1. Git 标签不是分支，无法直接切换到标签进行开发，而是进入“分离 HEAD”状态：`git checkout <tag_version>`
   2. 基于某个标签创建新分支：`git checkout -b <new_branch_name> <tag_version>`


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


## 其他操作

**回退**

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


**merge和rebase**

查看[git_usage](../install_related/git_usage.md)