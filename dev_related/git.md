- [git命令](#git命令)
  - [配置相关](#配置相关)
  - [仓库相关](#仓库相关)
  - [分支相关](#分支相关)
- [rebase和merge应用示例](#rebase和merge应用示例)
  - [1. main分支和dev分支同时开发](#1-main分支和dev分支同时开发)
  - [2. 将dev分支自己的记录rebase](#2-将dev分支自己的记录rebase)
  - [3. 采用merge的方式合并（推荐）](#3-采用merge的方式合并推荐)
    - [3.1. 将main分支merge到dev分支](#31-将main分支merge到dev分支)
    - [3.2. 将dev分支merge到主分支](#32-将dev分支merge到主分支)
    - [3.3. 删除dev分支](#33-删除dev分支)
  - [4. 采用rebase的方式合并（不推荐）](#4-采用rebase的方式合并不推荐)
    - [4.1. 在dev上rebase到main](#41-在dev上rebase到main)
    - [4.2. 将dev分支merge到主分支](#42-将dev分支merge到主分支)
    - [4.3. 删除dev分支](#43-删除dev分支)
- [git入门（较乱的记录）](#git入门较乱的记录)
  - [获取本地仓库](#获取本地仓库)
    - [工作流程](#工作流程)
    - [查看提交日志](#查看提交日志)
    - [版本回退](#版本回退)
    - [是否添加文件至git管理](#是否添加文件至git管理)
  - [分支](#分支)
    - [解决冲突](#解决冲突)
    - [分支在开发中使用的流程](#分支在开发中使用的流程)
  - [git远程仓库](#git远程仓库)
    - [配置SSH公钥](#配置ssh公钥)
    - [操作远程仓库](#操作远程仓库)
    - [从远程仓库克隆](#从远程仓库克隆)
    - [从远程仓库抓取和拉取](#从远程仓库抓取和拉取)
    - [远程冲突解决](#远程冲突解决)
  - [铁令](#铁令)
  - [其他](#其他)
    - [git stash](#git-stash)
    - [git fetch](#git-fetch)


# git命令

## 配置相关

1. 在`~/.bashrc`中配置`git-log`：`alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'`
   1. `-all`：显示所有分支
   2. `-pretty=oneline`：将提交信息显示为一行
   3. `-abbrev-commit`：是的输出的commit更简短
   4. `-graph`：以图的形式显示
2. 查看当前仓库配置：`git config --local --list`
3. 查看全局配置：`git config --global --list`
4. 查看所有配置（包括系统级、全局级和本地仓库级）：`git config --list --show-origin`
   1. 这个命令会列出 所有级别的 Git 配置，并显示它们存储在哪个配置文件中。
   2. 本地配置（仓库级）：存储在仓库中的 `.git/config` 文件中，优先级最高，会覆盖其他级别的设置。
   3. 全局配置（用户级）：存储在用户主目录下的 `.gitconfig` 文件中，优先级低于本地配置，但高于系统配置。
   4. 系统配置：存储在系统级别的配置文件中（例如 `/etc/gitconfig`），优先级最低。
5. 查看某个特定配置：`git config [--global || --local] <setting name such as core.fileMode>`
6. 删除某个配置项（单个最先匹配）：`git config --local --unset <配置项>`
7. 删除某个配置项（所有）：`git config --local --unset-all <配置项>`
8. [忽略文件权限导致的diff](https://www.jianshu.com/p/3b8ba804c47b)：`git config [--global || --local] core.filemode false`
9. 忽略系统换行导致的diff：`git config [--global || --local] core.autocrlf true`


## 仓库相关

1. 查看远端仓库：`git remote -v`
2. 重命名仓库：`git remote rename <old_remote_name> <new_remote_name>`（一般默认都叫`origin`）
3. 新建关联远端仓库：`git remote add <another_remote_name> <another_remote_url>`
4. 删除远端仓库关联：`git remote remove <remote_name>`
5. 查看本地仓库所有记录：`git reflog`（记录了本地仓库中 HEAD 和分支的移动记录，包括提交、合并、分支创建和删除等操作）

## 分支相关

1. 查看远端所有分支：`git branch -r`
2. 查看所有分支：`git branch -a`
3. 查看本地分支与远端分支的追踪关系：`git branch -vv`（只能追踪一个，所以如果有多个远端仓库关联，也只显示一个）
4. 设置本地分支与远端分支关联追踪：`git branch --set-upstream-to=<remote_name>/<remote_branch_name> <local_branch_name>`
5. 删除本地分支：`git branch -d <branch_name>`
6. 删除远端分支：`git push <remote_name> --delete <branch_name>`
7. 重命名本地分支：`git branch -m <old_branch_name> <new_branch_name>`
8. 重命名远端分支：先删除远端分支，重命名本地分支，再推送本地分支
9. 合并分支：`git merge [--no-commit] <branch_name>`
10. 推送分支到指定远端仓库：`git push <remote_name> <local_branch_name>`
11. 推送分支到所有远端仓库：`git push --all`（好像不好使？）
12. 获取追踪的远端分支信息：`git fetch <remote_name> <remote_branch_name>`
13. 获取所有远端仓库的所有分支信息：`git fetch --all`
14. 拉取远端分支：`git pull <remote_name> <remote-branch-name>:<local-branch-name>`
15. 恢复删除的分支：
    1. 用`git reflog`查找要恢复的分支的hash
    2. 然后`git checkout -b <branch-name> <reflog-hash>`
16. 回退：
    1. 回退并保留更改：`git reset <hash>`
    2. 回退并清空暂存区和工作区：`git reset --hard <hash>`

# rebase和merge应用示例

1. 查看分支的公共祖先：`git merge-base <branch_1> <branch_2> [<branch_n>]`
2. 查看分支在某节点之后的提交次数：`git rev-list --count <commit-hash>..<branch_name>`

> 一般我用的方法是，在开发分支dev合并到主分支main时候：
> 在dev分支进行自己的rebase
> 在dev分支merge最新的main分支
> 在main分支merge解决冲突后的dev分支

## 1. main分支和dev分支同时开发

初始化仓库，并从main分支checkout出dev分支，并与远端仓库同步，git log结果如下

```bash
git-log
* 2df8c24 (HEAD -> dev, origin/main, origin/dev, main) [zheng] 初始化仓库
```

dev分支第一次开发，main分支第一次开发，并与远端仓库同步，git log结果如下：

```bash
git-log
* b8056d4 (HEAD -> main, origin/main) [zheng] main分支第一次开发
| * 1b8cde2 (origin/dev, dev) [zheng] dev分支第一次开发
|/
* 2df8c24 [zheng] 初始化仓库
```

main分支第二次开发，dev分支第二次开发，dev分支第三次开发，并与远端仓库同步，git log结果如下：

```bash
git-log
* 375e55d (HEAD -> dev, origin/dev) [zheng] dev分支第三次开发
* 3c84934 [zheng] dev分支第二次开发
* 1b8cde2 [zheng] dev分支第一次开发
| * 7f080da (origin/main, main) [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
|/
* 2df8c24 [zheng] 初始化仓库
```

## 2. 将dev分支自己的记录rebase

在merge main分支之前，将dev分支进行rebase以精简提交记录（做不做都行，提交太多的话可以做一下）

可以先查看dev分支跟main分支的公共祖先节点：`git merge-base main dev`，结果如下：

```bash
git merge-base main dev
2df8c24e150037461618c1fa217ad665805ee3fc
```

接着查看dev分支从该节点checkout出来之后进行了几次提交，命令及结果如下：

```bash
# 查看有哪些提交
git rev-list 2df8c24e150037461618c1fa217ad665805ee3fc..dev
375e55d6e306288a3bf7c8369217a8c1ee53c753
3c8493453bb095e7fdfff588863b8c743af2f528
1b8cde2e799342c9ddd9b00223825e8ff739297d
# 加上--count查看有几次提交
git rev-list --count 2df8c24e150037461618c1fa217ad665805ee3fc..dev
3
```

将这几次提交用rebase合并成一次提交：`git rebase -i HEAD~3`，会弹出来下面的编辑器：

```bash
pick 1b8cde2 [zheng] dev分支第一次开发
pick 3c84934 [zheng] dev分支第二次开发
pick 375e55d [zheng] dev分支第三次开发

# Rebase 2df8c24..375e55d onto 2df8c24 (3 commands)
#
# Commands:
# p, pick <commit> = use commit
# r, reword <commit> = use commit, but edit the commit message
# e, edit <commit> = use commit, but stop for amending
# s, squash <commit> = use commit, but meld into previous commit
# f, fixup [-C | -c] <commit> = like "squash" but keep only the previous
#                    commit's log message, unless -C is used, in which case
#                    keep only this commit's message; -c is same as -C but
#                    opens the editor
# x, exec <command> = run command (the rest of the line) using shell
# b, break = stop here (continue rebase later with 'git rebase --continue')
# d, drop <commit> = remove commit
# l, label <label> = label current HEAD with a name
# t, reset <label> = reset HEAD to a label
# m, merge [-C <commit> | -c <commit>] <label> [# <oneline>]
# .       create a merge commit using the original merge commit's
# .       message (or the oneline, if no original merge commit was
# .       specified); use -c <commit> to reword the commit message
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
```

可以直接将后面的两次提交合并到第一次提交（将后面两个pick改成squash即可）

保存并关闭后会弹出另一个编辑框：

```bash
# This is a combination of 3 commits.
# This is the 1st commit message:

[zheng] dev分支第一次开发

# This is the commit message #2:

[zheng] dev分支第二次开发

# This is the commit message #3:

[zheng] dev分支第三次开发

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Date:      Fri Jun 21 15:18:15 2024 +0800
#
# interactive rebase in progress; onto 2df8c24
# Last commands done (3 commands done):
#    squash 3c84934 [zheng] dev分支第二次开发
#    squash 375e55d [zheng] dev分支第三次开发
# No commands remaining.
# You are currently rebasing branch 'dev' on '2df8c24'.
#
# Changes to be committed:
#	modified:   README.md
#
```

这边就是rebase之后要用什么信息进行commit，我改成下面这样：

```bash
# This is a combination of 3 commits.
# This is the 1st commit message:

# [zheng] dev分支第一次开发

# This is the commit message #2:

# [zheng] dev分支第二次开发

# This is the commit message #3:

# [zheng] dev分支第三次开发

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# Date:      Fri Jun 21 15:18:15 2024 +0800
#
# interactive rebase in progress; onto 2df8c24
# Last commands done (3 commands done):
#    squash 3c84934 [zheng] dev分支第二次开发
#    squash 375e55d [zheng] dev分支第三次开发
# No commands remaining.
# You are currently rebasing branch 'dev' on '2df8c24'.
#
# Changes to be committed:
#	modified:   README.md
#
[zheng] dev分支所有开发合并，待merge
```

保存并关闭后，查看git log：

```bash
git-log
* df63990 (HEAD -> dev) [zheng] dev分支所有开发合并，待merge
| * 375e55d (origin/dev) [zheng] dev分支第三次开发
| * 3c84934 [zheng] dev分支第二次开发
| * 1b8cde2 [zheng] dev分支第一次开发
|/
| * 7f080da (origin/main, main) [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
|/
* 2df8c24 [zheng] 初始化仓库
```

强制推送到远端后的git log（因为rebase后跟远端的提交记录不一样了，dev只能强制推送）：

```bash
git-log
* cf3c5ab (HEAD -> dev, origin/dev) [zheng] dev分支所有开发合并，待merge
| * 7f080da (origin/main, main) [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
|/
* 2df8c24 [zheng] 初始化仓库
```

## 3. 采用merge的方式合并（推荐）

### 3.1. 将main分支merge到dev分支

先将main分支最新修改进行拉取（多人开发的话，每次都要拉一下保持最新）：`git fetch origin main`

将main分支合并到dev分支：`git merge main`

处理完冲突之后，进行提交并推送。此时git log如下：

```bash
git-log
*   44a9c7c (HEAD -> dev, origin/dev) [zheng] dev merge main
|\
| * 7f080da (origin/main, main) [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
* | cf3c5ab [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

### 3.2. 将dev分支merge到主分支

切换到main分支，并将dev分支merge到main分支：`git merge dev`

这边的例子比较简单，可以直接merge，没有冲突也没有更改，可以直接推送。此时git log如下：

```bash
git-log
*   44a9c7c (HEAD -> main, origin/main, origin/dev, dev) [zheng] dev merge main
|\
| * 7f080da [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
* | cf3c5ab [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

### 3.3. 删除dev分支

此时，dev分支开发完毕并且合并到main分支，可以将其删除

删除本地dev分支：`git branch -d dev`

删除远端dev分支：`git push origin --delete dev`

删除后git log如下：

```bash
git-log
*   44a9c7c (HEAD -> main, origin/main) [zheng] dev merge main
|\
| * 7f080da [zheng] main分支第二次开发
| * b8056d4 [zheng] main分支第一次开发
* | cf3c5ab [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

## 4. 采用rebase的方式合并（不推荐）

### 4.1. 在dev上rebase到main

在dev上rebase到main：`git rebase main`

需要解决冲突，解决完后：`git add .` && `git rebase --continue`

会弹出一个编辑器，内容如下：

```bash
[zheng] dev分支所有开发合并，待merge

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# interactive rebase in progress; onto 7f080da
# Last command done (1 command done):
#    pick cf3c5ab [zheng] dev分支所有开发合并，待merge
# No commands remaining.
# You are currently rebasing branch 'dev' on '7f080da'.
#
# Changes to be committed:
#	modified:   README.md
#
```

我改成如下进行保存关闭：

```bash
# [zheng] dev分支所有开发合并，待merge

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
#
# interactive rebase in progress; onto 7f080da
# Last command done (1 command done):
#    pick cf3c5ab [zheng] dev分支所有开发合并，待merge
# No commands remaining.
# You are currently rebasing branch 'dev' on '7f080da'.
#
# Changes to be committed:
#	modified:   README.md
#
[zheng] dev上进行rebase main
```

此时的git log如下（可以跟2中的最后一个git log对比一下，发现此时的dev接到了main的后面）：

```bash
git-log
* 96bd056 (HEAD -> dev) [zheng] dev上进行rebase main
* 7f080da (origin/main, main) [zheng] main分支第二次开发
* b8056d4 [zheng] main分支第一次开发
| * cf3c5ab (origin/dev) [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

### 4.2. 将dev分支merge到主分支

不用提交，直接切换到main分支进行合并：`git merge dev`

此时的git log如下：

```bash
git-log
* 96bd056 (HEAD -> main, dev) [zheng] dev上进行rebase main
* 7f080da (origin/main) [zheng] main分支第二次开发
* b8056d4 [zheng] main分支第一次开发
| * cf3c5ab (origin/dev) [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

这时候可以直接将main分支push到远端，不需要-f就能成功，git log如下：

```bash
git-log
* 96bd056 (HEAD -> main, origin/main, dev) [zheng] dev上进行rebase main
* 7f080da [zheng] main分支第二次开发
* b8056d4 [zheng] main分支第一次开发
| * cf3c5ab (origin/dev) [zheng] dev分支所有开发合并，待merge
|/
* 2df8c24 [zheng] 初始化仓库
```

### 4.3. 删除dev分支

删除远端dev分支：`git push origin --delete dev`

删除本地dev分支：`git branch -d dev`

此时git log如下，是一个非常简洁的历史记录：

```bash
git-log
* 96bd056 (HEAD -> main, origin/main) [zheng] dev上进行rebase main
* 7f080da [zheng] main分支第二次开发
* b8056d4 [zheng] main分支第一次开发
* 2df8c24 [zheng] 初始化仓库
```

# git入门（较乱的记录）

参考链接：[https://www.bilibili.com/video/BV1MU4y1Y7h5](https://www.bilibili.com/video/BV1MU4y1Y7h5)

## 获取本地仓库

1. 本地创建一个空目录作为**本地git仓库**。
2. 在这个目录的终端中执行`git init`，成功的话可以看到里面有一个`.git`文件夹

### 工作流程

1. 工作区（workspace）--->暂存区（index）--->仓库（repository）
2. 工作区：修改已有文件（未暂存unstaged），新创建一个文件（未跟踪untracked）
3. 暂存区：（已暂存staged）
4. 仓库：修改进入到仓库就变成了一次提交记录，commit 01, 02 ...
5. 工作区到暂存区：`git add .`暂存所有，或者`git add 文件名`暂存指定文件
6. 暂存区到仓库：`git commit -m "提交消息"`
7. 查看状态：`git status`

### 查看提交日志

查看历史记录：`git log`，可以有参数：

1. `-all`：显示所有分支
2. `-pretty=oneline`：将提交信息显示为一行
3. `-abbrev-commit`：是的输出的commit更简短
4. `-graph`：以图的形式显示

想要快速应用以上所有，可以添加别名，使用更快：

1. 创建bashrc文件：`touch ~/.bashrc`

2. 在文件中输入以下内容：

   ```bash
   # 用于输出git提交日志, 之后使用git-log就相当于后面那一串
   alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'
   # 用于输出当前目录所有文件及基本信息
   alias ll='ls -al'
   ```

3. 执行`source ~/.bashrc`生效

### 版本回退

1. 命令：`git reset --hard commitID`，其中的`commitID`可以使用`git log`查看
2. `git reflog`命令可以查看已经删除的提交记录

### 是否添加文件至git管理

1. 创建文件：`touch .gitignore`
2. 然后把不需要git管理的文件名字写到里面

## 分支

一些相关操作：

1. 查看分支：`git branch`（`git branch -vv`可以查看本地与远程的关联关系，后面有讲到）
2. 创建分支：`git branch 分支名`
3. 切换分支：`git checkout 分支名`
4. 创建并切换分支：`git checkout -b 分支名`
5. 合并分支：`git merge 分支名`
6. 删除分支：`git branch -d 分支名`
7. 强制删除分支：`git branch -D 分支名`

### 解决冲突

两个分支上对文件的修改可能会存在冲突，例如修改了同一个文件的同一行，这时就需要手动解决冲突，解决冲突的步骤如下：

1. 处理文件中冲突的地方
2. 将解决完的冲突的文件加入暂存区（add）
3. 提交到仓库（commit）

### 分支在开发中使用的流程

在开发中，一般有如下分支使用原则与流程：

- master（生产）分支
  - 线上分支，主分支，中小规模项目作为线上运行的应用对应的分支
- develop（开发）分支
  - 是从master上创建的分支，一般作为开发部门的**主要开发分支**，如果没有其他并行开发不同期上线要求，都可以在此版本上进行开发，阶段开发完成后，需要合并到master分支，准备上线
- feature/xxxx分支
  - 从develop创建的分支，一般是**同期并行开发，但不同期上线**时创建的分支，分支上的研发任务完成后合并到develop分支
- hotfix/xxxx分支
  - 从master派生的分支，一般作为**线上bug修复使用**，修复完成后需要合并到master、test、develop分支
- 还有其他一些分支，例如test分支（用于代码测试）、pre分支（预上线分支）等

## git远程仓库

### 配置SSH公钥

1. 生成ssh公钥：`ssh-keygen -t rsa`，不断回车。如果存在，则覆盖
2. 获取公钥：`cat ~/.ssh/id_rsa.pub`，然后添加到远程仓库的配置
3. 验证是否成功：`ssh -T github@github.com`，或者别的平台，如gitee，后面的换一下就行

### 操作远程仓库

1. 在远程先创建个空的仓库，一定要空的，README都不要
2. `git remote add origin 远程仓库的ssh路径`，其中`origin`是**远端名称**，一般都用origin
3. 查看远程仓库：`git remote`
4. 推送到远程仓库：`git push [-f][--set-upstream][远端名称[本地分支名]:[远端分支名]]`
   1. 如果远程分支名和本地分支名相同，则可以只写本地分支名，如`git push origin master`。
   2. `f`的意思是如果本地分支改了，远程分支也改了，就强制用本地的覆盖。以本地推上去的为准。
   3. `-set-upstream`：推送到远端的同时建立起和远端分支的关联关系。
   4. 如果当前分支已经和远端分支关联，则可以省略分支名和远端名。`git push`将master分支推送到已经关联的远端分支。
   5. `git branch -vv`：可以查看本地仓库与远程仓库的关联关系。

### 从远程仓库克隆

如果有一个远端仓库，可以直接克隆到本地（这个操作不会很频繁）。命令：`git clone <仓库路径> [本地目录]`，本地目录可以省略，会自动生成一个目录。

### 从远程仓库抓取和拉取

- 抓取命令：`git fetch [远端名称] [分支名]`
  - **抓取指令就是将仓库里的更新都抓取到本地，不会进行合并**
  - 不过不指定远端名称和分支名，则**抓取所有分支**
- 拉取命令：`git pull [远端名称] [分支名]`
  - **拉取指令就是将远端仓库的修改拉到本地并自动进行合并，相当于fetch+merge**
  - 不过不指定远端名称和分支名，则**抓取所有并更新当前分支**

### 远程冲突解决

在一段时间，A、B两个用户修改了同一个文件，且修改了同一行位置的代码，此时会发生合并冲突。
A用户在本地修改代码后优先推送到远程仓库，此时B用户在本地修订代码，提交到本地仓库后，也需要推送到远程仓库，此时B用户晚于A用户，**故需要先拉取远程仓库的提交，经过合并后，才能推送到远端分支**。

## 铁令

1. 切换分支前先提交本地的修改
2. 代码及时提交，提交过了就不会丢了
3. 遇到任何问题都不要删除文件目录

## 其他

假设个场景：
工作项目远端有很多很多分支，git clone下来一个分支开发。需要合并分支，或者复制远端别的分支的时候。

### git stash

在当前分支有修改，想切换到别的分支查看的时候，可以用`git stash`命令：

1. `git stash list`：可以查看当前仓库有多少个stash，stash就是一个栈
2. `git stash save "message"`：可以用该命令stash，同时添加message之后需要pop的时候不会错
3. `git stash pop 编号(如stash@{1})`：查看完别的分支回来，需要继续开发的话，可以把暂存的stash给pop出来，其中`stash@{1}`是标识，可以用`git stash list`查看原来添加时候的信息来选择你要pop出来的stash记录。**千万不要pop错了，不然怪麻烦的**
4. `git stash drop 编号(如stash@{1})`：可以用这个命令删除指定stash记录，其中`stash@{1}`就是删除第二条
5. `git stash clear`：清空所有stash记录

### git fetch

一般我就用两个应用场景：

1. 需要切换到远程的一个分支的时候。`git fetch origin 远程分支名`，这样可以保证你待会儿切换过去的分支是远程最新的，否则有可能就是之前`git clone`时候的那个版本。（可以根据id查看看）。fetch之后就可以`git checkout -b 分支名 origin/远程分支名`，这样本地就有这个分支了
2. 需要merge的时候，也需要fetch一下最新的版本，本地已经有修改还没提交的话，就可以配合stash使用一下。先更新再`stash pop`出来解决冲突。