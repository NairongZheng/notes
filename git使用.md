# git学习
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
1. `--all`：显示所有分支
2. `--pretty=oneline`：将提交信息显示为一行
3. `--abbrev-commit`：是的输出的commit更简短
4. `--graph`：以图的形式显示

想要快速应用以上所有，可以添加别名，使用更快：
1. 创建bashrc文件：`touch ~/.bashrc`
2. 在文件中输入以下内容：
   ```bash
   # 用于输出git提交日志, 之后使用git-log就相当于后面那一串
   alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'
   # 用于输出当前目录所有文件及基本信息
   alias ll='ls-al'
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
   2. `-f`的意思是如果本地分支改了，远程分支也改了，就强制用本地的覆盖。以本地推上去的为准。
   3. `--set-upstream`：推送到远端的同时建立起和远端分支的关联关系。
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
