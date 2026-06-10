

# 安装与配置

**安装 rclone**

```shell
mkdir -p ~/.local/{bin,lib,include,src}
cd ~/.local/src
curl -O https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip rclone-current-linux-amd64.zip
cd rclone-*-linux-amd64
# 习惯用软链接
ln -s $(pwd)/rclone ~/.local/bin/rclone
# 然后需要在环境变量中添加 ~/.local/bin
```

**配置 rclone 需要的 aksk**

```shell
# ~/.config/rclone/rclone.conf
[remote_name]
type = s3
provider = xxx
access_key_id = xxxx
secret_access_key = xxxx
endpoint = url
```

# 常用命令

**基本命令**

```shell
# 配置远端存储，交互式创建远端，生成 rclone.conf
rclone config
# 列出配置的所有远端
rclone listremotes
# 查看 rclone 版本
rclone version
```

**列出文件和目录**

```shell
# 列出文件（递归）
rclone ls remote://path
# 列出文件详细信息（递归）
rclone lsl remote://path
# 列出目录（递归）
rclone lsd remote://path
# 灵活列出文件/目录
rclone lsf remote://path [options]
    # --dirs-only
    # --files-only
    # --max-depth 1
```

**上传/下载**

```shell
# copy 复制文件到目录（不删除源文件）
rclone copy source des
# copyto 上传单个文件并重命名（不删除源文件）
rclone copyto source des
# move 移动文件到目录（删除源文件）
rclone move source de
# moveto 移动单个文件并重命名（删除源文件）
rclone moveto source des
```

**删除文件/目录**

```shell
# 删除单个文件
rclone delete remote://path/to/file
# 递归删除目录及内容
rclone purge remote://path/to/dir
# 删除空目录
rclone rmdirs remote://path
```

**同步文件/目录**

```shell
# 将源同步到远端，删除远端多余文件（小心！）
rclone sync source remote:path
# 检查源和远端文件是否一致
rclone check source remote:path
```

**其他命令**

```shell
# 创建远端目录（前缀），对象存储目录本质是前缀
rclone mkdir remote:path
# 创建空文件，用于占位或测试
rclone touch remote:path/file
# 检查加密远端文件一致性
rclone cryptcheck remote:path
# 查看远端文件内容，输出到 stdout
rclone cat remote:path/file
# 查看配置文件内容，包括 AccessKey/Secret（可脱敏）
rclone config show
```

# 常用参数

| 参数                       | 功能                         | 常用说明                       |
| -------------------------- | ---------------------------- | ------------------------------ |
| `-P`                       | 显示进度                     | 上传/下载时推荐                |
| `--dry-run`                | 模拟运行，不真正执行         | 上传前测试路径                 |
| `-L` / `--copy-links`      | 跟随 symlink 上传真实文件    | 默认不跟随 symlink             |
| `--files-only`             | 只列文件                     | 搭配 lsf / ls 使用             |
| `--dirs-only`              | 只列目录                     | 同上                           |
| `--max-depth N`            | 限制递归深度                 | 一般只看一层使用 --max-depth 1 |
| `--checksum`               | 用 checksum 判断文件是否变化 | 同步或上传可用                 |
| `--progress`               | 显示详细进度                 | 同 -P 类似                     |
| `--exclude` / `--include`  | 过滤文件                     | 上传/同步时使用                |
| `--create-empty-src-dirs`  | 保留空目录                   | 同步时使用                     |
| `--transfers N`            | 并发传输文件数量             | 上传/下载/同步时常用           |
| `--checkers N`             | 并发检查数量                 | 与 transfers 配合使用          |
| `--multi-thread-streams N` | 单文件多线程并发流           | 适合大文件提升速度             |
| `--multi-thread-cutoff S`  | 启用单文件多线程的阈值大小   | 如 64M、128M                   |
