- [install](#install)
- [cmd](#cmd)
- [非对称加密](#非对称加密)
- [对称加密](#对称加密)

正常自己加密文件使用对称加密就行，比较简单。

# install

```bash
# 不一定是最新包
sudo apt update
sudo apt install gnupg
```

# cmd

简单情况下最好所有命令指定`--homedir`，不然会根据配置文件来运行。详细例子查看[非对称加密](#非对称加密)

**生成密钥对**

```bash
gpg --homedir=${user_gpg_home_dir} --quick-generate-key ${name <email>} ${密钥类型与大小} ${密钥用途} ${密钥有效期}
gpg --batch --homedir "$ALICE_HOME" --quick-generate-key "Alice <alice@example.com>" rsa2048 cert,sign,encrypt 1y
# --batch：启用批处理模式，无需用户交互
# --homedir：指定 GPG 主目录
# --quick-generate-key：快速生成密钥对
# --full-generate-key：生成密钥对，需要用户交互
```

| 特性           |           `gpg --full-generate-key`            | `gpg --quick-generate-key`     |
| -------------- | :--------------------------------------------: | ------------------------------ |
| **交互式**     |            是，用户需要输入多个选项            | 否，默认选项自动生成密钥       |
| **定制化选项** | 提供更多选项，可以配置密钥类型、大小、有效期等 | 仅提供基本的用户信息和默认选项 |
| **密钥类型**   |        用户可以选择（RSA, DSA, ECC等）         | 默认使用 RSA                   |
| **密钥大小**   |     用户可以选择（如 2048 位、4096 位等）      | 默认 2048 位                   |
| **密钥用途**   |       用户可以选择（签名、加密、认证等）       | 默认支持签名、加密和认证       |
| **适用场景**   |             需要详细配置密钥的情况             | 需要快速生成密钥的情况         |

**查看密钥**

```bash
gpg --list-keys        # 查看公钥
gpg --list-secret-keys # 查看私钥
gpg --fingerprint "Alice" # 查看密钥的详细信息
```

**导出密钥**

```bash
gpg --export -a "Alice" > alice_public.key              # 导出公钥
gpg --export-secret-key -a "Alice" > alice_private.key  # 导出私钥
# -a：ASCII 格式输出
```

**导入密钥**

```bash
gpg --import alice_public.key  # 导入 Alice 的公钥
gpg --import alice_private.key # 导入 Alice 的私钥
```

**加密文件**

```bash
# 非对称加密
gpg -e -r ${receiver_name} -o ${encrypted_file.gpg} ${encrypted_file}
gpg -e -r "Bob" -o message.txt.gpg message.txt
# -e：非对称加密
# -r "Bob"：指定接收者（Bob）的公钥。
# -o message.txt.gpg：指定加密后的输出文件。

# 对称加密
gpg -c ${encrypted_file}
# -c：对称加密
# 输入两次密码，生成${encrypted_file.gpg}
```

**解密文件**

```bash
gpg -d ${encrypted_file.gpg} > ${encrypted_file}
gpg -d message.txt.gpg > message.txt
# 对称加密的话需要输入密码
```

**签名与验证签名**

```bash
gpg --sign ${encrypted_file} # 签名，会生成${encrypted_file.gpg}
gpg --verify ${encrypted_file.gpg} # 验证签名，会显示签名的验证信息。
```

# 非对称加密

以下给一个脚本演示的例子，脚本路径在`~/tmp/gpg/gpg_asymmetric_demo.sh`：

```bash
#!/bin/bash
set -e
echo "🚀 GPG 非对称加密模拟开始"
# 创建两个独立的 GPG 环境
ALICE_HOME="./gpg-alice"
BOB_HOME="./gpg-bob"
MESSAGE_FILE="message.txt"
ENCRYPTED_FILE="message.txt.gpg"
DECRYPTED_FILE="decrypted.txt"

rm -rf "$ALICE_HOME" "$BOB_HOME"
mkdir "$ALICE_HOME" "$BOB_HOME"
chmod 700 "$ALICE_HOME" "$BOB_HOME"  # 添加此行修复权限问题

echo " ================== 正在为 Alice 生成密钥对 ================== "
gpg --batch --homedir=$ALICE_HOME --quick-generate-key "Alice <alice@example.com>" rsa2048 cert,sign,encrypt 1y # 生成密钥
gpg --homedir=$ALICE_HOME --export -a "Alice" > ${ALICE_HOME}/alice_public.key # alice导出公钥
gpg --homedir=$ALICE_HOME --export-secret-key -a "Alice" > ${ALICE_HOME}/alice_private.key # alice导出私钥

echo " ================== 正在为 Bob 生成密钥对 ================== "
gpg --batch --homedir=$BOB_HOME --quick-generate-key "Bob <bob@example.com>"
gpg --homedir=$BOB_HOME --export -a "Bob" > ${BOB_HOME}/bob_public.key # bob导出公钥
gpg --homedir=$BOB_HOME --export-secret-key -a "Bob" > ${BOB_HOME}/bob_private.key # bob导出私钥

echo " ================== Alice 添加 Bob 的公钥 ================== "
gpg --homedir=$ALICE_HOME --import ${BOB_HOME}/bob_public.key

echo " ================== Bob 添加 Alice 的公钥 ================== "
gpg --homedir=$BOB_HOME --import ${ALICE_HOME}/alice_public.key

# 模拟 Alice 写一条秘密消息
echo "这是 Alice 发给 Bob 的加密信息。" > "$MESSAGE_FILE"

echo " ================== Alice 用 Bob 的公钥加密消息 ================== "
gpg --homedir=$ALICE_HOME -e -r "Bob" -o "$ENCRYPTED_FILE" "$MESSAGE_FILE"

echo " ================== Bob 收到加密文件，正在解密 ================== "
gpg --homedir "$BOB_HOME" -d "$ENCRYPTED_FILE" > "$DECRYPTED_FILE"

echo " ================== 解密完成，Bob 收到的内容如下 ================== "
cat "$DECRYPTED_FILE"

# 清理临时文件（可选）
# rm -rf "$ALICE_HOME" "$BOB_HOME" *.key "$MESSAGE_FILE" "$ENCRYPTED_FILE" "$DECRYPTED_FILE"

echo "🎉 模拟完成！Alice 成功加密消息，Bob 成功解密！"

```

# 对称加密

```bash
# 加密
gpg -c ${encrypted_file} # 然后输入两次密码即可
# 解密
gpg -d ${encrypted_file.gpg} > ${decrypted_file} # 输入密码即可
```