- [一些命令](#一些命令)
- [MobaXterm配置](#mobaxterm配置)
  - [通过跳板机](#通过跳板机)
  - [直接连接开发机](#直接连接开发机)
- [非对称加密方案的登录流程](#非对称加密方案的登录流程)
- [SSH基于公钥认证](#ssh基于公钥认证)
- [几个文件的介绍](#几个文件的介绍)
- [关于known\_hosts](#关于known_hosts)


[参考链接1](https://blog.csdn.net/wang_qiu_hao/article/details/127902007)

# 一些命令
1. 重启ssh服务：`/etc/init.d/ssh restart`（`/etc/ssh/sshd_config`）
2. 已有私钥生成公钥：`ssh-keygen -y -f ${/path/to/private_key} > ${/path/to/gen_pub_key} -C <some tag such as email>`
   1. `-y`：从私钥提取公钥
3. 生成密钥对：`ssh-keygen -t rsa [-f </path/to/private_key> | -C <some tag such as email>]`
   1. `-t rsa`：生成RSA密钥对
   2. `-f </path/to/private_key>`：指定私钥或公钥文件
   3. `-C <some tag such as email>`：添加注释（例如邮箱）
4. 连接命令：
   1. 密码登录：`ssh <user_name>@<remote_ip> -p <remote_port> [-o HostKeyAlgorithms=+ssh-rsa]` 
   2. 密钥登录：`ssh <user_name>@<remote_ip> -p <remote_port> -i <private_key_path>`
   3. 密钥通过跳板机登录开发机：`ssh <user_name>@<dev_ip> -i <private_key_path> -o ProxyCommand="ssh <user_name>@<jumpserver_ip> -p <jumpserver_port> -i <private_key_path> -q -W <dev_ip>:<dev_port>"`
   4. 使用第三种有可能需要先在远程主机的`authorized_keys`中添加客户端的公钥

# MobaXterm配置

## 通过跳板机

1. Remote host：`<jumpserver_ip>`
2. Specify username：自己进去new一个，就是跳板机的账户名跟密码
3. Port：`<jumpserver_port>`

## 直接连接开发机

1. （需要先在远程主机的`authorized_keys`中添加客户端的公钥）
2. Remote host：`<dev_ip>`
3. Specify username：一样
4. Port：22
5. Use private key：`<private_key_path>`
6. Network Settings：SSH gateway (jump host)
    1. Gateway host：`<jumpserver_ip>`
    2. Username：`<user_name>`
    3. Port：`<jumpserver_port>`
    4. Use SSH key：`<private_key_path>`

# 非对称加密方案的登录流程

1. Server收到Client的登录请求后，Server把自己的公钥发给Client
2. Client使用这个公钥，将密码进行加密
3. Client将加密的密码发送给Server
4. Server用自己的私钥，解密登录密码，然后验证其合法性。若验证结成功，给Client相应的响应

私钥是Server端独有，这就保证了Client的登录信息即使在网络传输过程中被窃据，也没有私钥进行解密，保证了数据的安全性，这充分利用了非对称加密的特性。

但是，存在一些问题，详见参考链接1。

# SSH基于公钥认证

1. Client将自己的公钥存放在Server上，追加在文件authorized_keys中
2. Server接收到Client的连接请求后，会在authorized_keys中匹配到Client的公钥pubKey，并生成随机数R，用Client的公钥对该随机数进行加密得到pubKey(R)，然后将加密后信息发送给Client
3. Client通过私钥进行解密得到随机数R，然后对随机数R和本次会话的SessionKey利用MD5生成摘要Digest1，发送给Server
4. Server会也会对R和SessionKey利用同样摘要算法生成Digest2
5. Server会最后比较Digest1和Digest2是否相同，完成认证过程

# 几个文件的介绍

1. id_rsa：保存私钥
2. id_rsa.pub：保存公钥
3. authorized_keys：保存已授权的客户端公钥
4. known_hosts：保存已认证的远程主机ID。

需要注意的是：一台主机可能既是Client，也是Server。所以会同时拥有authorized_keys和known_hosts。

# 关于known_hosts

1. known_hosts中存储的内容是什么？

   known_hosts中存储是已认证的远程主机host key，每个SSH Server都有一个secret, unique ID, called a host key。

2. host key何时加入known_hosts的？

   当我们第一次通过SSH登录远程主机的时候，Client端会有如下提示：

   > Host key not found from the list of known hosts.
   > Are you sure you want to continue connecting (yes/no)?

   此时，如果我们选择yes，那么该host key就会被加入到Client的known_hosts中，格式如下：

   > \# domain name+encryption algorithm+host key
   > example.hostname.com ssh-rsa AAAAB4NzaC1yc2EAAAABIwAAAQEA。。。

3. 为什么需要known_hosts？

   known_hosts主要是通过Client和Server的双向认证，从而避免中间人（man-in-the-middle attack）攻击，每次Client向Server发起连接的时候，不仅仅Server要验证Client的合法性，Client同样也需要验证Server的身份，SSH client就是通过known_hosts中的host key来验证Server的身份的。（但是也不够安全，比如第一次连接一个未知Server的时候，known_hosts还没有该Server的host key，也可能遭到中间人攻击）