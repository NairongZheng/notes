

# 安装

```shell
# 安装基础工具
apt update
apt install -y curl gnupg lsb-release
# 添加 Cloudflare 源
mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | tee /etc/apt/sources.list.d/cloudflared.list
# 安装 Cloudflare
apt update
apt install -y cloudflared
```

# 使用

```shell
cloudflared tunnel --url http://localhost:<port>
# 然后就可以在打印出来的临时公网上访问了
```