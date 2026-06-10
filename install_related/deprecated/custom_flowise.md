- [Quick Start](#quick-start)
- [Developers](#developers)
- [Custom](#custom)


[Flowise：https://github.com/FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise)

# Quick Start

**安装基础环境**

```bash
# 运行容器（这边做了端口映射，所以容器3000上的服务都通过宿主机60014访问）
docker run -it -v /data:/data -p 60014:3000/tcp --name damonzheng_flowise ubuntu:20.04 bash
# 安装一些基础包（省略）
# 安装nvm (Node Version Manager)
git clone https://github.com/nvm-sh/nvm.git ~/.nvm
echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.bashrc
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.bashrc
source ~/.bashrc
# 安装nodejs
nvm install 18.18.2
nvm use 18.18.2
nvm alias default 18.18.2
    # nvm list: 查看nvm
    # node -v: 查看nodejs版本
    # npm -v: 查看npm版本
```

**flowise quick start**

```bash
# 设置npm代理
npm config set registry https://mirrors.cloud.tencent.com/npm/
# 安装flowise
npm install -g flowise
# 启动flowise
npx flowise start   # 即可在 http://<dev_ip>:60014 查看
```

# Developers

**获取源码**

```bash
# 安装pnpm
npm i -g pnpm
# 获取flowise源码
git clone https://github.com/FlowiseAI/Flowise.git
cd Flowise
```

**安装依赖**

```bash
pnpm install

# 安装完成后，显示如下：
devDependencies:
+ @babel/preset-env 7.24.0
+ @babel/preset-typescript 7.18.6
+ @types/express 4.17.21
+ @typescript-eslint/typescript-estree 7.13.1
+ eslint 8.57.0
+ eslint-config-prettier 8.10.0
+ eslint-config-react-app 7.0.1
+ eslint-plugin-jsx-a11y 6.8.0
+ eslint-plugin-markdown 3.0.1
+ eslint-plugin-prettier 3.4.1
+ eslint-plugin-react 7.34.0
+ eslint-plugin-react-hooks 4.6.0
+ eslint-plugin-unused-imports 2.0.0
+ husky 8.0.3
+ kill-port 2.0.1
+ lint-staged 13.3.0
+ prettier 2.8.8
+ pretty-quick 3.3.1
+ rimraf 3.0.2
+ run-script-os 1.1.6
+ turbo 1.10.16
+ typescript 5.5.2

╭ Warning ─────────────────────────────────────────────────────────────────────╮
│                                                                              │
│   Ignored build scripts: @swc/core, aws-sdk, bufferutil, canvas, core-js,    │
│   core-js-pure, couchbase, cpu-features, cypress, es5-ext, esbuild,          │
│   grpc-tools, msgpackr-extract, protobufjs, puppeteer, sharp, ssh2,          │
│   utf-8-validate.                                                            │
│   Run "pnpm approve-builds" to pick which dependencies should be allowed     │
│   to run scripts.                                                            │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

. postinstall$ husky install
│ husky - Git hooks installed
└─ Done in 277ms
Done in 25m 7.2s using pnpm v10.10.0
```

**build**

```bash
pnpm build

# build 成功后显示如下：
flowise-ui:build: ✓ built in 4m 14s
flowise-ui:build: Failed to mark outputs as cached for flowise-ui#build: rpc error: code = DeadlineExceeded desc = failed to load filewatching in time
flowise:build: cache miss, executing 17cb5425725bb56a
flowise:build:
flowise:build: > flowise@2.2.8 build /data/p3/damonzheng/code/some_tools/Flowise/packages/server
flowise:build: > tsc
flowise:build:

 Tasks:    4 successful, 4 total
Cached:    0 cached, 4 total
  Time:    5m12.698s

⠴ ...writing to cache...  [2s]
```

**启动**

```bash
pnpm start

# 启动log：
> flowise@2.2.8 start /data/p3/damonzheng/code/some_tools/Flowise
> run-script-os


> flowise@2.2.8 start:default
> cd packages/server/bin && ./run start

2025-05-09 03:18:18 [INFO]: Starting Flowise...
2025-05-09 03:18:19 [INFO]: 📦 [server]: Data Source is initializing...
2025-05-09 03:19:40 [INFO]: 📦 [server]: Data Source has been initialized!
2025-05-09 03:19:40 [INFO]: ⚡️ [server]: Flowise Server is listening at :3000

# 即可在 http://<dev_ip>:60014 查看
```

**开发调试更新**

```bash
# 安装依赖
apt install -y xdg-utils
# 启动
pnpm dev [-- --host]
# 不添加 -- --host 只能在localhost查看
# 可边修改边实时更新，不过端口在8080，需要做容器端口映射或者做个端口转发，类似：
# socat TCP-LISTEN:60010,reuseaddr,fork TCP:192.168.188.12:8080 &
```

# Custom

在`packages/components/nodes`中创建文件夹来添加类别，比如 `wa`

然后再在其中添加需要实现的工具，比如：

```bash
# 文件夹结构
wa
├── DebugOutput
│   ├── DebugOutput.ts
│   └── debugoutput.svg
└── HelloWorld
    ├── HelloWorld.ts
    └── helloworld.svg
```

```ts
// HelloWorld.ts
import { INode, INodeData, INodeParams } from '../../../src/Interface'

class HelloWorldNode implements INode {
    label: string
    name: string
    type: string
    icon: string
    category: string
    description: string
    inputs: INodeParams[]
    version: number
    baseClasses: string[]

    constructor() {
        this.label = 'Hello World'
        this.name = 'helloWorld'
        this.type = 'HelloWorld'
        this.icon = 'function.svg'
        this.category = 'wa'  // 这里定义在 UI 中的分类名
        this.description = '输出 Hello World'
        this.inputs = [
            {
                label: 'Name',
                name: 'name',
                type: 'string',
                placeholder: '输入你的名字'
            }
        ]
        this.version = 1.0
        this.baseClasses = ['HelloWorldNode']
    }

    async run(nodeData: INodeData): Promise<string> {
        const name = nodeData.inputs?.name as string;
        return `Hello, ${name || 'World'}!`;
    }
}

module.exports = { nodeClass: HelloWorldNode }
```

```bash
# helloworld.svg
<svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect x="5" y="7" width="22" height="18" rx="2" stroke="black" stroke-width="2"/>
<path d="M11 12L15 16L11 20" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M17 20H21" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
```

然后再重新编译运行即可

```bash
pnpm build
pnpm start
```