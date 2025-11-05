

# 安装

```shell
# 安装
pip install -U "ray[default]"
# 验证安装
ray --version
```

# 部署

在主节点上：

```shell
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
    # --head: 代表这是主节点
    # --port: 指定ray的通信端口
    # --dashboard-host: 允许从任意ip地址访问dashboard
    # dashboard-port: dashboard的端口
```

在其他节点上：

```shell
ray start --address='${master_node_ip}:6379'
```

可以查看节点状态：

```shell
ray list nodes
# 也可以在 http://${ip}:8265 查看dashboard
```

停止集群：在所有节点上分别操作：

```shell
ray stop
```

然后在任意节点执行程序，所有节点都会分担任务执行。

# 使用docker来模拟

因为没有多集群的测试环境，这边使用docker容器的方式来模拟。

```shell
# 1. 创建一个自定义网络
docker network create ray-network
# 2. 启动主节点
docker run -d --name ray-head \
  --network ray-network \
  -p 8265:8265 -p 6379:6379 \
  rayproject/ray:latest \
  bash -c "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --block"
# 3. 启动worker节点
docker run -d --name ray-worker-1 \
  --network ray-network \
  rayproject/ray:latest \
  bash -c "ray start --address='ray-head:6379' --block"
# 4. 查看ray集群信息
docker exec -it ray-head ray list nodes
docker exec -it ray-head ray status
# 5. 查看ray的dashboard: http://localhost:8265
```

可以使用以下脚本来测试：

```python
import ray
import socket
import os
import time

# 初始化 Ray，连接到 head 节点
print("Connecting to Ray cluster...")
ray.init(address="127.0.0.1:6379")

# 输出集群信息
print("\nCluster resources:")
print(ray.cluster_resources())

# 定义一个远程任务
@ray.remote
def remote_task(x):
    hostname = socket.gethostname()
    pid = os.getpid()
    time.sleep(1)
    return f"Task {x} executed on {hostname} (PID={pid})"

# 分发多个任务
tasks = [remote_task.remote(i) for i in range(20)]

# 获取执行结果
results = ray.get(tasks)

print("\n=== Execution Results ===")
for r in results:
    print(r)
```

然后在任意一个节点运行即可看到类似以下log：

```shell
(dev) zhengnairong@60099732M test % docker exec -it ray-head bash
(base) ray@b6e4e44c7f3f:~$ which python
/home/ray/anaconda3/bin/python
(base) ray@b6e4e44c7f3f:~$ python /tmp/test.py
Connecting to Ray cluster...
2025-11-04 22:52:23,049 INFO worker.py:1832 -- Connecting to existing Ray cluster at address: 172.18.0.2:6379...
2025-11-04 22:52:23,107 INFO worker.py:2012 -- Connected to Ray cluster.
/home/ray/anaconda3/lib/python3.9/site-packages/ray/_private/worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(

Cluster resources:
{'CPU': 16.0, 'memory': 5037716276.0, 'node:172.18.0.3': 1.0, 'object_store_memory': 2159021260.0, 'node:0.0.0.0': 1.0, 'node:__internal_head__': 1.0}
[2025-11-04 22:53:14,867 E 766 798] core_worker_process.cc:825: Failed to establish connection to the metrics exporter agent. Metrics will not be exported. Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14
(remote_task pid=799) [2025-11-04 22:53:10,878 E 799 852] core_worker_process.cc:825: Failed to establish connection to the metrics exporter agent. Metrics will not be exported. Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14

=== Execution Results ===
Task 0 executed on b6e4e44c7f3f (PID=800)
Task 1 executed on b6e4e44c7f3f (PID=802)
Task 2 executed on b6e4e44c7f3f (PID=801)
Task 3 executed on b6e4e44c7f3f (PID=803)
Task 4 executed on b6e4e44c7f3f (PID=804)
Task 5 executed on b6e4e44c7f3f (PID=799)
Task 6 executed on b6e4e44c7f3f (PID=805)
Task 7 executed on b6e4e44c7f3f (PID=806)
Task 8 executed on b6e4e44c7f3f (PID=806)
Task 9 executed on b6e4e44c7f3f (PID=801)
Task 10 executed on b6e4e44c7f3f (PID=803)
Task 11 executed on b6e4e44c7f3f (PID=806)
Task 12 executed on 13fe4cd9a5df (PID=212)
Task 13 executed on 13fe4cd9a5df (PID=208)
Task 14 executed on 13fe4cd9a5df (PID=211)
Task 15 executed on 13fe4cd9a5df (PID=214)
Task 16 executed on 13fe4cd9a5df (PID=213)
Task 17 executed on 13fe4cd9a5df (PID=210)
Task 18 executed on 13fe4cd9a5df (PID=209)
Task 19 executed on 13fe4cd9a5df (PID=207)
(remote_task pid=804) [2025-11-04 22:53:10,919 E 804 1006] core_worker_process.cc:825: Failed to establish connection to the metrics exporter agent. Metrics will not be exported. Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14 [repeated 7x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(base) ray@b6e4e44c7f3f:~$
```