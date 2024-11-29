[toc]

# ansible_test
首先项目文件夹是ansible_test，文件夹结构是：

```
.
|-- ansible
|   |-- ansible_playbook_client1.sh
|   |-- ansible_playbook_client2.sh
|   |-- ansible_playbook_client3.sh
|   |-- client1.txt
|   |-- client2.txt
|   |-- client3.txt
|   |-- deploy_client1.yml
|   |-- deploy_client2.yml
|   |-- deploy_client3.yml
|   |-- update_client1.sh
|   |-- update_client2.sh
|   `-- update_client3.sh
|-- config
|   |-- config.ini
|   |-- private_key.key
|   `-- private_key.key.pub
|-- home.py
`-- run.sh
```

用的是wsl测试的

在上面运行了三个容器，ip分别是：

| 容器     |            |
| -------- | ---------- |
| dev_main | 172.17.0.2 |
| dev1     | 172.17.0.3 |
| dev2     | 172.17.0.4 |

所以`./config/config.ini`配置文件内容为（label随便标注的，好区分是哪台）：

```bash
[LABEL]
dev_main = 172.17.0.4
dev1 = 172.17.0.3
dev2 = 172.17.0.2
[GLOBAL]
client1 = 172.17.0.4
client2 = 172.17.0.3
client3 = 172.17.0.2
```

在主机dev_main上安装好ansible跟python环境下的ansible

然后生成密钥：`ssh-keygen -t rsa -C “ansible_test” -f ./config/private_key.key`

运行的时候用的是用户运行，不是root运行，所以要配置一下用户：

```bash
sudo useradd -m -s /bin/bash damonzheng
sudo mkdir -p /home/damonzheng/.ssh
sudo cp <private_key_path> /home/damonzheng/.ssh/authorized_keys
sudo chown -R damonzheng:damonzheng/home/damonzheng/.ssh
sudo chmod 700 /home/damonzheng/.ssh
sudo chmod 600 /home/damonzheng/.ssh/authorized_keys
```

然后主机运行以下代码：

```python
import os
import configparser
import subprocess
import paramiko

proj_path = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config_path = os.path.join(proj_path, "config/config.ini")
config.read(config_path)

def get_info_from_host(host):
    private_key_path = os.path.join(proj_path, "config/private_key.key")
    remote_host = host
    remote_username = "damonzheng"
    remote_command = 'cd /tmp/ && ls -l'

    # 创建SSH客户端对象
    client = paramiko.SSHClient()

    # 加载本地私钥文件，并使用对端公钥进行身份验证
    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remote_host, username=remote_username, pkey=private_key)

    # 执行远程命令
    stdin, stdout, stderr = client.exec_command(remote_command)

    # 获取命令的输出
    output = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8')

    # 关闭SSH连接
    client.close()

    # 打印输出内容
    return output

def write_task_to_file(server_type):
    target_hosts = config["GLOBAL"][server_type].split(",")
    # 写txt文件
    txt_path = os.path.join(proj_path, f"ansible/{server_type}.txt")
    with open(txt_path, "w") as file:
        file.write("[ansible_hosts]\n")
        for i in range(len(target_hosts)):
            target_hosts[i] = target_hosts[i].strip()
            file.write(target_hosts[i])
    # 写任务执行文件
    sh_path = os.path.join(proj_path, f"ansible/update_{server_type}.sh")
    with open(sh_path, "w") as file:
        file.write("#!/bin/bash\n")
        file.write("cd /tmp\n")
        file.write("ls -l")
    # 写yml文件
    yml_path = os.path.join(proj_path, f"ansible/deploy_{server_type}.yml")
    with open(yml_path, "w") as file:
        cont = f"""
---
- hosts: "{{{{ TargetHost }}}}"
  remote_user: "{{{{ User }}}}"
  gather_facts: no
  become: yes
  tasks:
    - name: copy start scripts
      copy:
        src: "/mnt/d/code/test/ansible_test/ansible/update_{server_type}.sh"
        dest: "/tmp/"
        mode: 0755
        force: yes
        backup: no
    - name: run start scripts
      shell: bash /tmp/update_{server_type}.sh
"""
        file.write(cont)

    # 写分发文件
    deploy_path = os.path.join(proj_path, f"ansible/ansible_playbook_{server_type}.sh")
    with open(deploy_path, "w") as file:
        file.write(f"pwd={proj_path}\n")
        file.write("cd ${pwd}\n")
        file.write(f'ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook ${{pwd}}/ansible/deploy_{server_type}.yml --private-key ${{pwd}}/config/private_key.key --extra-vars "TargetHost=ansible_hosts User=damonzheng" -i ${{pwd}}/ansible/{server_type}.txt')

def deploy_task(server_type):
    deploy_file_path = os.path.join(proj_path, f"ansible/ansible_playbook_{server_type}.sh")
    command = f"bash {deploy_file_path}"
    result = subprocess.run(command, shell=True)
    return result

def main():
    # 获取信息
    for server_type in ["client1", "client2", "client3"]:
        hosts = config["GLOBAL"][server_type].split(",")
        for host in hosts:
            output = get_info_from_host(host)
            print(f"debug damonzheng, {host} result is:\n{output}")
    # 发送执行命令
    for server_type in ["client1", "client2", "client3"]:
        write_task_to_file(server_type)
        result = deploy_task(server_type)
        print(f"debug damonzheng, the result of {server_type} is:")
        print(f"result:{result}")
    pass

if __name__ == "__main__":
    main()

```