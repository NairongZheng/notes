

hugging face 提供 api 访问：[https://huggingface.co/docs/hub/api](https://huggingface.co/docs/hub/api)


api：

```shell
# models 或者 datasets api:
# https://huggingface.co/docs/hub/api

# datasets-server api:
# https://datasets-server.huggingface.co
```


示例代码：

```python
import json
import time
import requests
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

hf_api = HfApi()
HF_TOKEN = "hf_xxx"


def search_models(query: str, limit: int = 2):
    """搜索模型"""
    # 1. 使用requests搜索模型
    url = "https://huggingface.co/api/models"
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
    }
    response = requests.get(url, params=params)
    request_models_list = response.json()
    
    # 2. 使用huggingface_hub搜索模型
    models = hf_api.list_models(search=query, limit=limit, sort="downloads")
    huggingface_hub_models_list = [m for m in models]
    return request_models_list, huggingface_hub_models_list


def search_datasets(query: str, limit: int = 2):
    """搜索数据集"""
    # 1. 使用requests搜索数据集
    url = "https://huggingface.co/api/datasets"
    params = {
        "search": query,
        "limit": limit,
        "sort": "downloads",
    }
    response = requests.get(url, params=params)
    request_datasets_list = response.json()
    
    # 2. 使用huggingface_hub搜索数据集
    datasets = hf_api.list_datasets(search=query, limit=limit, sort="downloads")
    huggingface_hub_datasets_list = [d for d in datasets]
    return request_datasets_list, huggingface_hub_datasets_list


def get_model_info(model_id: str):
    """获取模型信息"""
    # 1. 使用requests获取模型信息
    url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(url)
    request_model_info = response.json()
    
    # 2. 使用huggingface_hub获取模型信息
    huggingface_hub_model_info = hf_api.model_info(model_id)
    return request_model_info, huggingface_hub_model_info


def get_dataset_info(dataset_id: str):
    """获取数据集信息"""
    # 1. 使用requests获取数据集信息
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = requests.get(url)
    request_dataset_info = response.json()
    
    # 2. 使用huggingface_hub获取数据集信息
    huggingface_hub_dataset_info = hf_api.dataset_info(dataset_id)
    return request_dataset_info, huggingface_hub_dataset_info


def get_dataset_more_info(dataset_id: str):
    """获取数据集更多信息"""
    params = {"dataset": dataset_id}
    # 1. 获取数据集的 schema 信息
    schema_url = f"https://datasets-server.huggingface.co/info"
    schema_response = requests.get(schema_url, params=params)
    request_dataset_schema_info = schema_response.json()
    # 2. 获取数据集的 split 信息
    splits_url = f"https://datasets-server.huggingface.co/splits"
    splits_response = requests.get(splits_url, params=params)
    request_dataset_splits_info = splits_response.json()
    # 3. 获取数据集的 parquet 信息
    parquet_url = f"https://datasets-server.huggingface.co/parquet"
    parquet_response = requests.get(parquet_url, params=params)
    request_dataset_parquet_info = parquet_response.json()
    # 4. 获取数据集的内容
    params = {
        "dataset": dataset_id,
        "config": "default",
        "split": "train",
        "limit": 1,
    }
    content_url = f"https://datasets-server.huggingface.co/rows"
    content_response = requests.get(content_url, params=params)
    request_dataset_content_info = content_response.json()
    return request_dataset_schema_info, request_dataset_splits_info, request_dataset_parquet_info, request_dataset_content_info


def download_hf_repo(repo_id: str, repo_type: str, local_dir: str = "./download"):
    """下载huggingface仓库"""
    if not repo_id:
        return "repo_id is required"
    max_retries = 3
    delay = 1
    res_str = ""
    for attempt in range(max_retries):
        try:
            res_str = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                token=HF_TOKEN
            )
            break
        except HfHubHTTPError as e:
            if e.response.status_code in [401, 403]:
                res_str = "HfHubHTTPError"
        except Exception as e:
            res_str = "error"
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return res_str


def main():
    request_models_list, huggingface_hub_models_list = search_models("gpt")
    request_datasets_list, huggingface_hub_datasets_list = search_datasets("jdaddyalbs/playwright-mcp-toolcalling")
    request_model_info, huggingface_hub_model_info = get_model_info(huggingface_hub_models_list[0].id)
    request_dataset_info, huggingface_hub_dataset_info = get_dataset_info(huggingface_hub_datasets_list[0].id)
    dataset_more_info = get_dataset_more_info(huggingface_hub_datasets_list[0].id)
    
    # 下载 hf 仓库
    download_res = download_hf_repo("inclusionAI/ASearcher-train-data", "dataset", "/tmp/ASearcher-train-data")


if __name__ == "__main__":
    main()
```