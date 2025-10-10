from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch


def load_model_config(model_name="bert-base-chinese"):
    """加载模型配置"""
    print("1. 模型配置管理...")
    config = AutoConfig.from_pretrained(model_name)
    print(f"模型配置类型: {type(config)}")
    print(f"隐藏层大小: {config.hidden_size}")
    print(f"层数: {config.num_hidden_layers}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"词汇表大小: {config.vocab_size}\n")
    return config


def create_model_from_config(config):
    """从配置创建模型框架（不加载权重）"""
    print("2. 从配置创建模型框架...")
    model = AutoModel.from_config(config)
    print(f"模型类型: {type(model)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    return model


def load_pretrained_model(model_name="bert-base-chinese"):
    """直接加载预训练模型（配置+权重）"""
    print("3. 直接加载预训练模型...")
    model = AutoModel.from_pretrained(model_name)
    print(f"预训练模型加载完成！")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}\n")
    return model


def demonstrate_model_states(model):
    """演示模型状态管理"""
    print("4. 模型状态管理...")
    print(f"模型训练模式: {model.training}")
    model.eval()  # 设置为评估模式
    print(f"设置为评估模式后: {model.training}")
    model.train()  # 设置为训练模式
    print(f"设置为训练模式后: {model.training}\n")


def demonstrate_forward_pass(model, tokenizer, text="你好，世界！"):
    """演示模型前向传播"""
    print("5. 模型前向传播...")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():  # 推理时不计算梯度
        outputs = model(**inputs)
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"输出形状: {outputs.last_hidden_state.shape}")
        print(f"池化输出形状: {outputs.pooler_output.shape}\n")
    return inputs, outputs


def save_and_load_transformers_way(model, config, save_path="./saved_model"):
    """使用 Transformers 方式保存和加载模型"""
    print("6. Transformers 方式保存和加载...")
    
    # 保存模型
    model.save_pretrained(save_path)    # 会连同 config.json 一起保存，下面这行会重复，不过也无所谓
    config.save_pretrained(save_path)
    print(f"模型已保存到: {save_path}")
    
    # 从保存的路径加载模型
    loaded_model = AutoModel.from_pretrained(save_path)
    print(f"从保存路径加载模型成功！")
    print(f"加载的模型参数数量: {sum(p.numel() for p in loaded_model.parameters()):,}\n")
    return loaded_model


def save_and_load_pytorch_way(model, config, state_dict_path="./model_state_dict.pth"):
    """使用 PyTorch 标准方式保存和加载模型"""
    print("7. PyTorch 标准权重加载方法...")
    
    # 保存 state_dict
    torch.save(model.state_dict(), state_dict_path)
    print(f"State dict 已保存到: {state_dict_path}")
    
    # 创建新的模型实例（只加载框架）
    new_model = AutoModel.from_config(config)
    print(f"新模型参数数量（未加载权重）: {sum(p.numel() for p in new_model.parameters()):,}")
    
    # 加载 state_dict
    state_dict = torch.load(state_dict_path)
    new_model.load_state_dict(state_dict)
    print(f"使用 load_state_dict 加载权重成功！")
    print(f"加载后模型参数数量: {sum(p.numel() for p in new_model.parameters()):,}\n")
    
    return new_model, state_dict


def verify_models_identical(model1, model2):
    """验证两个模型在相同输入下输出是否一致（需权重完全相同）"""
    print("验证两个模型是否相同...")
    # 都设定为eval模式，不然Dropout会使输出有随机性。
    model1.eval()
    model2.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 1000, (1, 10))  # 随机输入
        output1 = model1(test_input)
        output2 = model2(test_input)
        is_identical = torch.allclose(output1.last_hidden_state, output2.last_hidden_state)
        print(f"两个模型输出是否相同: {is_identical}\n")
    return is_identical


def create_custom_model_config(base_model_name="bert-base-chinese", 
                              hidden_size=512, 
                              num_hidden_layers=6):
    """创建自定义模型配置"""
    print("8. 模型配置修改...")
    cfg = AutoConfig.from_pretrained(base_model_name)

    cfg.hidden_size = hidden_size
    # 选择一个能整除 hidden_size 的注意力头数（尽量接近原始）
    # 下面逻辑：优先尝试保留原值；否则回退为 hidden_size 的最大因子 <= 原值
    heads = getattr(cfg, "num_attention_heads", 12)
    if hidden_size % heads != 0:
        # 找一个合适的因子
        factors = [h for h in range(min(heads, hidden_size), 0, -1) if hidden_size % h == 0]
        cfg.num_attention_heads = factors[0] if factors else 1
    # 保持常见比例
    cfg.intermediate_size = 4 * cfg.hidden_size
    cfg.num_hidden_layers = num_hidden_layers

    custom_model = AutoModel.from_config(cfg)
    print(f"自定义模型参数数量: {sum(p.numel() for p in custom_model.parameters()):,}")
    base_params = sum(p.numel() for p in AutoModel.from_pretrained(base_model_name).parameters())
    print(f"原始模型参数数量: {base_params:,}\n")
    return custom_model, cfg


def demonstrate_device_management(model, inputs, device=None):
    """演示模型设备管理"""
    print("9. 模型设备管理...")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_on_device = model.to(device)
    print(f"模型已移动到: {next(model_on_device.parameters()).device}")
    
    # 将输入也移动到相同设备
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model_on_device(**inputs_on_device)
        print(f"在 {device} 上推理成功！\n")
    
    return model_on_device, outputs


def main():
    """主函数：演示所有功能"""
    print("=== LLM 模型学习示例 ===\n")
    
    # 1. 加载配置
    config = load_model_config()
    
    # 2. 从配置创建模型框架
    model = create_model_from_config(config)
    
    # 3. 加载预训练模型
    pretrained_model = load_pretrained_model()
    
    # 4. 模型状态管理
    demonstrate_model_states(model)
    
    # 5. 前向传播
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    inputs, outputs = demonstrate_forward_pass(pretrained_model, tokenizer)
    
    # 6. Transformers 方式保存加载
    loaded_model = save_and_load_transformers_way(pretrained_model, config)
    
    # 7. PyTorch 标准方式保存加载
    new_model, state_dict = save_and_load_pytorch_way(pretrained_model, config)
    
    # 验证模型一致性
    verify_models_identical(loaded_model, new_model)
    
    # 8. 自定义配置
    custom_model, custom_config = create_custom_model_config()
    
    # 9. 设备管理
    demonstrate_device_management(pretrained_model, inputs)
    
    print("=== 模型学习完成！===")


if __name__ == "__main__":
    main()