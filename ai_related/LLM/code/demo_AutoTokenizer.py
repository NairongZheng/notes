from transformers import AutoTokenizer


def main():
    print("=== LLM Tokenizer 学习示例 ===\n")

    # 1. 加载一个预训练的tokenizer
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-chinese",  # 使用中文BERT模型
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"Tokenizer加载完成！词汇表大小: {len(tokenizer)}")

    # 2. 设置pad_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Pad token ID: {tokenizer.pad_token_id}\n")

    # 3. 基本文本编码和解码
    print("2. 基本编码和解码...")
    text = "你好，世界！这是一个tokenizer学习示例。"
    print(f"原始文本: {text}")

    # 编码
    tokens = tokenizer.encode(text, add_special_tokens=True)
    print(f"Token IDs: {tokens}")

    # 解码
    decoded_text = tokenizer.decode(tokens)
    print(f"解码文本: {decoded_text}\n")

    # 4. 查看token详情
    print("3. Token详情...")
    tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    print(f"Input IDs: {tokenized['input_ids']}")
    print(f"Attention Mask: {tokenized['attention_mask']}")

    # 5. 添加特殊token（参考tmp.py中的代码）
    print("\n4. 添加特殊token...")
    special_tokens = ["<SPECIAL_START>", "<SPECIAL_END>", "<CUSTOM_TOKEN>"]
    num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"添加了 {num_new_tokens} 个新token")
    print(f"新的词汇表大小: {len(tokenizer)}")

    # 6. 使用新添加的特殊token
    print("\n5. 使用特殊token...")
    text_with_special = f"<SPECIAL_START>你好<SPECIAL_END>"
    tokens_with_special = tokenizer.encode(text_with_special, add_special_tokens=True)
    print(f"包含特殊token的文本: {text_with_special}")
    print(f"Token IDs: {tokens_with_special}")
    print(f"解码结果: {tokenizer.decode(tokens_with_special)}")

    # 7. 批量处理
    print("\n6. 批量处理...")
    texts = ["第一个句子", "第二个句子", "第三个句子"]
    batch_tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=20, return_tensors="pt"
    )
    print(f"批量处理结果:")
    print(f"Input IDs shape: {batch_tokens['input_ids'].shape}")
    print(f"Attention Mask shape: {batch_tokens['attention_mask'].shape}")

    print("\n=== 学习完成！===")


if __name__ == "__main__":
    main()
