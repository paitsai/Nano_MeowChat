from transformers import AutoTokenizer

# 填入你 tokenizer 所在的文件夹路径
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

print(f"<s> 的 ID: {tokenizer.convert_tokens_to_ids('<s>')}")
print(f"</s> 的 ID: {tokenizer.convert_tokens_to_ids('</s>')}")
print(f"<unk> 的 ID: {tokenizer.convert_tokens_to_ids('<unk>')}")
print(f"pad_token 的 ID: {tokenizer.pad_token_id}")
