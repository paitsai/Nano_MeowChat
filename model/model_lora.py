import torch
from torch import nn, optim


class LoRA(nn.Module):
    """修正后的LoRA模块（符合原论文设计）"""
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int = 1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha  # 控制LoRA贡献的缩放系数（通常设为rank）
        self.scaling = alpha / rank  # 预计算缩放因子
        
        # LoRA低秩矩阵：A（输入→秩）、B（秩→输出）
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化（原论文推荐）：A用高斯分布缩放，B初始化为0
        nn.init.normal_(self.A.weight, mean=0.0, std=1.0 / self.rank**0.5)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：A→B→缩放"""
        return self.B(self.A(x)) * self.scaling

import re  

def apply_lora(model: nn.Module, rank: int = 64, start_layer: int = 0, end_layer: int = 15) -> None:
    """给模型的指定线性层（含"proj"）添加LoRA，并冻结原模型参数"""
    device = next(model.parameters()).device  # 获取模型当前设备
    pattern = r"layers\.(\d+)"  

    for name, module in model.named_modules():
        # 仅对含"proj"的线性层应用LoRA（如q_proj、k_proj）
        # lora_item_list = ['q_proj', 'k_proj']
        lora_item_list = ['proj']
        apply = False
        if isinstance(module, nn.Linear) and "lora" not in name:
            for item in lora_item_list:
                if item in name:
                    apply = True
            if not apply:
                continue
            
            match = re.search(pattern, name)
            if not match:
                continue
            layer_num = int(match.group(1))
            if layer_num < start_layer or layer_num > end_layer:
                continue
            # 初始化LoRA模块（alpha=rank → scaling=1）
            lora_module = LoRA(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=rank
            ).to(device)
            
            # 绑定LoRA到原模块
            setattr(module, "lora", lora_module)
            original_forward = module.forward
            
            # 定义新的前向传播：原输出 + LoRA输出
            def forward_with_lora(x: torch.Tensor, orig_forward=original_forward, lora=lora_module):
                orig_out = orig_forward(x)
                lora_out = lora(x)
                return orig_out + lora_out
            
            module.forward = forward_with_lora
            
            # 冻结原模块参数（仅训练LoRA）
            for param in module.parameters():
                param.requires_grad = False


def save_lora(model: nn.Module, save_path: str) -> None:
    """保存LoRA参数（仅保存新增的LoRA权重）"""
    # 处理DDP包装的模型（获取原始模型）
    raw_model = getattr(model, "_orig_mod", model)
    lora_state_dict = {}
    
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # 保存LoRA参数，key格式："模块名.lora.参数名"
            for k, v in module.lora.state_dict().items():
                lora_state_dict[f"{name}.lora.{k}"] = v
    
    torch.save(lora_state_dict, save_path)
    print(f"LoRA权重已保存到：{save_path}")


def load_lora(model: nn.Module, load_path: str) -> None:
    """加载LoRA参数到模型"""
    device = next(model.parameters()).device
    lora_state_dict = torch.load(load_path, map_location=device)
    print(f"加载的LoRA权重键：{list(lora_state_dict.keys())}")  # 打印加载的LoRA权重键，便于调试
   
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # 提取当前模块对应的LoRA参数
            print(name)
            module_lora_keys = [k for k in lora_state_dict.keys() if k.startswith(f"module.{name}.lora.")]
            print(f"模块{name}的LoRA键：{module_lora_keys}")  # 打印当前模块的LoRA键，便于调试
            module_lora_state = {
                k.replace(f"module.{name}.lora.", ""): lora_state_dict[k] 
                for k in module_lora_keys
            }
            # 加载到LoRA模块
            if len(module_lora_state)>0:
                module.lora.load_state_dict(module_lora_state)
                module.lora.to(device)
    
    print(f"LoRA权重已从：{load_path} 加载")


# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    # 示例：创建一个简单的线性层模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(512, 2048)  # 输入512，输出2048
        
        def forward(self, x):
            return self.q_proj(x)
    
    # 1. 初始化模型
    model = SimpleModel()
    print("原模型参数：", list(model.parameters())[0].shape)  # (2048,512)
    
    # 2. 应用LoRA（rank=8）
    apply_lora(model, rank=8)
    print("应用LoRA后，q_proj是否有lora属性：", hasattr(model.q_proj, "lora"))  # True
    
    # 3. 获取可训练参数（仅LoRA）
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print("可训练参数数量：", len(lora_params))  # 2（A和B的权重）
    
    # 4. 保存LoRA权重
    save_lora(model, "lora_weights.pth")
    
    # 5. 加载LoRA权重到新模型
    new_model = SimpleModel()
    apply_lora(new_model, rank=8)  # 先应用LoRA结构
    load_lora(new_model, "lora_weights.pth")
    
    print("加载完成！")
