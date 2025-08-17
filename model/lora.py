
import torch
import math
# import torch.nn as nn

class VisionLoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_dim = lora_dim

        # 保存原始 forward，不受 apply_to 影响
        self._original_forward = org_module.forward  
        self.org_module = org_module
        # 为四个模态分别定义 lora_down 和 lora_up
        self.lora_down_sk = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up_sk = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
        self.lora_down_cp = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up_cp = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
        self.lora_down_nir = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up_nir = torch.nn.Linear(self.lora_dim, out_dim, bias=False)
        self.lora_down_vis = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
        self.lora_up_vis = torch.nn.Linear(self.lora_dim, out_dim, bias=False)


        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha)) 
        # 对每组 lora_down 和 lora_up 进行初始化

        torch.nn.init.zeros_(self.lora_up_sk.weight)
        torch.nn.init.zeros_(self.lora_up_cp.weight)
        torch.nn.init.zeros_(self.lora_up_nir.weight)
        torch.nn.init.zeros_(self.lora_up_vis.weight)

        fan_in = self.lora_down_sk.weight.size(1)  # 输入维度 手动进行kaiming_uniform计算
        a = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + a ** 2))
        bound = gain * math.sqrt(3.0 / fan_in) 
        self.lora_down_sk.weight.data.uniform_(-bound, bound)
        self.lora_down_cp.weight.data.uniform_(-bound, bound)
        self.lora_down_nir.weight.data.uniform_(-bound, bound)
        self.lora_down_vis.weight.data.uniform_(-bound, bound)
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
    
    @property
    def original_forward(self):
        return self._original_forward

    @original_forward.setter
    def original_forward(self, value):
        raise AttributeError("Cannot modify '_original_forward' once it is set.")

    def apply_to(self,modality):
        device = "cuda"
        if modality == "sk":
            self.lora_down = self.lora_down_sk
            self.lora_up = self.lora_up_sk
        elif modality == "cp":
            self.lora_down = self.lora_down_cp
            self.lora_up = self.lora_up_cp
        elif modality == "nir":
            self.lora_down = self.lora_down_nir
            self.lora_up = self.lora_up_nir
        elif modality == "vis":
            self.lora_down = self.lora_down_vis
            self.lora_up = self.lora_up_vis
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        self.lora_down = self.lora_down.to(device)
        self.lora_up = self.lora_up.to(device)
        self.alpha = self.alpha.to(device)
        # 每次 apply_to 都基于 _original_forward
        self.org_forward = self._original_forward
        self.org_module.forward = self.forward


    def forward(self, x):
        org_forwarded = self.org_forward(x)
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask
            # scaling for rank dropout: treat as if the rank is changed
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale
        lx = self.lora_up(lx)
        return org_forwarded + lx * self.multiplier * scale
    



def inject_vision_lora(model, lora_dim=4, alpha=1.0, dropout=None, rank_dropout=None, module_dropout=None):
    """
    递归遍历模型，将所有 Linear 层添加 LoRA。
    返回一个 lora_modules 列表，便于之后管理。
    """
    lora_modules = []
    def _inject(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (torch.nn.Linear)):
                 if  name == 'q_proj' or name == 'v_proj' :
                    lora_name = full_name.replace('.', '_') + '_lora'
                    lora = VisionLoRAModule(
                        lora_name=lora_name,
                        org_module=child,
                        lora_dim=lora_dim,
                        alpha=alpha,
                        dropout=dropout,
                        rank_dropout=rank_dropout,
                        module_dropout=module_dropout,
                    )
                    # lora.apply_to()  # 替换 forward 方法
                    lora_modules.append(lora)
            else:
                _inject(child, full_name)
    _inject(model)
    return lora_modules