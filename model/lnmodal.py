import torch
import math

class VisionLayerNormModule(torch.nn.Module):
    """
    Replaces LayerNorm with modality-specific LayerNorm.
    """

    def __init__(self, lnmodal_name, org_module: torch.nn.Module):
        super().__init__()
        self.lnmodal_name = lnmodal_name
        # 保存原始 forward，不受 apply_to 影响
        self._original_forward = org_module.forward  
        self.org_module = org_module

        # 为四个模态分别定义 LayerNorm
        self.layernorm_sk = torch.nn.LayerNorm(org_module.normalized_shape, eps=org_module.eps)
        self.layernorm_cp = torch.nn.LayerNorm(org_module.normalized_shape, eps=org_module.eps)
        self.layernorm_nir = torch.nn.LayerNorm(org_module.normalized_shape, eps=org_module.eps)
        self.layernorm_vis = torch.nn.LayerNorm(org_module.normalized_shape, eps=org_module.eps)
    
    def copy_params_from_org_module(self):
        """
        Copies the parameters (weight, bias) from the original LayerNorm module to
        the corresponding modality-specific LayerNorm.
        """
        self.layernorm_sk.weight.data = self.org_module.weight.data.clone()
        self.layernorm_cp.weight.data = self.org_module.weight.data.clone()
        self.layernorm_nir.weight.data = self.org_module.weight.data.clone()
        self.layernorm_vis.weight.data = self.org_module.weight.data.clone()

        if self.org_module.bias is not None:
            self.layernorm_sk.bias.data = self.org_module.bias.data.clone()
            self.layernorm_cp.bias.data = self.org_module.bias.data.clone()
            self.layernorm_nir.bias.data = self.org_module.bias.data.clone()
            self.layernorm_vis.bias.data = self.org_module.bias.data.clone()


    @property
    def original_forward(self):
        return self._original_forward

    @original_forward.setter
    def original_forward(self, value):
        raise AttributeError("Cannot modify '_original_forward' once it is set.")

    def apply_to(self, modality):
        device = "cuda"
        if modality == "sk":
            self.layernorm = self.layernorm_sk
        elif modality == "cp":
            self.layernorm = self.layernorm_cp
        elif modality == "nir":
            self.layernorm = self.layernorm_nir
        elif modality == "vis":
            self.layernorm = self.layernorm_vis
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        self.layernorm = self.layernorm.to(device)
        # 每次 apply_to 都基于 _original_forward
        self.org_forward = self._original_forward
        self.org_module.forward = self.forward

    def forward(self, x):
        # print(self.org_module.weight)
        # print(self.layernorm.weight)
        # print(self.layernorm_vis.weight)
        # if self.layernorm.weight is self.layernorm_vis.weight:
        #     print("They are the same object.")
        # else:
        #     print("They are different objects.")
        return self.layernorm(x)


def inject_vision_layernorm(model):
    """
    递归遍历模型，将所有 LayerNorm 层添加 模态对应的ln。
    返回一个 layernorm_modules 列表，便于之后管理。
    """
    layernorm_modules = []

    def _inject(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # if isinstance(child, (torch.nn.LayerNorm)) and ('pre_layrnorm' in full_name or 'post_layernorm' in full_name):
            if isinstance(child, (torch.nn.LayerNorm)):
                # Replace LayerNorm with modality-specific LayerNorm
                lnmodal_name = full_name.replace('.', '_') + '_modality'
                layernorm_module = VisionLayerNormModule(
                    lnmodal_name = lnmodal_name,
                    org_module=child,
                )
                layernorm_modules.append(layernorm_module)
            else:
                _inject(child, full_name)

    _inject(model)
    return layernorm_modules
