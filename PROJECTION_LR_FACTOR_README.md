# Projection层学习率因子设置功能

## 功能概述

本功能允许在使用`--add_multimodal_projections`参数时，自动为所有模态的projection层设置4倍学习率。这有助于加速多模态projection层的训练收敛。

## 实现原理

### 修改的文件
- `solver/build.py`: 在`build_optimizer`函数中添加了projection层学习率因子的逻辑

### 核心逻辑

```python
# 检查是否启用了多模态projection层
add_projections = getattr(args, 'add_multimodal_projections', False) or getattr(args, 'add_multimodal_layers', False)

# 为projection层设置学习率因子
if add_projections and ("visual_projection" in key or "text_projection" in key or "text_filip_projection" in key or "modality_visual_projections" in key):
    lr = args.lr * 4.0  # 提高4倍学习率
    print(f'Setting 4x learning rate for projection layer: {key}')
```

### 影响的参数层

当启用`--add_multimodal_projections`或`--add_multimodal_layers`参数时，以下参数层将获得4倍学习率：

1. **text_projection**: FGCLIPModel中定义的文本projection层  
2. **modality_visual_projections**: 多模态视觉projection层字典中的所有层
   - `modality_visual_projections.vis`
   - `modality_visual_projections.cp` 
   - `modality_visual_projections.sk`
   - `modality_visual_projections.nir`

注意：这个功能专门针对`model/fgclip.py`中定义的特定projection层，不会影响其他模型组件中可能包含类似名称的参数。

## 使用方法

### 1. 基本使用

在训练脚本中添加`--add_multimodal_projections`参数：

```bash
python train.py \
  --lr 2.4e-5 \
  --add_multimodal_projections \
  [其他参数...]
```

这将使：
- 基础学习率: `2.4e-5`
- Projection层学习率: `9.6e-5` (4倍)

### 2. 使用多模态层参数

```bash
python train.py \
  --lr 2.4e-5 \
  --add_multimodal_layers \
  [其他参数...]
```

`--add_multimodal_layers`等价于同时使用`--add_multimodal_embeddings`和`--add_multimodal_projections`。

### 3. 示例训练脚本

参考提供的示例脚本：
```bash
./script/train/fgclip_large_multiloss_train_with_projection_lr.sh
```

## 参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--add_multimodal_projections` | 启用多模态projection层并设置4倍学习率 | False |
| `--add_multimodal_layers` | 启用多模态embedding和projection层 | False |
| `--lr` | 基础学习率 | 1e-5 |

## 日志输出

启用此功能时，你会在训练日志中看到类似输出：

```
Using 5.0 times learning rate for random init module 
Using 4x learning rate for projection layers when add_multimodal_projections is enabled
Setting 4x learning rate for projection layer: base_model.visual_projection.weight
Setting 4x learning rate for projection layer: base_model.text_projection.weight
Setting 4x learning rate for projection layer: base_model.text_filip_projection.weight
Setting 4x learning rate for projection layer: base_model.modality_visual_projections.vis.weight
Setting 4x learning rate for projection layer: base_model.modality_visual_projections.cp.weight
Setting 4x learning rate for projection layer: base_model.modality_visual_projections.sk.weight
Setting 4x learning rate for projection layer: base_model.modality_visual_projections.nir.weight
```

## 测试功能

运行测试脚本验证功能是否正常工作：

```bash
python test_projection_lr.py
```

## 注意事项

1. **参数优先级**: projection层的4倍学习率设置在其他学习率因子（如bias_lr_factor）之后应用

2. **兼容性**: 此功能与现有的学习率因子设置兼容，不会影响其他参数的学习率

3. **模型要求**: 需要使用FGCLIPModel，并且模型需要有相应的projection层

4. **训练阶段**: 建议在多模态训练的微调阶段使用此功能

## 技术细节

### 学习率设置顺序

```python
# 1. 基础学习率
lr = args.lr

# 2. 特殊模块学习率因子
if "cross" in key:
    lr = args.lr * args.lr_factor
if "classifier" in key or "mlm_head" in key:
    lr = args.lr * args.lr_factor

# 3. Bias学习率因子
if "bias" in key:
    lr = args.lr * args.bias_lr_factor

# 4. Projection层学习率因子 (新增)
if add_projections and (projection_layer_condition):
    lr = args.lr * 4.0
```

### 参数检测逻辑

```python
add_projections = getattr(args, 'add_multimodal_projections', False) or getattr(args, 'add_multimodal_layers', False)
```

这确保了无论使用`--add_multimodal_projections`还是`--add_multimodal_layers`参数，都能正确启用projection层的学习率因子。

## 常见问题

**Q: 为什么选择4倍学习率？**
A: 4倍学习率是一个经验值，有助于加速新添加的projection层的收敛，同时不会过于激进导致训练不稳定。

**Q: 可以修改学习率倍数吗？**
A: 可以，在`solver/build.py`中修改`lr = args.lr * 4.0`这一行的倍数值。

**Q: 这个功能会影响预训练权重吗？**
A: 不会，这只影响训练时的学习率，不会改变模型结构或预训练权重的加载。