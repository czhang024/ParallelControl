# StateFT / ParallelControlModel Save and Load Guide

本文档说明如何保存和加载 `ParallelControlModel` / `ParallelControlv2Model` 的 adapter 权重。

## 架构说明

`ParallelControlModel` 使用 hooks 机制将 adapter 模块（`shortcut_modules`）连接到基础模型上：

1. **`build_edges`**: 在模型初始化时调用，用于注册 forward hooks
2. **`shortcut_modules`**: 存储 adapter 模块（如 LoRAsideLayer）
3. **`dag_hook_handles`**: 存储 hook handles，用于管理 hooks 的生命周期

在保存和加载时，需要同时处理：
- Adapter 权重（可序列化）
- Hooks（需要重新注册）

## 保存 Adapter

### 基本保存

```python
from peft.tuners.stateft import ParallelControlv2Model, StateFTLorav2Config

# 创建模型和 adapter
config = StateFTLorav2Config(
    target_modules=["q_proj", "v_proj"],
    r=8,
    lora_alpha=16,
    in_features=768,
    out_features=768,
)

peft_model = ParallelControlv2Model(base_model, {"default": config}, "default")

# ... 训练 ...

# 保存 adapter（保存权重和配置）
peft_model.save_pretrained("./my_adapter")
```

### 保存参数

- `save_directory`: 保存目录路径
- `safe_serialization`: 是否使用 safetensors 格式（默认 True）
- `selected_adapters`: 要保存的 adapter 列表（默认保存所有）
- `is_main_process`: 是否为主进程（用于分布式训练）

```python
# 保存指定的 adapters
peft_model.save_pretrained(
    "./my_adapter",
    safe_serialization=True,
    selected_adapters=["default", "adapter2"],
)
```

## 加载 Adapter

### 方法 1: 从预训练加载（推荐）

这是最简单的方式，会自动处理配置加载、模块创建、hooks 注册和权重加载：

```python
from peft.tuners.stateft import ParallelControlv2Model

# 加载预训练的 adapter
peft_model = ParallelControlv2Model.from_pretrained(
    base_model,
    "./my_adapter",
    adapter_name="default",
    is_trainable=False,  # 设置为 True 如果需要继续训练
)
```

### 方法 2: 加载额外的 Adapter 到已有模型

`load_adapter` 方法会自动检测 adapter 是否存在，如果不存在会：
1. 加载配置
2. 创建 adapter edges
3. 注册 hooks
4. 加载权重

```python
# 加载一个新的 adapter 到已有模型
peft_model.load_adapter(
    "./another_adapter",
    adapter_name="adapter2",
    is_trainable=False,
)

# 切换 adapter
peft_model.set_adapter("adapter2")
```

### 方法 3: 只更新已有 Adapter 的权重

如果 adapter 已经存在（已注册 hooks），`load_adapter` 只会更新权重：

```python
# 更新现有 adapter 的权重
peft_model.load_adapter(
    "./updated_weights",
    adapter_name="default",  # 已存在的 adapter
    is_trainable=False,
)
```

## 加载多个 Adapters

```python
# 创建模型
peft_model = ParallelControlv2Model(base_model, {"adapter1": config1}, "adapter1")

# 加载第二个 adapter（会自动创建 edges 和注册 hooks）
peft_model.load_adapter("./adapter2", adapter_name="adapter2")

# 切换 adapter
peft_model.set_adapter("adapter1")  # 或 "adapter2"
```

## 从 Hugging Face Hub 加载

```python
# 从 Hub 加载
peft_model = ParallelControlv2Model.from_pretrained(
    base_model,
    "username/my-adapter-repo",
    adapter_name="default",
)
```

## State Dict 操作

### 获取 Adapter State Dict

```python
# 获取特定 adapter 的 state dict
state_dict = peft_model.get_adapter_state_dict("default")
```

### 设置 Adapter State Dict

```python
# 直接设置 state dict（adapter 必须已存在）
peft_model.set_adapter_state_dict(state_dict, adapter_name="default")
```

## 内部机制

### Save 流程
1. `save_pretrained` 调用 `get_adapter_state_dict` 获取权重
2. 保存权重到文件（safetensors 或 pickle）
3. 保存配置到 `adapter_config.json`

### Load 流程
1. `load_adapter` 检查 adapter 是否存在（`_check_adapter_exists`）
2. 如果不存在：
   - 加载配置
   - 调用 `_create_adapter_edges` 创建 edges
   - 调用 `_build_adapter_hooks` 注册 hooks
3. 加载权重文件
4. 调用 `set_adapter_state_dict` 更新模块权重
5. 设置 `requires_grad` 状态

## 注意事项

1. **Hooks 重建**: 加载新 adapter 时会自动重建 hooks
2. **配置保存**: `save_pretrained` 会同时保存配置文件 (`adapter_config.json`) 和权重文件
3. **Safetensors 格式**: 推荐使用 safetensors 格式保存，更安全且加载更快
4. **多 Adapter**: 每个 adapter 会保存到独立的子目录中
5. **分布式训练**: 在分布式训练中，只有主进程需要保存 (`is_main_process=True`)
6. **设备管理**: 加载时会自动将权重移动到正确的设备
7. **继承关系**: `ParallelControlv2Model` 继承自 `ParallelControlModel`，后者继承自 `BaseDAGControlModel`，子类通过重写 `create_lora_edges` 和 `_create_adapter_edges` 方法来定制 edge 创建逻辑
