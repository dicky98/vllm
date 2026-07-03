# LoRA 解析器插件 (LoRA Resolver Plugins)

此目录包含基于 `LoRAResolver` 框架构建的 vLLM LoRA 解析器插件。
它们会自动在指定的本地存储路径中发现并加载 LoRA 适配器，无需手动配置或重启服务器。

## 概述 (Overview)

LoRA 解析器插件提供了一种在运行时动态加载 LoRA 适配器的灵活方式。当 vLLM 收到一个尚未加载的 LoRA 适配器请求时，解析器插件将尝试在其配置的存储位置中查找并加载该适配器。这实现了：

- **动态 LoRA 加载**：按需加载适配器，无需重启服务器。
- **多种存储后端**：支持文件系统、S3 和自定义后端。内置的 `lora_filesystem_resolver` 需要本地存储路径，而内置的 `hf_hub_resolver` 则会从 Hugging Face Hub 拉取 LoRA 适配器，并以完全相同的方式处理。通常，可以实现自定义的解析器以从任何数据源获取。
- **自动发现**：与现有的 LoRA 工作流无缝集成。
- **可扩展部署**：在多个 vLLM 实例之间进行集中式的适配器管理。

## 前提条件 (Prerequisites)

在使用 LoRA 解析器插件之前，请确保已配置以下环境变量：

### 必填环境变量

1. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING`**：必须设置为 `true` 或 `1` 以启用动态 LoRA 加载。
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   ```

2. **`VLLM_PLUGINS`**：必须包含所需的解析器插件（以逗号分隔的列表）。
   ```bash
   export VLLM_PLUGINS=lora_filesystem_resolver
   ```

3. **`VLLM_LORA_RESOLVER_CACHE_DIR`**：对于文件系统解析器，必须将其设置为有效的目录路径。
   ```bash
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

### 可选环境变量

- **`VLLM_PLUGINS`**：如果未设置，将加载所有可用的插件。如果设置为空字符串，则不会加载任何插件。

## 可用的解析器 (Available Resolvers)

### lora_filesystem_resolver

文件系统解析器默认随 vLLM 安装，它允许从本地目录结构加载 LoRA 适配器。

#### 设置步骤

1. **创建 LoRA 适配器存储目录**：
   ```bash
   mkdir -p /path/to/lora/adapters
   ```

2. **设置环境变量**：
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   export VLLM_PLUGINS=lora_filesystem_resolver
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

3. **启动 vLLM 服务器**：
   您的基座模型可以是 `meta-llama/Llama-2-7b-hf`。请确保您在环境变量中设置了 Hugging Face Token，即 `export HF_TOKEN=xxx235`。
   ```bash
   vllm serve your-base-model \
       --enable-lora
   ```

#### 目录结构要求

文件系统解析器要求 LoRA 适配器组织成如下的结构：

```text
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (如果适用)
├── adapter2/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (如果适用)
└── ...
```

每个适配器目录必须包含：

- **`adapter_config.json`**：必需的配置文件，具有以下结构：
  ```json
  {
    "peft_type": "LORA",
    "base_model_name_or_path": "your-base-model-name",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "modules_to_save": null,
    "use_rslora": false,
    "use_dora": false
  }
  ```

- **`adapter_model.bin`**：LoRA 适配器的权重文件。

#### 使用示例

1. **准备您的 LoRA 适配器**：
   ```bash
   # 假设您的 LoRA 适配器位于 /tmp/my_lora_adapter
   cp -r /tmp/my_lora_adapter /path/to/lora/adapters/my_sql_adapter
   ```

2. **验证目录结构**：
   ```bash
   ls -la /path/to/lora/adapters/my_sql_adapter/
   # 应当显示：adapter_config.json, adapter_model.bin 等。
   ```

3. **使用该适配器发起请求**：
   ```bash
   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "my_sql_adapter",
           "prompt": "Generate a SQL query for:",
           "max_tokens": 50,
           "temperature": 0.1
       }'
   ```

#### 工作机制

1. 当 vLLM 收到一个对名为 `my_sql_adapter` 的 LoRA 适配器的请求时。
2. 文件系统解析器会检查 `/path/to/lora/adapters/my_sql_adapter/` 是否存在。
3. 如果找到，它会验证 `adapter_config.json` 配置文件。
4. 如果配置与基座模型匹配且有效，则会加载该适配器。
5. 请求随后使用新加载的适配器照常进行处理。
6. 该适配器会继续保持加载状态以响应后续请求。

## 高级配置

### 多个解析器 (Multiple Resolvers)

您可以配置多个解析器插件以从不同来源加载适配器：

以下是自定义解析器 `lora_s3_resolver`（您需要自行实现）的配置示例：

```bash
export VLLM_PLUGINS=lora_filesystem_resolver,lora_s3_resolver
```

所有列出的解析器都会被启用；在请求时，vLLM 会按顺序依次尝试，直到有一个解析成功为止。

### 自定义解析器的实现 (Custom Resolver Implementation)

要实现您自己的解析器插件：

1. **创建一个新的解析器类**：
   ```python
   from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
   from vllm.lora.request import LoRARequest
   
   class CustomResolver(LoRAResolver):
       async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
           # 在此编写您的自定义解析逻辑
           pass
   ```

2. **注册该解析器**：
   ```python
   def register_custom_resolver():
       resolver = CustomResolver()
       LoRAResolverRegistry.register_resolver("Custom Resolver", resolver)
   ```

## 问题排查 (Troubleshooting)

### 常见问题

1. **"VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory"**
   - 确保目录存在且可访问。
   - 检查该目录的文件系统权限。

2. **"LoRA adapter not found"**
   - 验证适配器目录的名称是否与请求的模型名称匹配。
   - 检查 `adapter_config.json` 是否存在且为有效的 JSON。
   - 确保该目录下存在 `adapter_model.bin` 文件。

3. **"Invalid adapter configuration"**
   - 验证 `peft_type` 是否被设置为 "LORA"。
   - 检查 `base_model_name_or_path` 是否与您的基座模型匹配。
   - 确保 `target_modules` 已经正确配置。

4. **"LoRA rank exceeds maximum"**
   - 检查 `adapter_config.json` 中的 `r` 值没有超过 `max_lora_rank` 设置。

### 调试建议 (Debugging Tips)

1. **启用调试日志**：
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

2. **验证环境变量**：
   ```bash
   echo $VLLM_ALLOW_RUNTIME_LORA_UPDATING
   echo $VLLM_PLUGINS
   echo $VLLM_LORA_RESOLVER_CACHE_DIR
   ```

3. **测试适配器配置**：
   ```bash
   python -c "
   import json
   with open('/path/to/lora/adapters/my_adapter/adapter_config.json') as f:
       config = json.load(f)
   print('Config valid:', config)
   "
   ```
