# BasePlugin

`utils/plugin.py` 提供轻量级名称注册和路由。每个 `BasePlugin` 子类拥有
独立 `_registry`。

## 注册函数插件

```python
class OptimizerPlugin(BasePlugin):
    pass


@OptimizerPlugin("example").register()
def create_optimizer(model, optim_config):
    ...
```

调用时由名称解析实现：

```python
optimizer = OptimizerPlugin("example")(model, config)
```

## 注册类插件

一个插件需要多个操作时，可以注册实现类。例如分布式后端提供
`shard_model`、`save_model`、`save_checkpoint` 和 `load_checkpoint`：

```python
@DistributedPlugin("example").register()
class ExampleDistributed(BaseDistributed):
    @staticmethod
    def shard_model(model, dist_config, **kwargs):
        ...
```

`BasePlugin.__getattr__` 将方法访问转发到已注册对象。

## 解析插件参数

插件入口调用 `parse_params(config, ParamsDataclass)`：

- `config` 必须是 dict 或目标 dataclass；
- 未知字段会抛出 `ValueError`；
- 默认值只由参数 dataclass 定义；
- 每个插件入口拥有自己的参数结构。

```python
@dataclass
class ExampleParams:
    name: Literal["example"] = "example"
    enabled: bool = True


params = ExamplePlugin.parse_params(config, ExampleParams)
```

每个实现的专属字段分别记录在对应参数文档中。
