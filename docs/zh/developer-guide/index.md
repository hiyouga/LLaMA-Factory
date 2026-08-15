# 开发者指南

开发者指南说明 v1 的运行路径和扩展点。阅读顺序建议为架构、Core、
Plugin。

## 架构与扩展机制

| 页面 | 内容 |
|------|------|
| [架构概览](architecture_overview.md) | 入口、引擎、训练器和并行层 |
| [BasePlugin](baseplugin_mechanism.md) | 注册、路由和严格参数解析 |

## Core

[Core](core/index.md) 说明数据加载、模型加载、样本渲染、批处理和训练循环。

## Plugin

[Plugin](plugins/index.md) 说明数据、模型和训练器插件，以及融合算子加速的
注册与调用方式。
