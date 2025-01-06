# wavrail

> 一个面向 Python 的流式语音识别工具包。
> A streaming-friendly speech recognition toolkit for Python.

## 时间线说明

本项目于 2026 年 9 月创建和验证。Git 中 2025 年至 2026 年 8 月的日期是本次生成的演示时间线，不代表那些日期已经开展的工作。每次开发提交均执行构建、测试与格式检查；未使用空提交。项目未宣称论文成果、预训练模型或真实语音数据集成绩。

## 概述

wavrail 提供了一套模块化的 Python 组件，用于构建可流式工作的自动语音识别（ASR）流程：

- **音频 I/O**：读取/写入 PCM16、PCM32 与 Float32 格式的 WAV 文件，严格校验头部，拒绝不支持的编码。
- **特征提取**：分帧、加窗、预加重、功率谱、Mel 滤波组、对数 Mel、CMVN 与一阶/二阶差分。
- **CTC 解码**：贪心解码与前缀束搜索，可控的束宽与剪枝策略。
- **同步发射与强制对齐**：流式发射循环与基于 Viterbi 的强制对齐。
- **指标与文本规范化**：中文 CER、英文 WER、编辑对齐与文本预处理。
- **流式管道**：有界前瞻的管道、回压、取消语义与实时因子统计。
- **可选声学模型**：基于 CPU-only PyTorch 的小型 GRU+CTC 模型，附带合成数据训练示例。

唯一的运行时依赖是 NumPy。PyTorch 是可选的 `torch` 扩展，用于训练小型声学模型；所有 torch 相关的测试在缺少该扩展时干净跳过。

## 安装

```bash
uv venv
uv pip install --python .venv/bin/python -e ".[dev]"
# 可选：安装 PyTorch（CPU 版本）
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -e ".[torch,dev]"
```

## 快速上手

```bash
# 生成一段合成音频，提取对数 Mel 特征
wavrail demo

# 离线评估 CLI 帮助
wavrail --help
```

## 开发

```bash
make build          # 构建 wheel
make test           # 快速测试（排除 slow）
make test-all       # 完整测试套件（CI 中运行）
make format-check   # 风格检查
make lint
make typecheck
make release        # 构建并验证 wheel 可安装
```

## 许可

MIT 许可证。详见 [LICENSE](LICENSE)。