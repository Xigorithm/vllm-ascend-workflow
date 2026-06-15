# vllm-ascend-workflow

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

[vllm-ascend](https://github.com/vllm-project/vllm-ascend)（vLLM Ascend NPU 插件）AI 辅助开发工作流仓库。基于 [OpenCode](https://opencode.ai) Agent Skills 构建。

本仓库提供三类资源，加速 vllm-ascend 开发：

- **Skills** — 可复用的 AI Agent 技能，覆盖特性设计、PR 审查、测试、性能基准分析和文档生成
- **Docs** — 技术文档，涵盖特性深度分析、设计方案和工程经验
- **Sessions** — 开发会话记录，保留问题分析、方案设计和经验教训的完整过程

## 仓库结构

```
vllm-ascend-workflow/
├── skill/           # vllm-ascend 开发 Agent Skills
├── docs/            # 技术文档与经验笔记
└── sessions/        # 关键开发会话记录
```

## Skills（技能）

Agent Skills 定义专业工作流，将 AI 编码助手转变为 vllm-ascend 领域专家。每个 Skill 是一个自包含的 `.skill` 包。

| Skill | 说明 | 触发示例 |
|-------|------|----------|
| **feature-design** | 设计并实现 vllm-ascend 特性，产出代码实现和中文技术设计文档 | `帮我在 vllm-ascend 中实现 W4A16 量化` |
| **feature-learning** | 生成中文技术学习文档，包含 Mermaid 图表、GPU vs NPU 对比、源码走读 | `我想了解 vllm-ascend 的 ACL Graph` |
| **model-tutorial** | 生成特定模型在 vllm-ascend 上运行的中文技术教程文档 | `帮我生成 DeepSeek-V3 在昇腾上的模型教程` |
| **pr-description-creator** | 从 GitHub PR 代码变更生成 vllm-ascend 风格 PR 描述（Purpose / Test Plan / Test Result） | `帮我生成 vllm-ascend PR 1234 的描述` |
| **pr-summary-creator** | 获取并分析 vllm-ascend PR，包含代码变更分析、Mermaid 图表和风险评估 | `帮我分析 vllm-ascend PR 1234` |
| **rfc-creator** | 生成 vllm-ascend 风格 RFC 文档，用于重大架构变更 | `帮我生成一个 vllm-ascend RFC` |
| **test-creator** | 为 vllm-ascend 仓库生成单元测试和集成测试 | `帮我写 vllm-ascend 中 XXX 的测试用例` |
| **benchmark-result-summary** | 对比代码变更前后的 vllm-ascend 服务基准测试输出，含百分比分析 | *（粘贴基准测试输出并要求对比）* |

## Docs（文档）

技术文档，总结特性洞察、设计决策和工程经验。*（欢迎贡献）*

## Sessions（会话记录）

关键开发 Session 记录，供团队成员参考。每个 Session 保留开发者与 AI Agent 的完整交互过程——从问题分析、方案设计到代码变更和经验教训。*（欢迎贡献）*

## 使用前提

- 安装 [OpenCode](https://opencode.ai)
- 配置 LLM Provider API Key
- 本地克隆 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 仓库（用于源码走读）

## 安装

将 `skill/` 目录配置到 OpenCode 的 skills 路径中。在项目根目录创建或编辑 `opencode.json`：

```json
{
  "skills": {
    "paths": ["./skill"]
  }
}
```

## 贡献指南

- **新增 Skill**：在 `skill/` 下创建 `.skill` 文件，按 OpenCode Skill 规范编写
- **新增文档**：使用 Skill 生成文档，输出到对应的 `docs/` 子目录
- **分享 Session**：导出关键 Session 到 `sessions/`，使用描述性文件名

## 许可证

与 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 相同。
