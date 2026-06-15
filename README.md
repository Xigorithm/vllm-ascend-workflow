# vllm-ascend-workflow

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

AI-assisted development workflow repository for [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (vLLM Ascend NPU plugin). Powered by [OpenCode](https://opencode.ai) Agent Skills.

This repository provides three types of resources to accelerate vllm-ascend development:

- **Skills** — Reusable AI agent skills for feature design, PR review, testing, benchmarking, and documentation
- **Docs** — Technical documents covering feature deep-dives, design proposals, and engineering experience
- **Sessions** — Recorded development sessions capturing problem analysis, solution design, and lessons learned

## Repository Structure

```
vllm-ascend-workflow/
├── skill/           # Agent Skills for vllm-ascend development
├── docs/            # Technical documents and experience notes
└── sessions/        # Key development session records
```

## Skills

Agent Skills define specialized workflows that turn an AI coding agent into a vllm-ascend domain expert. Each skill is a self-contained `.skill` package.

| Skill | Description | Trigger Example |
|-------|-------------|-----------------|
| **feature-design** | Design and implement vllm-ascend features with code implementation and Chinese technical design documents | `帮我在 vllm-ascend 中实现 W4A16 量化` |
| **feature-learning** | Generate comprehensive Chinese technical learning documents with Mermaid diagrams, GPU-vs-NPU comparisons, and source code walkthroughs | `我想了解 vllm-ascend 的 ACL Graph` |
| **model-tutorial** | Generate Chinese technical tutorial documents for specific models running on vllm-ascend | `帮我生成 DeepSeek-V3 在昇腾上的模型教程` |
| **pr-description-creator** | Generate vllm-ascend style PR descriptions (Purpose / Test Plan / Test Result) from GitHub PR code changes | `帮我生成 vllm-ascend PR 1234 的描述` |
| **pr-summary-creator** | Fetch and analyze vllm-ascend PRs with code change analysis, Mermaid diagrams, and risk assessment | `帮我分析 vllm-ascend PR 1234` |
| **rfc-creator** | Generate vllm-ascend style RFC documents for major architectural changes | `帮我生成一个 vllm-ascend RFC` |
| **test-creator** | Generate unit tests and integration tests for the vllm-ascend repository | `帮我写 vllm-ascend 中 XXX 的测试用例` |
| **benchmark-result-summary** | Compare vllm-ascend serving benchmark outputs before and after a code change with percentage analysis | *(Paste benchmark output and ask to compare)* |

## Docs

Technical documents summarizing feature insights, design decisions, and engineering experience. *(Contributions welcome)*

## Sessions

Key development session records for team reference. Each session captures the full interaction between a developer and the AI agent — from problem analysis and solution design to code changes and lessons learned. *(Contributions welcome)*

## Prerequisites

- Install [OpenCode](https://opencode.ai)
- Configure an LLM Provider API Key
- Clone [vllm-ascend](https://github.com/vllm-project/vllm-ascend) locally (for source code walkthrough)

## Installation

Add the `skill/` directory to OpenCode's skills path. Create or edit `opencode.json` in the project root:

```json
{
  "skills": {
    "paths": ["./skill"]
  }
}
```

## Contributing

- **Add a skill**: Create a `.skill` file under `skill/` following OpenCode skill conventions
- **Add a document**: Use a skill to generate docs into the appropriate `docs/` subdirectory
- **Share a session**: Export a key session to `sessions/` with a descriptive filename

## License

Same as [vllm-ascend](https://github.com/vllm-project/vllm-ascend).
