# vllm-ascend-workflow

<div align='left'>
  <a href="https://github.com/vllm-project/vllm-ascend"><img src="https://img.shields.io/badge/vllm--ascend-plugin-blue.svg"></a>
  <img src="https://img.shields.io/badge/License-Apache--2.0-turquoise.svg">
  <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">
  <img src="https://img.shields.io/badge/Contributions-welcome-blue">
  <img src="https://img.shields.io/badge/OpenCode-Agent%20Skills-orange.svg">
  <img src="https://img.shields.io/badge/Ascend-NPU-red.svg">
  <img src="https://img.shields.io/github/watchers/Xigorithm/vllm-ascend-workflow?color=9cc">
  <img src="https://img.shields.io/github/forks/Xigorithm/vllm-ascend-workflow.svg?style=social">
  <img src="https://img.shields.io/github/stars/Xigorithm/vllm-ascend-workflow.svg?style=social">
</div>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

## 📌 Overview

[vllm-ascend-workflow](https://github.com/Xigorithm/vllm-ascend-workflow) is an **AI-assisted development workflow repository** for [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (vLLM Ascend NPU plugin), powered by [OpenCode](https://opencode.ai) Agent Skills.

This repository covers the full vllm-ascend development pipeline — **Learning → RFC → Develop → Test → PR Review → BenchMark** — mapping each stage to specialized Agent Skills and repository directories:

```mermaid
graph TB
    subgraph "🔄 Development Pipeline"
        direction LR
        P1["🔍 Learning"] --> P2["📝 RFC"]
        P2 --> P3["🧑‍💻 Develop"]
        P3 --> P4["🧪 Test"]
        P4 --> P5["🔎 PR Review"]
        P5 --> P6["📊 BenchMark"]
    end

    subgraph "🧠 Agent Skills"
        direction LR
        S1["feature-learning<br>model-tutorial"] ~~~ S2["rfc-creator"]
        S2 ~~~ S3["feature-design<br>pr-description-creator"]
        S3 ~~~ S4["test-creator"]
        S4 ~~~ S5["pr-summary-creator"]
        S5 ~~~ S6["benchmark-result-summary"]
    end

    subgraph "📁 Repository"
        direction LR
        D1["skill/ — Skill definitions"]
        D2["docs/ — Technical documents & RFCs"]
        D3["sessions/ — Development session records"]
    end

    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4
    P5 --> S5
    P6 --> S6

    S1 --> D1
    S2 --> D1
    S3 --> D1
    S4 --> D1
    S5 --> D1
    S6 --> D1

    S1 --> D2
    S2 --> D2
    S3 --> D2

    S3 --> D3
    S5 --> D3

    style P1 fill:#e1f5fe
    style P2 fill:#e1f5fe
    style P3 fill:#e8f5e9
    style P4 fill:#fff3e0
    style P5 fill:#fff3e0
    style P6 fill:#fce4ec
    style D1 fill:#f3e5f5
    style D2 fill:#f3e5f5
    style D3 fill:#f3e5f5
```

---

## 🏗️ Repository Structure

```
vllm-ascend-workflow/
├── skill/                  # Agent Skills — reusable AI coding skills
│   ├── feature-design.skill
│   ├── feature-learning.skill
│   ├── model-tutorial.skill
│   ├── pr-description-creator.skill
│   ├── pr-summary-creator.skill
│   ├── rfc-creator.skill
│   ├── test-creator.skill
│   └── benchmark-result-summary.skill
├── docs/                   # Technical documents — feature analysis & design proposals
│   ├── quantization.md                    # Quantization feature deep-dive
│   ├── design-quantization-refactor.md    # Quantization refactoring design document
│   └── rfc-quantization-code-refactoring.md  # Quantization refactoring RFC
└── sessions/               # Session records — full development interaction logs
    └── session-quantization-refactor.md   # Quantization refactoring session
```

---

## 🚀 Agent Skills

Agent Skills define specialized workflows that turn an AI coding assistant into a vllm-ascend domain expert. Each skill is a self-contained `.skill` package containing prompt templates, reference materials, and output specifications.

### 🔍 Code Learning

| Skill | Description | Trigger Example |
|:------|:------------|:----------------|
| **feature-learning** | Generate comprehensive Chinese technical learning documents with Mermaid architecture diagrams, GPU-vs-NPU comparisons, and source code walkthroughs | `我想了解 vllm-ascend 的量化特性` |
| **model-tutorial** | Generate Chinese technical tutorial documents for specific models running on Ascend NPU | `帮我生成 DeepSeek-V3 在昇腾上的模型教程` |

### 🧑‍💻 Feature Development

| Skill | Description | Trigger Example |
|:------|:------------|:----------------|
| **feature-design** | Design and implement vllm-ascend features with code implementation and Chinese technical design documents | `帮我在 vllm-ascend 中实现 W4A16 量化` |
| **rfc-creator** | Generate vllm-ascend style RFC documents for major architectural change proposals | `帮我生成一个 vllm-ascend RFC` |
| **pr-description-creator** | Generate vllm-ascend style PR descriptions (Purpose / Test Plan / Test Result) from GitHub PR code changes | `帮我生成 vllm-ascend PR 1234 的描述` |

### 🧪 Testing

| Skill | Description | Trigger Example |
|:------|:------------|:----------------|
| **test-creator** | Generate unit tests and integration tests for the vllm-ascend repository | `帮我写 vllm-ascend 中 XXX 的测试用例` |

### 🔎 Code Review

| Skill | Description | Trigger Example |
|:------|:------------|:----------------|
| **pr-summary-creator** | Fetch and deeply analyze vllm-ascend PRs, generating comprehensive reports with code change analysis, Mermaid architecture diagrams, and risk assessment | `帮我分析 vllm-ascend PR 1234` |

### 📊 Performance Analysis

| Skill | Description | Trigger Example |
|:------|:------------|:----------------|
| **benchmark-result-summary** | Compare vllm-ascend serving benchmark outputs before and after a code change with percentage change analysis | *(Paste benchmark output and ask to compare)* |

---

## 🔄 Typical Workflow

Below illustrates a complete **Feature Development → PR Submission → Code Review** workflow:

```mermaid
sequenceDiagram
    participant Dev as 👨‍💻 Developer
    participant Agent as 🤖 AI Agent
    participant Repo as 📦 vllm-ascend

    Dev->>Agent: "Implement W4A16 quantization in vllm-ascend"
    Agent->>Repo: Read source code, analyze architecture
    Agent->>Agent: feature-design skill
    Agent-->>Dev: Code implementation + design document

    Dev->>Agent: "Write tests for W4A16"
    Agent->>Agent: test-creator skill
    Agent-->>Dev: Unit tests + integration tests

    Dev->>Agent: "Generate PR 1234 description"
    Agent->>Agent: pr-description-creator skill
    Agent-->>Dev: PR description

    Dev->>Agent: "Analyze vllm-ascend PR 1234"
    Agent->>Agent: pr-summary-creator skill
    Agent-->>Dev: PR analysis report
```

---

## 📚 Technical Documents (Docs)

Technical documents summarizing feature insights, design decisions, and engineering experience, assisted by Agent Skills.

| Document | Type | Description |
|:---------|:-----|:------------|
| [quantization.md](./docs/quantization.md) | Feature Learning | Deep-dive into vLLM Ascend quantization: plugin registration, config parsing, NPU implementation for each quantization method |
| [design-quantization-refactor.md](./docs/design-quantization-refactor.md) | Design Document | Quantization code refactoring design: eliminating MoE redundancy, unifying weight processing, simplifying 310P registry |
| [rfc-quantization-code-refactoring.md](./docs/rfc-quantization-code-refactoring.md) | RFC | Quantization code refactoring RFC: architectural change proposal for improved maintainability and usability |

> 💡 **Contribute docs**: Use `feature-learning` or `feature-design` skill to generate documents, then submit to the appropriate `docs/` subdirectory.

---

## 📝 Session Records (Sessions)

Key development session records capturing the full interaction between developers and AI agents — from problem analysis and solution design to code changes and lessons learned.

| Session | Description |
|:--------|:------------|
| [session-quantization-refactor.md](./sessions/session-quantization-refactor.md) | Quantization feature principles and code implementation: a complete development session from feature learning to refactoring design |

> 💡 **Share sessions**: Export key sessions to `sessions/` with descriptive filenames (e.g., `session-<topic>.md`).

---

## ⚡ Quick Start

### Prerequisites

- Install [OpenCode](https://opencode.ai) (AI coding assistant)
- Configure an LLM Provider API Key (e.g., Claude, GPT, etc.)
- Clone [vllm-ascend](https://github.com/vllm-project/vllm-ascend) locally (for source code walkthrough and code analysis)

### Installation

**1. Clone this repository**

```bash
git clone https://github.com/Xigorithm/vllm-ascend-workflow.git
cd vllm-ascend-workflow
```

**2. Configure OpenCode Skills Path**

Create or edit `opencode.json` in your vllm-ascend project root:

```json
{
  "skills": {
    "paths": ["<path-to>/vllm-ascend-workflow/skill"]
  }
}
```

**3. Start Using**

Enter trigger commands directly in OpenCode:

```
> 我想了解 vllm-ascend 的 ACL Graph
> 帮我在 vllm-ascend 中实现 W4A16 量化
> 帮我分析 vllm-ascend PR 1234
```

---

## 🛠️ Related Resources

- [vllm-ascend](https://github.com/vllm-project/vllm-ascend) — vLLM Ascend NPU plugin main repository
- [vllm-dev-skills](https://github.com/Xigorithm/vllm-dev-skills) — vLLM general Agent Skills collection (inspiration for this project)
- [OpenCode](https://opencode.ai) — AI coding assistant
- [Agent Skills Specification](https://agentskills.io/home#why-agent-skills) — Agent Skills standards and best practices
- [vLLM Documentation](https://docs.vllm.ai/) — vLLM inference engine documentation

---

## 🤝 Contributing

Contributions are welcome! Here are ways to get involved:

| Contribution Type | How |
|:------------------|:----|
| **Add a Skill** | Create a `.skill` file under `skill/` following OpenCode Skill conventions |
| **Add a Document** | Use a Skill to generate docs, submit to the appropriate `docs/` subdirectory |
| **Share a Session** | Export a key session to `sessions/` with a descriptive filename |
| **Improve Existing** | Suggest improvements or submit PRs for existing Skills / Docs / Sessions |

---

## ©️ Citation

```bibtex
@misc{vllm-ascend-workflow@2026,
  title  = {vllm-ascend-workflow},
  url    = {https://github.com/Xigorithm/vllm-ascend-workflow},
  note   = {AI-assisted development workflow for vllm-ascend},
  author = {Xigorithm},
  year   = {2026}
}
```

---

## 📜 License

Same as [vllm-ascend](https://github.com/vllm-project/vllm-ascend) (Apache-2.0).

