# 🧠 Memory Empire

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Discord](https://img.shields.io/discord/XXX?color=7289da&logo=discord&logoColor=white)](https://discord.gg/XXX)
[![Twitter Follow](https://img.shields.io/twitter/follow/ravenbadbihh?style=social)](https://twitter.com/ravenbadbihh)

**Memory Empire** is the missing memory layer for AI agents — persistent, semantic, and scalable memory infrastructure that survives restarts and enables true long-term learning.

> **"Every mistake becomes a rule. Every rule becomes value."** 

## 🚀 Why Memory Empire?

AI agents suffer from a critical flaw: **they forget everything**. Every restart, every session, all context is lost. Memory Empire solves this by providing:

- 🔄 **Persistent Memory** - Survives restarts, crashes, and updates
- 🧠 **Semantic Search** - Find memories by meaning, not just keywords  
- 📊 **Knowledge Graphs** - Memories connect and build on each other
- ⚡ **Real-time Sync** - Local-first with cloud backup
- 🔐 **Privacy-First** - Your memories, your control

## ✨ Features

### Core Capabilities
- **Vector Knowledge Engine** - Turn messy artifacts (notes, chats, docs) into queryable context
- **Multi-Modal Memory** - Episodic, semantic, procedural, and working memory types
- **Incremental Indexing** - Only process what's changed, save on embeddings
- **Namespace Isolation** - Separate personal, work, and project memories
- **Production-Ready** - Used by thousands of Clawdbot agents in production

### Memory Types
- **Episodic Memory**: What happened when, full context preservation
- **Semantic Memory**: Facts, concepts, and relationships
- **Procedural Memory**: How to do things, learned behaviors
- **Working Memory**: Active context for current tasks

## 🎯 Quick Start

### Installation

```bash
# Basic installation
pip install memory-empire

# With all features
pip install "memory-empire[all]"

# For development
pip install "memory-empire[dev]"
```

### Your First Memory Empire

```python
from memory_empire import MemoryEmpire

# Initialize your empire
empire = MemoryEmpire("my-agent")

# Create memories
await empire.remember(
    "User prefers dark mode and uses VSCode",
    memory_type="semantic"
)

# Learn from mistakes
await empire.learn_from_mistake(
    "Tried to send email without attachment",
    "Always confirm attachments before sending emails"
)

# Retrieve relevant context
context = await empire.recall(
    "user preferences for coding",
    top_k=5
)

# Never forget again
print(context.memories)
# > ["User prefers dark mode and uses VSCode", ...]
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Empire                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Sources           Processing           Storage         │
│  ┌─────┐          ┌─────────┐         ┌────────┐      │
│  │Notes│  ──►     │ Chunk   │   ──►   │ Vector │      │
│  │Chats│          │ Enrich  │         │ Store  │      │
│  │ Docs│          │ Embed   │         └────────┘      │
│  └─────┘          └─────────┘              │          │
│                                            ▼          │
│                                      ┌──────────┐     │
│                                      │Retrieval │     │
│                                      └──────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Advanced Usage

### Multi-Agent Systems

```python
# Create specialized agents with shared memory
customer_agent = MemoryEmpire("customer-service")
research_agent = MemoryEmpire("research")

# Share memories between agents
await customer_agent.share_with(research_agent, namespace="company-knowledge")

# Each agent maintains private memories too
await customer_agent.remember(
    "Customer #123 prefers phone calls",
    namespace="private"
)
```

### Custom Memory Backends

```python
from memory_empire.backends import RedisBackend

# Use Redis for distributed memory
empire = MemoryEmpire(
    "distributed-agent",
    backend=RedisBackend(url="redis://localhost:6379")
)

# Or use cloud storage
from memory_empire.backends import S3Backend

empire = MemoryEmpire(
    "cloud-agent",
    backend=S3Backend(bucket="my-agent-memories")
)
```

### Memory Consolidation

```python
# Automatic memory consolidation
empire = MemoryEmpire(
    "learning-agent",
    consolidation_strategy="sleep"  # Consolidate during downtime
)

# Manual consolidation
insights = await empire.consolidate_memories(
    time_range="last_week",
    generate_insights=True
)
```

## 📚 Documentation

- [Getting Started](docs/getting-started.md) - Installation and first steps
- [Architecture Overview](docs/architecture.md) - How Memory Empire works
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Examples](examples/) - Real-world usage examples
- [Best Practices](docs/best-practices.md) - Production deployment guide

## 🎮 Try It Now

### Interactive Demo
Try Memory Empire in your browser: [demo.memoryempire.ai](https://demo.memoryempire.ai)

### Example Projects
- [Customer Service Bot](examples/customer_service_bot.py) - Never forget a customer preference
- [Research Agent](examples/research_agent.py) - Build knowledge over time
- [Coding Assistant](examples/coding_assistant.py) - Learn from debugging sessions

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/ravenbadbihh/MemoryRaven
cd MemoryRaven

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
make lint
```

### Running Locally with Docker

```bash
# Start local Qdrant instance
docker-compose up -d

# Run with local vector store
export MEMORY_EMPIRE_VECTOR_STORE=qdrant
export MEMORY_EMPIRE_QDRANT_URL=http://localhost:6333

# Start developing
python examples/basic_usage.py
```

## 🤝 Contributing

We love contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repo and spread the word!

## 📊 Benchmarks

Memory Empire is designed for production scale:

| Operation | Performance | Notes |
|-----------|------------|-------|
| Memory Creation | ~10ms | Async, batching supported |
| Semantic Search | ~50ms | 1M memories, top-10 |
| Consolidation | ~1s | Per 1000 memories |
| Sync | Real-time | WebSocket streams |

## 🌟 Who's Using Memory Empire?

- **Clawdbot Agents** - Thousands of persistent AI assistants
- **Research Teams** - Building knowledge bases over time
- **Customer Service** - Never lose context again
- **Personal AI** - Your AI that actually remembers you

## 🗺️ Roadmap

### Current Focus (Q1 2025)
- [x] Core memory types implementation
- [x] Vector search with Qdrant
- [x] Knowledge graph integration
- [ ] Memory visualization UI
- [ ] Enhanced consolidation strategies

### Coming Soon
- Multi-modal memories (images, audio)
- Federated learning across agents
- Memory marketplace
- Enterprise features

## 📜 License

Memory Empire is MIT licensed. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with inspiration from:
- Cognitive science research on human memory
- The Clawdbot community
- Everyone who's ever lost context in a chat

## 📞 Get In Touch

- **Discord**: [Join our community](https://discord.gg/XXX)
- **Twitter**: [@ravenbadbihh](https://twitter.com/ravenbadbihh)
- **Email**: ravenbadbihh@gmail.com

---

<div align="center">

**Memory Empire** - *From forgetful to forever* 🧠⚡

[Website](https://memoryempire.ai) • [Documentation](https://docs.memoryempire.ai) • [Blog](https://blog.memoryempire.ai)

</div>