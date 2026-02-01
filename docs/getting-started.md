# Getting Started with MemoryRaven

Welcome to MemoryRaven - the future of AI memory powered by $RAVEN! 🦅

## Prerequisites

- Python 3.9 or higher
- A $RAVEN wallet (optional, for advanced features)
- 500MB free disk space

## Installation

### Basic Installation

```bash
pip install memoryraven
```

### Installation with $RAVEN Features

```bash
pip install "memoryraven[raven]"
```

### Development Installation

```bash
git clone https://github.com/MemoryRaven/MemoryRaven
cd MemoryRaven
pip install -e ".[dev]"
```

## Your First Memory

Let's create an AI that remembers:

```python
from memoryraven import MemoryRaven

# Create your raven
raven = MemoryRaven("my-first-agent")

# Store a memory
await raven.remember("User prefers concise responses")

# Recall memories
memories = await raven.recall("user preferences")
```

## Memory Types

MemoryRaven supports four types of memory:

1. **Episodic Memory** - Events and experiences
2. **Semantic Memory** - Facts and knowledge
3. **Procedural Memory** - Skills and processes
4. **Prospective Memory** - Future intentions

## $RAVEN Integration

To unlock the full power of MemoryRaven:

1. Get $RAVEN tokens from [Uniswap](https://app.uniswap.org)
2. Connect your wallet:

```python
from memoryraven import RavenWallet

wallet = RavenWallet("0xYourAddress", private_key)
raven = MemoryRaven("powered-agent", wallet=wallet)
```

3. Stake for enhanced features:

```python
await raven.stake_raven(amount=1000)  # Unlock premium features
```

## Next Steps

- [Memory Mining Guide](./mining.md) - Earn $RAVEN
- [API Reference](./api.md) - Detailed documentation
- [Examples](../examples/) - Learn by doing

## Need Help?

- 💬 [Telegram Community](https://t.me/RavenBadBihh_Bot)
- 🐦 [Twitter Support](https://x.com/memoryraven)

Happy remembering! 🧠✨