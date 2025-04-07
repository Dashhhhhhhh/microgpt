# MicroGPT

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/microgpt.svg)](https://badge.fury.io/py/microgpt)

A lightweight, modular framework for building composable AI agents.

## Overview

MicroGPT is a Python framework designed for creating small, focused AI agents that can work independently or together. It provides the building blocks for creating sophisticated AI applications through simple, composable components.

Key features:
- 🧩 **Modular Design**: Build specialized agents for specific tasks
- 🔄 **Composability**: Chain agents together in pipelines and networks
- 💾 **Memory Management**: Flexible memory options for storing state and context
- 🌐 **Web Integration**: Built-in web capabilities for fetching and processing content
- 📊 **Token Tracking**: Monitor and manage LLM token usage
- 🛠️ **Utility Decorators**: Enhance agent functionality with logging, caching, and retry mechanisms

## Installation

```bash
pip install microgpt
```

Or install directly from the repository:

```bash
git clone https://github.com/username/microgpt.git
cd microgpt
pip install -e .
```

## Quick Start

```python
from microgpt import LLMAgent, OpenAIProvider
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Create an LLM provider using your API key
provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

# Create a simple agent
agent = LLMAgent(
    name="Helpdesk",
    description="I provide helpful responses to user questions",
    llm_provider=provider
)

# Ask a question
response = agent.run("What are some ways to improve productivity?")
print(response)
```

## Examples

Check out the [`/examples`](examples) directory for complete working examples:

- [Summarization Agent](examples/summarization_agent.py): Summarize text with customizable formats
- [Conversational Chatbot](examples/conversational_chatbot.py): Build a chatbot with memory
- [News Research Assistant](examples/news_research_assistant.py): Analyze articles with multiple agents in a pipeline
- [Research Assistant Network](examples/research_assistant_network.py): Complex agent network for research tasks

## Environment Setup

Copy the `.env.example` file to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env to add your API keys
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
