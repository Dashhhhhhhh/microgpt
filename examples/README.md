# MicroGPT Examples

This directory contains example implementations that demonstrate how to use the MicroGPT framework for various applications.

## Available Examples

### 1. Summarization Agent (`summarization_agent.py`)

A simple agent that summarizes text with customizable formats and lengths.

**Usage:**
```bash
python -m microgpt.examples.summarization_agent --file path/to/text_file.txt --length medium --format paragraph
```

### 2. Conversational Chatbot (`conversational_chatbot.py`)

A chatbot that maintains conversation history and context across multiple interactions.

**Usage:**
```bash
python -m microgpt.examples.conversational_chatbot --persona "friendly assistant"
```

### 3. News Research Assistant (`news_research_assistant.py`)

A pipeline of agents that fetch, analyze, and summarize news articles.

**Usage:**
```bash
python -m microgpt.examples.news_research_assistant --url https://example.com/news-article
```

### 4. Research Assistant Network (`research_assistant_network.py`)

A complex network of specialized agents working together to research a topic.

**Usage:**
```bash
python -m microgpt.examples.research_assistant_network --topic "renewable energy trends"
```

## Running the Examples

1. Make sure you have set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run any example using the command format:
   ```bash
   python -m microgpt.examples.example_name [arguments]
   ```

## Creating Your Own Examples

Want to create a new example?

1. Start by copying an existing example that's closest to what you want to build
2. Modify the agent classes and functionality to suit your needs
3. Add a command-line interface with argparse for easy usage
4. Consider adding memory persistence if your agent needs to remember things
5. Document your example in this README

Happy coding!