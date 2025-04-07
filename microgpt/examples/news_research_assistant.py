"""
News Research Assistant Example

This example demonstrates how to build a news research assistant
that can search for and summarize news articles on a given topic.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path for importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from microgpt import LLMAgent, OpenAIProvider
from microgpt.web import WebSearch, WebPageReader
from microgpt.memory import JSONFileStorage
from microgpt.utils import log_calls


class NewsResearchAgent(LLMAgent):
    """
    A news research assistant that can search for and summarize news articles.
    
    This agent demonstrates web search and web content processing capabilities,
    along with basic memory features to store research results.
    """
    
    def __init__(self, model="gpt-3.5-turbo", memory_path="./data/news_research.json", **kwargs):
        """
        Initialize the news research assistant.
        
        Args:
            model (str): The LLM model to use.
            memory_path (str): Path to store research results.
            **kwargs: Additional arguments to pass to LLMAgent.__init__().
        """
        # Create a prompt template for summarizing articles
        summarization_template = (
            "You are a news research assistant. Your task is to create a concise summary "
            "of the following news article. Focus on the key facts, events, and quotes.\n\n"
            "Article content:\n{article_content}\n\n"
            "Please provide a concise summary of this article:"
        )
        
        # Create a prompt template for synthesizing research
        synthesis_template = (
            "You are a news research assistant. Your task is to create a comprehensive report "
            "based on the following news article summaries related to: {topic}\n\n"
            "Article summaries:\n{article_summaries}\n\n"
            "Please provide a well-structured research report that synthesizes these findings. "
            "Include key trends, points of agreement/disagreement between sources, and highlight "
            "any particularly important developments:"
        )
        
        # Set up the LLM provider
        api_key = os.getenv("OPENAI_API_KEY")
        provider = OpenAIProvider(api_key=api_key, model=model)
        
        # Set up web tools
        self.web_search = WebSearch()
        self.web_reader = WebPageReader()
        
        # Set up persistent memory
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        memory = JSONFileStorage(memory_path)
        
        # Initialize the base agent
        super().__init__(
            name="NewsResearchAssistant",
            description="I research and summarize news articles on specific topics",
            llm_provider=provider,
            prompt_template=summarization_template,  # Default template
            memory=memory,
            **kwargs
        )
        
        # Store additional templates
        self.summarization_template = summarization_template
        self.synthesis_template = synthesis_template
    
    @log_calls
    def search_news(self, topic, max_results=5):
        """
        Search for news articles on a specific topic.
        
        Args:
            topic (str): The topic to search for.
            max_results (int): Maximum number of search results to return.
            
        Returns:
            list: A list of search result dictionaries with 'title' and 'url'.
        """
        print(f"Searching for news on: {topic}")
        search_query = f"{topic} news"
        search_results = self.web_search.search(search_query, max_results=max_results)
        
        # Filter and format results
        news_results = []
        for result in search_results:
            news_results.append({
                'title': result.get('title', 'No title'),
                'url': result.get('link', '')
            })
        
        return news_results
    
    @log_calls
    def summarize_article(self, url):
        """
        Fetch and summarize a news article from a given URL.
        
        Args:
            url (str): The URL of the news article to summarize.
            
        Returns:
            dict: A dictionary containing article information and summary.
        """
        print(f"Reading article: {url}")
        
        # Fetch article content
        try:
            article_content = self.web_reader.read_webpage(url)
            if not article_content or len(article_content) < 100:
                return {"url": url, "error": "Could not extract meaningful content"}
        except Exception as e:
            return {"url": url, "error": f"Error fetching article: {str(e)}"}
        
        # Set prompt template for summarization
        self.prompt_template = self.summarization_template
        
        # Generate summary
        print("Generating summary...")
        summary = self.run({"article_content": article_content})
        
        # Create article record
        article_info = {
            "url": url,
            "content_length": len(article_content),
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
        return article_info
    
    @log_calls
    def research_topic(self, topic, max_articles=3):
        """
        Conduct comprehensive research on a topic by searching for,
        reading, and summarizing multiple news articles.
        
        Args:
            topic (str): The topic to research.
            max_articles (int): Maximum number of articles to process.
            
        Returns:
            dict: Research results including article summaries and a synthesis.
        """
        print(f"Starting research on topic: {topic}")
        
        # Search for relevant news articles
        search_results = self.search_news(topic, max_results=max_articles + 2)
        
        # Process each article
        article_summaries = []
        for idx, result in enumerate(search_results[:max_articles]):
            print(f"\nProcessing article {idx + 1}/{min(max_articles, len(search_results))}")
            article_info = self.summarize_article(result['url'])
            
            if "error" not in article_info:
                article_summaries.append({
                    "title": result['title'],
                    "url": result['url'],
                    "summary": article_info['summary']
                })
        
        # Generate research synthesis if we have summaries
        synthesis = ""
        if article_summaries:
            # Set prompt template for synthesis
            self.prompt_template = self.synthesis_template
            
            # Format article summaries for the prompt
            formatted_summaries = ""
            for idx, article in enumerate(article_summaries):
                formatted_summaries += f"Article {idx + 1}: {article['title']}\n"
                formatted_summaries += f"Source: {article['url']}\n"
                formatted_summaries += f"Summary: {article['summary']}\n\n"
            
            # Generate synthesis
            print("\nGenerating research synthesis...")
            synthesis = self.run({
                "topic": topic,
                "article_summaries": formatted_summaries
            })
        
        # Create research record
        research_results = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "article_summaries": article_summaries,
            "synthesis": synthesis
        }
        
        # Save research to memory
        memory_key = f"research_{topic.replace(' ', '_').lower()}"
        self.remember(memory_key, research_results)
        
        return research_results
    
    @log_calls
    def get_past_research(self, topic=None):
        """
        Retrieve past research results from memory.
        
        Args:
            topic (str, optional): Specific topic to retrieve.
            
        Returns:
            dict or list: Research results for a topic or all research.
        """
        if topic:
            memory_key = f"research_{topic.replace(' ', '_').lower()}"
            return self.recall(memory_key)
        else:
            # Get all research topics
            research_topics = {}
            for key in self.memory.list_keys():
                if key.startswith("research_"):
                    research_data = self.recall(key)
                    if research_data:
                        research_topics[key] = {
                            "topic": research_data.get("topic"),
                            "timestamp": research_data.get("timestamp"),
                            "num_articles": len(research_data.get("article_summaries", []))
                        }
            return research_topics


def main():
    """Run the news research assistant from the command line."""
    parser = argparse.ArgumentParser(description="News Research Assistant")
    
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic to research (required for research mode)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["research", "list", "view"],
        default="research",
        help="Operation mode: research a topic, list past research, or view a specific research"
    )
    
    parser.add_argument(
        "--articles", "-a",
        type=int,
        default=3,
        help="Number of articles to process (for research mode)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--memory", "-mem",
        default="./data/news_research.json",
        help="Path to store research results"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create the research assistant
    assistant = NewsResearchAgent(
        model=args.model,
        memory_path=args.memory
    )
    
    # Handle different modes
    if args.mode == "research":
        if not args.topic:
            print("Error: Topic is required for research mode.")
            parser.print_help()
            return
        
        # Conduct research
        results = assistant.research_topic(args.topic, max_articles=args.articles)
        
        # Print results
        print("\n" + "="*80)
        print(f"RESEARCH RESULTS: {args.topic}")
        print("="*80)
        
        print(f"\nProcessed {len(results['article_summaries'])} articles")
        
        print("\nRESEARCH SYNTHESIS:")
        print("-"*80)
        print(results['synthesis'])
        
        # Print token usage if available
        if hasattr(assistant.llm_provider, 'token_tracker'):
            usage = assistant.llm_provider.token_tracker.get_total_usage()
            print("\n" + "-"*80)
            print(f"Total token usage: {usage['total_tokens']} tokens (${usage['total_cost']:.4f})")
    
    elif args.mode == "list":
        # List all past research
        research_topics = assistant.get_past_research()
        
        print("\n" + "="*80)
        print("PAST RESEARCH TOPICS")
        print("="*80 + "\n")
        
        if not research_topics:
            print("No past research found.")
        else:
            for idx, (key, info) in enumerate(research_topics.items()):
                print(f"{idx+1}. Topic: {info['topic']}")
                print(f"   Date: {info['timestamp']}")
                print(f"   Articles: {info['num_articles']}")
                print()
    
    elif args.mode == "view":
        if not args.topic:
            print("Error: Topic is required for view mode.")
            parser.print_help()
            return
        
        # View specific research
        results = assistant.get_past_research(args.topic)
        
        if not results:
            print(f"No research found for topic: {args.topic}")
            return
        
        print("\n" + "="*80)
        print(f"RESEARCH RESULTS: {args.topic}")
        print("="*80)
        
        print(f"\nResearch date: {results['timestamp']}")
        print(f"Processed {len(results['article_summaries'])} articles")
        
        print("\nARTICLE SUMMARIES:")
        for idx, article in enumerate(results['article_summaries']):
            print(f"\n{idx+1}. {article['title']}")
            print(f"   Source: {article['url']}")
            print(f"   Summary: {article['summary']}")
        
        print("\nRESEARCH SYNTHESIS:")
        print("-"*80)
        print(results['synthesis'])


if __name__ == "__main__":
    main()