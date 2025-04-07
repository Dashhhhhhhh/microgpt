"""
Research Assistant Network Example

This example demonstrates how to build a research assistant using an agent network
with parallel processing, allowing multiple agents to work together on a research task.
"""

import os
import sys
import argparse
import time
from dotenv import load_dotenv

# Add the parent directory to the path for importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from microgpt import LLMAgent, WebAgent, AgentNetwork, OpenAIProvider
from microgpt.memory import JSONFileStorage
from microgpt.utils import log_calls, cache_result


class WebSearchAgent(WebAgent):
    """An agent that searches the web for information."""
    
    def __init__(self, **kwargs):
        """Initialize the web search agent."""
        super().__init__(
            name="WebSearcher",
            description="I search the web for information",
            **kwargs
        )
    
    @log_calls
    @cache_result(ttl=3600)  # Cache results for 1 hour
    def run(self, input_data):
        """
        Search for information on the web.
        
        Args:
            input_data: Either a search query string or a dict with a 'query' key.
            
        Returns:
            dict: The search results.
        """
        # Extract the query
        if isinstance(input_data, dict):
            query = input_data.get('query')
            if not query:
                return {"error": "No query provided in the input data."}
        else:
            query = input_data
        
        # Simulate a web search
        # In a real implementation, this would use a search API like Google, Bing, etc.
        print(f"🔍 Searching the web for: '{query}'")
        time.sleep(1)  # Simulate network delay
        
        # Use a placeholder search result for demo purposes
        search_result = (
            f"Here are some search results for '{query}':\n"
            f"1. Example result 1 about {query}\n"
            f"2. Example result 2 about {query}\n"
            f"3. Example result 3 about {query}\n"
        )
        
        # Try to get a real result using the web_client if possible
        try:
            # Get a simple website with info about the topic
            # For safety, we'll just fetch example.com in this demo
            # but in a real implementation, you would use the query to find relevant pages
            html = self.web_client.fetch_html("http://example.com")
            web_content = self.web_client.extract_text(html)
            search_result += f"\nContent from example.com: {web_content[:500]}..."
        except Exception as e:
            search_result += f"\nError fetching real web content: {e}"
        
        # Remember this search
        result = {
            "query": query,
            "results": search_result,
            "source": "web_search",
            "timestamp": time.time()
        }
        self.remember(f"search:{query}", result)
        
        return result


class WikipediaAgent(WebAgent):
    """An agent that searches Wikipedia for information."""
    
    def __init__(self, **kwargs):
        """Initialize the Wikipedia agent."""
        super().__init__(
            name="WikipediaResearcher",
            description="I search Wikipedia for factual information",
            **kwargs
        )
    
    @log_calls
    @cache_result(ttl=86400)  # Cache results for 24 hours
    def run(self, input_data):
        """
        Search Wikipedia for information.
        
        Args:
            input_data: Either a search query string or a dict with a 'query' key.
            
        Returns:
            dict: The search results from Wikipedia.
        """
        # Extract the query
        if isinstance(input_data, dict):
            query = input_data.get('query')
            if not query:
                return {"error": "No query provided in the input data."}
        else:
            query = input_data
        
        # Simulate a Wikipedia search
        # In a real implementation, this would use Wikipedia's API
        print(f"📚 Searching Wikipedia for: '{query}'")
        time.sleep(1.5)  # Simulate network delay
        
        # Use a placeholder Wikipedia result for demo purposes
        wiki_result = (
            f"Wikipedia article about '{query}':\n\n"
            f"{query} is a topic of interest described in Wikipedia. "
            f"It has various aspects and characteristics that make it notable. "
            f"The article provides information about its history, significance, and related concepts."
        )
        
        # Remember this search
        result = {
            "query": query,
            "results": wiki_result,
            "source": "wikipedia",
            "timestamp": time.time()
        }
        self.remember(f"wiki:{query}", result)
        
        return result


class NewsSearchAgent(WebAgent):
    """An agent that searches for recent news on a topic."""
    
    def __init__(self, **kwargs):
        """Initialize the news search agent."""
        super().__init__(
            name="NewsSearcher",
            description="I search for recent news on topics",
            **kwargs
        )
    
    @log_calls
    @cache_result(ttl=3600)  # Cache results for 1 hour
    def run(self, input_data):
        """
        Search for recent news on a topic.
        
        Args:
            input_data: Either a search query string or a dict with a 'query' key.
            
        Returns:
            dict: The news search results.
        """
        # Extract the query
        if isinstance(input_data, dict):
            query = input_data.get('query')
            if not query:
                return {"error": "No query provided in the input data."}
        else:
            query = input_data
        
        # Simulate a news search
        # In a real implementation, this would use a news API
        print(f"📰 Searching news for: '{query}'")
        time.sleep(1.2)  # Simulate network delay
        
        # Use a placeholder news result for demo purposes
        current_date = time.strftime("%Y-%m-%d")
        news_result = (
            f"Recent news about '{query}' as of {current_date}:\n\n"
            f"• News Article 1: New developments regarding {query}\n"
            f"• News Article 2: Experts discuss the impact of {query}\n"
            f"• News Article 3: Recent trends in {query}\n"
        )
        
        # Remember this search
        result = {
            "query": query,
            "results": news_result,
            "source": "news",
            "timestamp": time.time()
        }
        self.remember(f"news:{query}", result)
        
        return result


class ResearchAnalyzerAgent(LLMAgent):
    """
    An agent that analyzes and synthesizes research results.
    
    This agent takes the outputs from multiple research sources and
    combines them into a cohesive, well-structured research summary.
    """
    
    def __init__(self, model="gpt-3.5-turbo", **kwargs):
        """Initialize the research analyzer agent."""
        # Custom template for research analysis
        prompt_template = (
            "You are a research analysis assistant. Your task is to synthesize "
            "information from multiple sources into a coherent research summary.\n\n"
            "Research Query: {query}\n\n"
            "Source 1 (Web Search):\n{web_results}\n\n"
            "Source 2 (Wikipedia):\n{wiki_results}\n\n"
            "Source 3 (News):\n{news_results}\n\n"
            "Please synthesize this information into a comprehensive research summary "
            "on the topic. Include key facts, different perspectives, and recent developments. "
            "Structure your response with clear sections and highlight particularly "
            "important information. Cite the sources used (Web Search, Wikipedia, News).\n\n"
            "Research Summary:"
        )
        
        # Create the LLM provider
        api_key = os.getenv("OPENAI_API_KEY")
        provider = OpenAIProvider(api_key=api_key, model=model)
        
        super().__init__(
            name="ResearchAnalyzer",
            description="I analyze and synthesize research from multiple sources",
            llm_provider=provider,
            prompt_template=prompt_template,
            **kwargs
        )
    
    @log_calls
    def run(self, input_data):
        """
        Analyze and synthesize research results.
        
        Args:
            input_data: A dictionary containing research results from multiple sources.
            
        Returns:
            dict: A research summary combining all sources.
        """
        if not isinstance(input_data, dict):
            return {"error": "Invalid input: Expected a dictionary with research results"}
        
        # Generate the research summary
        summary = self.generate_text({
            "query": input_data.get("query", "Unknown topic"),
            "web_results": input_data.get("web_results", {}).get("results", "No web search results available"),
            "wiki_results": input_data.get("wiki_results", {}).get("results", "No Wikipedia results available"),
            "news_results": input_data.get("news_results", {}).get("results", "No news results available")
        })
        
        # Create the final research report
        result = {
            "query": input_data.get("query", "Unknown topic"),
            "summary": summary,
            "sources": [
                input_data.get("web_results", {}).get("source", "web_search"),
                input_data.get("wiki_results", {}).get("source", "wikipedia"),
                input_data.get("news_results", {}).get("source", "news")
            ],
            "timestamp": time.time()
        }
        
        # Remember this research
        self.remember(f"research:{result['query']}", result)
        
        return result


class ResearchAssistantNetwork:
    """
    A network of agents that collaborate on research tasks.
    
    This class demonstrates how to build a non-linear network of agents
    that can work in parallel and combine their results.
    """
    
    def __init__(self, model="gpt-3.5-turbo", memory_path="./data/research_memory"):
        """
        Initialize the research assistant network.
        
        Args:
            model (str): The LLM model to use for the analyzer.
            memory_path (str): Path for storing persistent memory.
        """
        # Create a persistent memory store
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        memory = JSONFileStorage(memory_path)
        
        # Create the individual agents
        self.web_agent = WebSearchAgent(memory=memory)
        self.wiki_agent = WikipediaAgent(memory=memory)
        self.news_agent = NewsSearchAgent(memory=memory)
        self.analyzer = ResearchAnalyzerAgent(model=model, memory=memory)
        
        # Set up the agent network
        self.network = AgentNetwork(name="ResearchNetwork")
        
        # Add nodes (agents)
        self.network.add_node("web_search", self.web_agent)
        self.network.add_node("wiki_search", self.wiki_agent)
        self.network.add_node("news_search", self.news_agent)
        self.network.add_node("analyzer", self.analyzer)
        
        # Add edges (connections between agents)
        # No direct edges - we'll manually collect results from parallel searches
    
    def research(self, query):
        """
        Perform comprehensive research on a query.
        
        Args:
            query (str): The research query.
            
        Returns:
            dict: The research results and summary.
        """
        print(f"🔍 Starting research on: '{query}'\n")
        
        # Run the search agents in "parallel"
        # In a real implementation, you could use async/await or threading
        web_results = self.web_agent.run({"query": query})
        wiki_results = self.wiki_agent.run({"query": query})
        news_results = self.news_agent.run({"query": query})
        
        # Combine the results
        combined_results = {
            "query": query,
            "web_results": web_results,
            "wiki_results": wiki_results,
            "news_results": news_results
        }
        
        # Analyze and synthesize the results
        print("\n📊 Analyzing and synthesizing research results...")
        result = self.analyzer.run(combined_results)
        
        return result
    
    def display_results(self, result):
        """
        Display the research results in a readable format.
        
        Args:
            result (dict): The research results to display.
        """
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print(f"RESEARCH SUMMARY: {result.get('query', 'Unknown topic').upper()}")
        print("="*60)
        
        print(f"\n{result.get('summary', 'No summary available')}")
        
        print("\n" + "-"*60)
        print(f"Sources: {', '.join(result.get('sources', ['Unknown']))}")
        print("-"*60)
        
        # Print token usage if available
        if hasattr(self.analyzer, 'llm_provider') and hasattr(self.analyzer.llm_provider, 'token_tracker'):
            usage = self.analyzer.llm_provider.token_tracker.get_total_usage()
            print(f"\nToken usage: {usage['total_tokens']} tokens (${usage['total_cost']:.4f})")


def main():
    """Run the research assistant from the command line."""
    parser = argparse.ArgumentParser(description="Research Assistant Network")
    parser.add_argument("--query", "-q", required=True, help="The research query")
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use for analysis"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create and run the research assistant
    assistant = ResearchAssistantNetwork(model=args.model)
    result = assistant.research(args.query)
    assistant.display_results(result)


if __name__ == "__main__":
    main()