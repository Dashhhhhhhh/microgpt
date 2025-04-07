"""
Text Summarization Agent Example

This example demonstrates how to build a simple text summarization agent
using the MicroGPT framework.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Add the parent directory to the path for importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from microgpt import LLMAgent, OpenAIProvider
from microgpt.utils import log_calls


class TextSummarizerAgent(LLMAgent):
    """
    An agent that summarizes text content.
    
    This is a simple example of an LLMAgent that takes text input
    and produces a concise summary.
    """
    
    def __init__(self, model="gpt-3.5-turbo", **kwargs):
        """
        Initialize the text summarizer agent.
        
        Args:
            model (str): The LLM model to use.
            **kwargs: Additional arguments to pass to LLMAgent.__init__().
        """
        # Create a custom prompt template for text summarization
        prompt_template = (
            "You are a summarization assistant. Your task is to create a {length} summary "
            "of the following text in {format} format.\n\n"
            "Text to summarize:\n\n{text}\n\n"
            "Please provide a {length} summary in {format} format:"
        )
        
        # Set up the LLM provider
        api_key = os.getenv("OPENAI_API_KEY")
        provider = OpenAIProvider(api_key=api_key, model=model)
        
        # Initialize the base agent
        super().__init__(
            name="TextSummarizer",
            description="I create concise summaries of text content",
            llm_provider=provider,
            prompt_template=prompt_template,
            **kwargs
        )
    
    @log_calls
    def summarize(self, text, length="medium", format="paragraph"):
        """
        Summarize the provided text.
        
        Args:
            text (str): The text to summarize.
            length (str): Desired summary length ('short', 'medium', 'long').
            format (str): Desired summary format ('paragraph', 'bullets', 'outline').
            
        Returns:
            str: The generated summary.
        """
        # Validate inputs
        valid_lengths = ["short", "medium", "long"]
        valid_formats = ["paragraph", "bullets", "outline"]
        
        if length not in valid_lengths:
            logging.warning(f"Invalid length '{length}'. Using 'medium' instead.")
            length = "medium"
        
        if format not in valid_formats:
            logging.warning(f"Invalid format '{format}'. Using 'paragraph' instead.")
            format = "paragraph"
        
        # Generate the summary
        return self.run({
            "text": text,
            "length": length,
            "format": format
        })


def read_file(file_path):
    """
    Read text content from a file.
    
    Args:
        file_path (str): Path to the file to read.
        
    Returns:
        str: The file content.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None


def main():
    """Run the text summarizer from the command line."""
    parser = argparse.ArgumentParser(description="Summarize text content.")
    
    # Input options - either file or direct text
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "-f", help="Path to a text file to summarize")
    input_group.add_argument("--text", "-t", help="Text to summarize")
    
    # Summarization options
    parser.add_argument(
        "--length", "-l",
        choices=["short", "medium", "long"],
        default="medium",
        help="Desired summary length"
    )
    parser.add_argument(
        "--format", "-fmt",
        choices=["paragraph", "bullets", "outline"],
        default="paragraph",
        help="Desired summary format"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get the text to summarize
    if args.file:
        text = read_file(args.file)
        if not text:
            print("Error: Could not read the specified file.")
            return
    else:
        text = args.text
    
    # Create and run the summarizer
    summarizer = TextSummarizerAgent(model=args.model)
    summary = summarizer.summarize(text, length=args.length, format=args.format)
    
    # Print the summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{summary}\n")
    
    # Print token usage if available
    if hasattr(summarizer.llm_provider, 'token_tracker'):
        usage = summarizer.llm_provider.token_tracker.get_total_usage()
        print(f"Token usage: {usage['total_tokens']} tokens (${usage['total_cost']:.4f})")


if __name__ == "__main__":
    main()