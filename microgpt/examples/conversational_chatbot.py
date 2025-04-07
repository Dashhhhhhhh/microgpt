"""
Conversational Chatbot Example

This example demonstrates how to build a conversational chatbot
with memory using the MicroGPT framework.
"""

import os
import sys
import argparse
import time
import json
from dotenv import load_dotenv

# Add the parent directory to the path for importing the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from microgpt import LLMAgent, OpenAIProvider
from microgpt.memory import JSONFileStorage
from microgpt.utils import log_calls


# Define the base directory for the project (root of microgpt)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Default memory path relative to the base directory
DEFAULT_MEMORY_PATH = os.path.join(BASE_DIR, "data", "chat_memory.json")


class ConversationalAgent(LLMAgent):
    """
    A conversational agent that maintains conversation history.
    
    This agent demonstrates the use of memory to maintain context
    across multiple interactions.
    """
    
    def __init__(self, persona="helpful assistant", model="gpt-3.5-turbo", 
                 memory_path=DEFAULT_MEMORY_PATH, max_history=10, **kwargs):
        """
        Initialize the conversational agent.
        
        Args:
            persona (str): The persona the chatbot should adopt.
            model (str): The LLM model to use.
            memory_path (str): Path to store conversation memory.
            max_history (int): Maximum number of conversation turns to remember.
            **kwargs: Additional arguments to pass to LLMAgent.__init__().
        """
        # Create a custom prompt template for conversation
        prompt_template = (
            "You are a {persona}. Engage in a helpful and natural conversation.\n\n"
            "Conversation history:\n{history}\n\n"
            "User: {user_message}\n\n"
            "Please respond as a {persona}:"
        )
        
        # Create the LLM provider
        api_key = os.getenv("OPENAI_API_KEY")
        provider = OpenAIProvider(api_key=api_key, model=model)
        
        # Set up persistent memory with absolute path
        memory_dir = os.path.dirname(memory_path)
        os.makedirs(memory_dir, exist_ok=True)
        memory = JSONFileStorage(memory_path)
        
        # Initialize the base agent
        super().__init__(
            name="ConversationalChatbot",
            description=f"I am a {persona} that can have natural conversations",
            llm_provider=provider,
            prompt_template=prompt_template,
            memory=memory,
            **kwargs
        )
        
        # Store additional properties
        self.persona = persona
        self.max_history = max_history
        # Use a fixed conversation ID if you want conversations to persist across sessions
        # or keep the timestamped version for new conversations each time
        self.conversation_id = "main_conversation"  # Fixed ID for persistence
        
        # Initialize conversation history
        if not self.recall(self.conversation_id):
            self.remember(self.conversation_id, {"turns": []})
            print(f"Created new conversation history at: {memory_path}")
        else:
            print(f"Loaded existing conversation history from: {memory_path}")
    
    @log_calls
    def chat(self, user_message):
        """
        Process a user message and generate a response.
        
        Args:
            user_message (str): The user's message.
            
        Returns:
            str: The chatbot's response.
        """
        # Retrieve conversation history
        conversation = self.recall(self.conversation_id) or {"turns": []}
        history = conversation["turns"]
        
        # Format conversation history for the prompt
        history_text = ""
        for turn in history[-self.max_history:]:
            history_text += f"User: {turn['user']}\n"
            history_text += f"Assistant: {turn['assistant']}\n\n"
        
        # Generate response
        response = self.generate_text({
            "persona": self.persona,
            "history": history_text,
            "user_message": user_message
        })
        
        # Update conversation history
        history.append({
            "user": user_message,
            "assistant": response,
            "timestamp": time.time()
        })
        
        # Trim history if needed
        if len(history) > self.max_history * 2:
            history = history[-self.max_history * 2:]
        
        # Save updated history
        conversation["turns"] = history
        self.remember(self.conversation_id, conversation)
        
        return response


def main():
    """Run the conversational chatbot from the command line."""
    parser = argparse.ArgumentParser(description="Interactive Conversational Chatbot")
    
    parser.add_argument(
        "--persona", "-p",
        default="helpful assistant",
        help="The persona for the chatbot"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-3.5-turbo",
        help="LLM model to use"
    )
    parser.add_argument(
        "--memory", "-mem",
        default=DEFAULT_MEMORY_PATH,
        help="Path to store conversation memory"
    )
    parser.add_argument(
        "--max-history", "-hist",
        type=int,
        default=10,
        help="Maximum conversation turns to remember"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create the chatbot
    chatbot = ConversationalAgent(
        persona=args.persona,
        model=args.model,
        memory_path=args.memory,
        max_history=args.max_history
    )
    
    print(f"Chatbot initialized with persona: {args.persona}")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Type 'memory' to see what the chatbot remembers.")
    print("Type 'clear' to clear conversation history.")
    print("="*50)
    
    # Start conversation loop
    while True:
        try:
            # Get user input
            user_message = input("\nYou: ").strip()
            
            # Check for exit commands
            if user_message.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye! Conversation ended.")
                break
            
            # Check for memory inspection command
            if user_message.lower() == "memory":
                conversation = chatbot.recall(chatbot.conversation_id)
                print("\n" + "="*50)
                print("CONVERSATION HISTORY")
                print("="*50)
                
                if conversation and "turns" in conversation and conversation["turns"]:
                    for i, turn in enumerate(conversation["turns"]):
                        print(f"\nTurn {i+1}:")
                        print(f"User: {turn['user']}")
                        print(f"Assistant: {turn['assistant']}")
                else:
                    print("No conversation history found.")
                
                print("="*50)
                continue
            
            # Check for clear history command
            if user_message.lower() == "clear":
                chatbot.remember(chatbot.conversation_id, {"turns": []})
                print("\nConversation history cleared.")
                continue
            
            # Generate response
            if not user_message:
                continue
                
            print("\nAssistant:", end=" ")
            response = chatbot.chat(user_message)
            print(response)
            
            # Print token usage for the turn if available
            if hasattr(chatbot.llm_provider, 'token_tracker'):
                last_usage = chatbot.llm_provider.token_tracker.get_last_usage()
                if last_usage:
                    print(f"\n[{last_usage['total_tokens']} tokens used, ${last_usage['total_cost']:.4f}]")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()