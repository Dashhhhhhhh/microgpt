"""
Core functionality for the MicroGPT framework.

This module provides utility functions and core components for
running and managing MicroGPT agents.
"""

from typing import Dict, List, Any, Callable, Optional
from .agent import MicroAgent

def run_agent(agent, input_data):
    """
    Run a MicroAgent with the provided input data.
    
    This is a convenience function for executing agents.
    
    Args:
        agent: A MicroAgent instance to execute.
        input_data: The data to be processed by the agent.
        
    Returns:
        The result from the agent's run method.
    """
    return agent.run(input_data)


class AgentPipeline:
    """
    A pipeline of agents that can be executed in sequence.
    
    This allows connecting multiple agents together, where the output
    of one agent becomes the input to the next.
    """
    
    def __init__(self, name=None):
        """
        Initialize a new AgentPipeline.
        
        Args:
            name (str, optional): A name for this pipeline. Defaults to None.
        """
        self.name = name or "AgentPipeline"
        self.agents = []
        self.transformers = []
    
    def add(self, agent: MicroAgent, transformer: Optional[Callable] = None):
        """
        Add an agent to the pipeline.
        
        Args:
            agent (MicroAgent): The agent to add.
            transformer (callable, optional): A function to transform the output
                of this agent before passing it to the next agent.
                Should take the form: transformer(output) -> transformed_output.
                If None, no transformation will be applied.
        
        Returns:
            self: The pipeline instance, for method chaining.
        """
        self.agents.append(agent)
        self.transformers.append(transformer)
        return self
    
    def run(self, input_data: Any) -> Any:
        """
        Run the pipeline with the provided input data.
        
        The input will be processed by each agent in sequence, with the output
        of each agent becoming the input to the next. Transformers will be
        applied between agents if provided.
        
        Args:
            input_data: The initial input data for the first agent.
            
        Returns:
            The final output from the last agent in the pipeline.
        """
        if not self.agents:
            raise ValueError("Pipeline has no agents")
        
        current_data = input_data
        
        for i, agent in enumerate(self.agents):
            # Run the agent with the current data
            output = agent.run(current_data)
            
            # Transform the output if a transformer is provided (except for the last agent)
            if i < len(self.agents) - 1 and self.transformers[i] is not None:
                current_data = self.transformers[i](output)
            else:
                current_data = output
        
        return current_data


class AgentNetwork:
    """
    A network of interconnected agents that can send messages to each other.
    
    This provides a more flexible structure than a linear pipeline,
    allowing agents to communicate in more complex patterns.
    """
    
    def __init__(self, name=None):
        """
        Initialize a new AgentNetwork.
        
        Args:
            name (str, optional): A name for this network. Defaults to None.
        """
        self.name = name or "AgentNetwork"
        self.agents = {}  # name -> agent mapping
        self.connections = {}  # agent_name -> list of connected agent names
    
    def add_agent(self, agent: MicroAgent, connections: Optional[List[str]] = None):
        """
        Add an agent to the network.
        
        Args:
            agent (MicroAgent): The agent to add.
            connections (list, optional): A list of agent names that this agent
                should be connected to. Defaults to None.
                
        Returns:
            self: The network instance, for method chaining.
        """
        self.agents[agent.name] = agent
        if connections:
            self.connections[agent.name] = connections
        else:
            self.connections[agent.name] = []
        return self
    
    def connect(self, from_agent: str, to_agent: str):
        """
        Create a connection between two agents.
        
        Args:
            from_agent (str): The name of the agent sending messages.
            to_agent (str): The name of the agent receiving messages.
            
        Returns:
            self: The network instance, for method chaining.
        """
        if from_agent not in self.agents:
            raise ValueError(f"Agent '{from_agent}' not found in network")
        if to_agent not in self.agents:
            raise ValueError(f"Agent '{to_agent}' not found in network")
        
        if from_agent not in self.connections:
            self.connections[from_agent] = []
        
        if to_agent not in self.connections[from_agent]:
            self.connections[from_agent].append(to_agent)
        
        return self
    
    def send_message(self, from_agent: str, to_agent: str, message: Any) -> Any:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent (str): The name of the agent sending the message.
            to_agent (str): The name of the agent receiving the message.
            message: The message to send.
            
        Returns:
            The result from the receiving agent.
            
        Raises:
            ValueError: If either agent is not found or if they are not connected.
        """
        if from_agent not in self.agents:
            raise ValueError(f"Agent '{from_agent}' not found in network")
        if to_agent not in self.agents:
            raise ValueError(f"Agent '{to_agent}' not found in network")
        
        if to_agent not in self.connections.get(from_agent, []):
            raise ValueError(f"Agent '{from_agent}' is not connected to '{to_agent}'")
        
        return self.agents[to_agent].run(message)
        
    def broadcast(self, from_agent: str, message: Any) -> Dict[str, Any]:
        """
        Broadcast a message from one agent to all connected agents.
        
        Args:
            from_agent (str): The name of the agent sending the message.
            message: The message to broadcast.
            
        Returns:
            dict: A mapping of agent names to their responses.
        """
        if from_agent not in self.agents:
            raise ValueError(f"Agent '{from_agent}' not found in network")
        
        results = {}
        for to_agent in self.connections.get(from_agent, []):
            results[to_agent] = self.agents[to_agent].run(message)
        
        return results