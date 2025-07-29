"""Prompt manager for loading prompts from LangSmith."""
import os
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langsmith import Client
from pydantic import BaseModel
from dotenv import load_dotenv

from core.config import config
from core.langsmith_config import LANGSMITH_PROMPTS

# Load environment variables
load_dotenv()


class PromptManager:
    """Manages loading prompts from LangSmith."""
    
    def __init__(self):
        """Initialize the prompt manager."""
        self._cache = {}
        # LangSmith client will use LANGCHAIN_API_KEY from environment
        api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            raise ValueError("LANGCHAIN_API_KEY or LANGSMITH_API_KEY must be set")
        self.langsmith_client = Client(api_key=api_key)
    
    def get_prompt(self, agent_name: str) -> ChatPromptTemplate:
        """Get a prompt for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            ChatPromptTemplate for the agent
        """
        cache_key = f"langsmith:{agent_name}"
        
        # Check cache
        if config.prompts.cache_prompts and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load prompt from LangSmith
        prompt = self._load_langsmith_prompt(agent_name)
        
        # Cache if enabled
        if config.prompts.cache_prompts:
            self._cache[cache_key] = prompt
        
        return prompt
    
    def _load_langsmith_prompt(self, agent_name: str) -> ChatPromptTemplate:
        """Load a prompt from LangSmith.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            ChatPromptTemplate
        """
        # Get the LangSmith prompt name
        langsmith_name = LANGSMITH_PROMPTS.get(agent_name)
        
        if not langsmith_name:
            raise ValueError(f"No LangSmith prompt found for agent: {agent_name}")
        
        try:
            # Pull prompt from LangSmith
            # Just use the prompt name directly - LangSmith will use the project from the client context
            prompt = self.langsmith_client.pull_prompt(langsmith_name)
            return prompt
        except Exception as e:
            raise ValueError(f"Failed to load prompt from LangSmith: {e}")
    
    def clear_cache(self):
        """Clear the prompt cache."""
        self._cache.clear()


# Global instance
prompt_manager = PromptManager() 