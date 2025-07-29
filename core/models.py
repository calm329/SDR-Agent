"""Core models for SDR Agent."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AgentResult(BaseModel):
    """Result from an agent execution."""
    agent_name: str
    data: Dict[str, Any]
    citations: List[str]
    error: Optional[str] = None 