"""State definition for the SDR Agent workflow."""
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class OutputFormat(BaseModel):
    """Output format specification from user input."""
    format: str = Field(description="Output format: 'json' or 'text'")
    fields: Optional[Dict[str, str]] = Field(
        default=None, 
        description="Field specifications for JSON output"
    )


class AgentResult(BaseModel):
    """Result from an individual agent execution."""
    agent_name: str
    data: Dict[str, Any]
    citations: List[str]
    timestamp: str
    error: Optional[str] = None


class SDRState(TypedDict):
    """Main state object that flows through the LangGraph workflow."""
    # Input fields
    user_query: str
    raw_input: str
    task_content: str
    output_format: Optional[OutputFormat]
    
    # Workflow control
    identified_agents: List[str]
    execution_plan: Dict[str, Any]  # Can be flat or phased
    agent_dependencies: Dict[str, List[str]]
    completed_agents: List[str]
    current_phase: str
    recommended_model: str  # Model recommendation based on query complexity
    
    # Agent results
    agent_results: Dict[str, AgentResult]
    
    # Final output
    raw_content: Dict[str, Any]
    formatted_output: Optional[Any]
    citations: List[str]
    
    # Metadata
    validation_attempts: int
    error_messages: List[str]
    execution_time: float 