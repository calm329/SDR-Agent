"""LangSmith tracing utilities for monitoring and debugging."""
import asyncio
import functools
import os
import time
from typing import Any, Callable, Dict, Optional

from langsmith import Client
from langsmith.run_helpers import traceable


def setup_langsmith_tracing() -> Client:
    """Initialize LangSmith client with proper configuration."""
    # Ensure environment variables are set
    required_vars = ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Environment variable {var} is not set")
    
    # Enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    
    client = Client()
    return client


def trace_agent(agent_name: str):
    """Decorator to trace agent execution with metadata."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @traceable(
            name=f"agent_{agent_name}",
            metadata={"agent_type": agent_name}
        )
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Add execution metadata
                if hasattr(result, "__dict__"):
                    result.execution_time = execution_time
                
                return result
            except Exception as e:
                # Log error to LangSmith
                raise e
        
        @functools.wraps(func)
        @traceable(
            name=f"agent_{agent_name}",
            metadata={"agent_type": agent_name}
        )
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Add execution metadata
                if hasattr(result, "__dict__"):
                    result.execution_time = execution_time
                
                return result
            except Exception as e:
                # Log error to LangSmith
                raise e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def trace_tool(tool_name: str):
    """Decorator to trace tool execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @traceable(
            name=f"tool_{tool_name}",
            metadata={"tool_type": tool_name}
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_token_usage(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: Optional[float] = None
):
    """Log token usage metrics to LangSmith."""
    metadata = {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    
    if total_cost:
        metadata["cost"] = total_cost
    
    # This will be logged within the current trace context
    return metadata


class MetricsCollector:
    """Collect and aggregate metrics for LangSmith evaluation."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "agent_calls": {},
            "tool_calls": {},
            "errors": [],
            "latencies": []
        }
    
    def record_agent_call(self, agent_name: str, tokens: int, latency: float):
        """Record metrics for an agent call."""
        if agent_name not in self.metrics["agent_calls"]:
            self.metrics["agent_calls"][agent_name] = {
                "count": 0,
                "total_tokens": 0,
                "avg_latency": 0
            }
        
        agent_metrics = self.metrics["agent_calls"][agent_name]
        agent_metrics["count"] += 1
        agent_metrics["total_tokens"] += tokens
        
        # Update average latency
        prev_avg = agent_metrics["avg_latency"]
        prev_count = agent_metrics["count"] - 1
        agent_metrics["avg_latency"] = (prev_avg * prev_count + latency) / agent_metrics["count"]
        
        self.metrics["total_tokens"] += tokens
        self.metrics["latencies"].append(latency)
    
    def record_error(self, agent_name: str, error: str):
        """Record an error occurrence."""
        self.metrics["errors"].append({
            "agent": agent_name,
            "error": error,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary for evaluation."""
        return {
            **self.metrics,
            "avg_latency": sum(self.metrics["latencies"]) / len(self.metrics["latencies"]) 
                          if self.metrics["latencies"] else 0,
            "error_rate": len(self.metrics["errors"]) / sum(
                agent["count"] for agent in self.metrics["agent_calls"].values()
            ) if self.metrics["agent_calls"] else 0
        } 