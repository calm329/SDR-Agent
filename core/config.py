"""Configuration settings for the SDR Agent system."""
import os
from typing import Literal

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for prompt management."""
    
    # Enable prompt caching
    cache_prompts: bool = Field(
        default=True,
        description="Whether to cache prompts after loading"
    )


class AgentConfig(BaseModel):
    """Configuration for agent behavior."""
    max_retries: int = Field(
        default=3,
        description="Maximum retries for agent operations"
    )
    
    timeout_seconds: int = Field(
        default=60,
        description="Timeout for individual agent operations"
    )
    
    parallel_execution: bool = Field(
        default=True,
        description="Whether to enable parallel agent execution"
    )


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    provider: str = Field(
        default="openai",
        description="LLM provider"
    )
    
    model_name: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name to use"
    )
    
    temperature: float = Field(
        default=0.0,
        description="Temperature for model responses"
    )
    
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for responses"
    )


class BrightdataConfig(BaseModel):
    """Configuration for Brightdata integration."""
    api_key: str = Field(
        default_factory=lambda: os.getenv("BRIGHTDATA_API_KEY", ""),
        description="Brightdata API key"
    )
    
    web_unlocker_zone: str = Field(
        default="mcp_unlocker",
        description="Brightdata web unlocker zone"
    )
    
    browser_zone: str = Field(
        default="mcp_browser",
        description="Brightdata browser zone"
    )
    
    rate_limit: str = Field(
        default="100/1h",
        description="Rate limit configuration"
    )


class EmailEnrichmentConfig(BaseModel):
    """Configuration for email enrichment services."""
    apollo_api_key: str = Field(
        default_factory=lambda: os.getenv("APOLLO_API_KEY", ""),
        description="Apollo.io API key for contact enrichment"
    )
    
    hunter_api_key: str = Field(
        default_factory=lambda: os.getenv("HUNTER_API_KEY", ""),
        description="Hunter.io API key for email finding"
    )
    
    clearbit_api_key: str = Field(
        default_factory=lambda: os.getenv("CLEARBIT_API_KEY", ""),
        description="Clearbit API key (optional)"
    )
    
    timeout_seconds: int = Field(
        default=30,
        description="Timeout for enrichment API calls"
    )
    
    max_retries: int = Field(
        default=2,
        description="Maximum retries for failed enrichment attempts"
    )


class SDRAgentConfig(BaseModel):
    """Main configuration for the SDR Agent system."""
    prompts: PromptConfig = Field(
        default_factory=PromptConfig,
        description="Prompt configuration"
    )
    
    agents: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent configuration"
    )
    
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Model configuration"
    )
    
    brightdata: BrightdataConfig = Field(
        default_factory=BrightdataConfig,
        description="Brightdata configuration"
    )
    
    email_enrichment: EmailEnrichmentConfig = Field(
        default_factory=EmailEnrichmentConfig,
        description="Email enrichment configuration"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    @classmethod
    def from_env(cls) -> "SDRAgentConfig":
        """Create config from environment variables."""
        return cls(
            prompts=PromptConfig(),
            agents=AgentConfig(
                max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3")),
                timeout_seconds=int(os.getenv("AGENT_TIMEOUT", "60")),
                parallel_execution=os.getenv("PARALLEL_EXECUTION", "true").lower() == "true"
            ),
            model=ModelConfig(
                provider=os.getenv("MODEL_PROVIDER", "openai"),
                model_name=os.getenv("MODEL_NAME", "gpt-4-turbo-preview"),
                temperature=float(os.getenv("MODEL_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4096"))
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )


# Global config instance
config = SDRAgentConfig.from_env() 