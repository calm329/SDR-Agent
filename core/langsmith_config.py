"""LangSmith prompt configuration - all prompts are stored in LangSmith, not in code."""
from typing import Dict, List

# Prompt references - these are the names of prompts stored in LangSmith
# NO ACTUAL PROMPTS IN CODE!
LANGSMITH_PROMPTS = {
    # Agent prompts
    "router": "router",
    "company_research": "company_research",
    "contact_discovery": "contact_discovery", 
    "lead_qualification": "lead_qualification",
    "outreach_personalization": "outreach_personalization",
    "output_formatter": "output_formatter",
    
    # Extraction prompts
    "company_extraction": "company_extraction",
    "company_name_extractor": "company_name_extractor",
    "serp_person_extractor": "serp_person_extractor",
    "tech_stack_extractor": "tech_stack_extractor",
    "email_pattern_detector": "email_pattern_detector",
    "funding_signal_extractor": "funding_signal_extractor",
}

# Track prompt versions for experiments (if needed in future)
PROMPT_VERSIONS: Dict[str, List[str]] = {
    # Currently all prompts use default version
}

# Evaluation dataset names in LangSmith
EVALUATION_DATASETS = {
    "company_research": "sdr-company-research-eval",
    "contact_discovery": "sdr-contact-discovery-eval",
    "lead_qualification": "sdr-lead-qualification-eval",
    "output_formatting": "sdr-output-formatting-eval",
    "end_to_end": "sdr-end-to-end-eval",
}

# Metrics to track in LangSmith
TRACKED_METRICS = [
    "accuracy",
    "helpfulness",
    "token_count",
    "latency",
    "citation_accuracy",
    "format_compliance",
] 