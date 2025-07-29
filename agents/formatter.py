"""Output formatter agent with validation and retry logic."""
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from core.prompt_manager import prompt_manager
from core.state import AgentResult, OutputFormat, SDRState
from utils.tracing import trace_agent


class OutputFormatterAgent:
    """Agent responsible for formatting output according to user specifications."""
    
    def __init__(self, llm: ChatOpenAI = None, max_retries: int = 3):
        """Initialize the output formatter agent."""
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.max_retries = max_retries
    
    @trace_agent("output_formatter")
    async def format_output(self, state: SDRState) -> SDRState:
        """Format the output based on user specifications."""
        output_format = state.get("output_format")
        agent_results = state.get("agent_results", {})
        
        # Compile all results
        compiled_data = self._compile_data(agent_results)
        citations = self._compile_citations(agent_results)
        
        if output_format is None:
            # Plain text output
            formatted_output = await self._format_plain_text(compiled_data, citations)
        else:
            # Structured JSON output with validation
            formatted_output = await self._format_json_with_validation(
                compiled_data, 
                output_format,
                citations
            )
        
        # Update state
        state["formatted_output"] = formatted_output
        state["citations"] = citations
        
        return state
    
    def _compile_data(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Compile all agent results into a single data structure."""
        compiled = {}
        errors = []
        
        for agent_name, result in agent_results.items():
            if result.error is None and result.data:
                # Check for error fields within data
                if "error" in result.data:
                    errors.append({
                        "agent": agent_name,
                        "error": result.data.get("error"),
                        "message": result.data.get("message", ""),
                        "suggestions": result.data.get("suggestions", [])
                    })
                else:
                    compiled[agent_name] = result.data
            elif result.error:
                errors.append({
                    "agent": agent_name,
                    "error": result.error
                })
        
        # Add errors to compiled data so they can be formatted
        if errors:
            compiled["errors"] = errors
        
        return compiled
    
    def _compile_citations(self, agent_results: Dict[str, AgentResult]) -> list[str]:
        """Compile all unique citations from agent results."""
        all_citations = []
        
        for result in agent_results.values():
            if result.citations:
                all_citations.extend(result.citations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in all_citations:
            if citation and citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations
    
    async def _format_plain_text(
        self, 
        data: Dict[str, Any], 
        citations: list[str]
    ) -> str:
        """Format output as plain text."""
        # Pull prompt from LangSmith
        prompt = prompt_manager.get_prompt("output_formatter")
        
        # Format with plain text instructions
        messages = prompt.format_messages(
            data=json.dumps(data, indent=2),
            format_type="plain_text",
            instructions="Format this data as clear, readable plain text suitable for SDRs.",
            citations=json.dumps(citations)
        )
        
        response = self.llm.invoke(messages)
        
        # Add citations
        if citations:
            citations_text = "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
            return response.content + citations_text
        
        return response.content
    
    async def _format_json_with_validation(
        self,
        data: Dict[str, Any],
        output_format: OutputFormat,
        citations: list[str]
    ) -> Dict[str, Any]:
        """Format output as JSON with validation and retry logic."""
        required_fields = output_format.fields or {}
        
        for attempt in range(self.max_retries):
            try:
                # Generate JSON output
                formatted = await self._generate_json_output(
                    data, 
                    required_fields,
                    attempt > 0  # Include error feedback after first attempt
                )
                
                # Validate against schema
                validated = self._validate_json_output(formatted, required_fields)
                
                if validated:
                    # Add citations to the output
                    formatted["_citations"] = citations
                    return formatted
                else:
                    # Prepare for retry with validation feedback
                    data["_validation_error"] = "Output did not match required schema"
                    
            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    data["_json_error"] = f"Invalid JSON: {str(e)}"
                    continue
                else:
                    # Final attempt failed - return best effort
                    return self._fallback_json_format(data, required_fields, citations)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    data["_error"] = str(e)
                    continue
                else:
                    return self._fallback_json_format(data, required_fields, citations)
        
        # Should not reach here, but just in case
        return self._fallback_json_format(data, required_fields, citations)
    
    async def _generate_json_output(
        self,
        data: Dict[str, Any],
        required_fields: Dict[str, str],
        include_error_feedback: bool = False
    ) -> Dict[str, Any]:
        """Generate JSON output using LLM."""
        # Pull prompt from LangSmith
        prompt = prompt_manager.get_prompt("output_formatter")
        
        # Build field specification
        field_spec = json.dumps(required_fields, indent=2)
        
        # Add error feedback if this is a retry
        instructions = f"""
        Format the provided data according to this exact JSON schema:
        {field_spec}
        
        Rules:
        1. Include ONLY the fields specified in the schema
        2. Match the exact field names (case-sensitive)
        3. Use the correct data types (string, integer, etc.)
        4. Return ONLY valid JSON, no explanations
        5. If data is missing for a field, use null
        """
        
        if include_error_feedback and "_validation_error" in data:
            instructions += f"\n\nPrevious attempt failed: {data.get('_validation_error', '')}"
        
        messages = prompt.format_messages(
            data=json.dumps(data, indent=2),
            format_type="json",
            instructions=instructions,
            citations=json.dumps(citations)
        )
        
        response = self.llm.invoke(messages)
        
        # Parse JSON
        return json.loads(response.content)
    
    def _validate_json_output(
        self, 
        output: Dict[str, Any], 
        required_fields: Dict[str, str]
    ) -> bool:
        """Validate JSON output against required schema."""
        for field, dtype in required_fields.items():
            if field not in output:
                return False
            
            value = output[field]
            
            # Allow null values
            if value is None:
                continue
            
            # Type validation
            if dtype == "string" and not isinstance(value, str):
                return False
            elif dtype == "integer" and not isinstance(value, int):
                return False
            elif dtype == "number" and not isinstance(value, (int, float)):
                return False
            elif dtype == "boolean" and not isinstance(value, bool):
                return False
            elif dtype == "array" and not isinstance(value, list):
                return False
            elif dtype == "object" and not isinstance(value, dict):
                return False
        
        return True
    
    def _fallback_json_format(
        self, 
        data: Dict[str, Any], 
        required_fields: Dict[str, str],
        citations: list[str]
    ) -> Dict[str, Any]:
        """Create a fallback JSON format when validation fails."""
        result = {}
        
        # Try to extract values for each required field
        for field, dtype in required_fields.items():
            # Search for the field in the data
            value = self._search_for_field_value(data, field)
            
            # Convert to appropriate type or use null
            if value is None:
                result[field] = None
            elif dtype == "string":
                result[field] = str(value) if value else None
            elif dtype == "integer":
                try:
                    result[field] = int(value)
                except:
                    result[field] = None
            else:
                result[field] = value
        
        result["_citations"] = citations
        result["_fallback_format"] = True
        
        return result
    
    def _search_for_field_value(self, data: Dict[str, Any], field: str) -> Any:
        """Search for a field value in nested data with smart field mapping."""
        # Common field mappings for SDR use cases
        field_mappings = {
            "company_name": ["name", "company", "company_name", "organization"],
            "industry": ["industry", "sector", "vertical", "business_type"],
            "headquarters": ["headquarters", "hq", "location", "address", "office"],
            "employee_count": ["employee_count", "employees", "size", "company_size", "headcount"],
            "key_products": ["products", "services", "offerings", "solutions", "key_products"],
            "revenue": ["revenue", "annual_revenue", "income"],
            "website": ["website", "url", "domain", "web"],
            "founded": ["founded", "established", "year_founded"],
            "description": ["description", "about", "overview", "summary"]
        }
        
        # Check if we have a mapping for this field
        field_lower = field.lower()
        possible_fields = field_mappings.get(field_lower, [field_lower])
        
        # Search through all agent results
        for agent_name, agent_data in data.items():
            if isinstance(agent_data, dict):
                # Try each possible field name
                for possible_field in possible_fields:
                    if possible_field in agent_data:
                        return agent_data[possible_field]
                    # Check lowercase version
                    for key, value in agent_data.items():
                        if key.lower() == possible_field.lower():
                            return value
        
        # Direct match at top level
        if field in data:
            return data[field]
        
        # Try variations of field name
        field_variations = [
            field_lower,
            field_lower.replace("_", ""),
            field_lower.replace("_", " ")
        ]
        
        for key, value in data.items():
            if key.lower() in field_variations:
                return value
        
        return None 