"""Router agent that analyzes queries and determines execution plan."""
import json
import re
from typing import Dict, List, Optional, Tuple, Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from core.state import SDRState, OutputFormat
from utils.tracing import trace_agent


class RouterDecision(BaseModel):
    """Router decision output."""
    identified_agents: List[str] = Field(description="List of agents needed")
    execution_plan: Dict[str, List[str]] = Field(
        description="Execution plan with parallel and sequential phases"
    )
    task_type: str = Field(description="Primary task type identified")
    requires_structured_output: bool = Field(description="Whether structured output is needed")


class RouterAgent:
    """Agent responsible for analyzing queries and routing to appropriate agents."""
    
    AGENT_TRIGGERS = {
        "company_research": [
            "company", "about", "tell me about", "what does", 
            "company summary", "describe", "overview"
        ],
        "contact_discovery": [
            "find", "who is", "contact", "email", "decision maker",
            "head of", "vp of", "director", "manager", "ceo", "cto"
        ],
        "lead_qualification": [
            "qualify", "good fit", "hiring", "buying signals",
            "job posting", "opportunities", "pain points"
        ],
        "outreach_personalization": [
            "personalize", "hook", "reach out", "approach",
            "outreach", "personalization", "tailor"
        ]
    }
    
    AGENT_DEPENDENCIES = {
        "outreach_personalization": ["company_research", "contact_discovery"],
        "lead_qualification": ["company_research"]
    }
    
    # Indicators of complex queries that require reasoning
    COMPLEXITY_INDICATORS = {
        "multi_entity": ["compare", "versus", "vs", "competitors", "alternatives"],
        "analysis": ["analyze", "evaluate", "assess", "determine", "explain why"],
        "multi_step": ["and then", "after that", "followed by", "also"],
        "specific_role": ["vp of", "head of", "director of", "chief", "senior"],
        "detailed_request": ["including", "with details about", "comprehensive", "in-depth"],
        "reasoning": ["why", "how does", "what makes", "explain"],
        "current_events": ["latest", "recent", "current", "today", "this week", "funding round"],
        "multiple_tasks": ["and also", "as well as", "plus", "additionally"]
    }
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the router agent."""
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        from core.prompt_manager import prompt_manager
        self.prompt_manager = prompt_manager
    
    @trace_agent("router")
    async def analyze_query(self, state: SDRState) -> SDRState:
        """Analyze the user query and determine routing strategy."""
        # Detect query complexity and appropriate model
        recommended_model = self._detect_query_complexity(state["raw_input"])
        
        # Get the prompt
        prompt = self.prompt_manager.get_prompt("router")
        
        # Format messages - handle multiple variable names
        format_vars = {}
        for var in prompt.input_variables:
            if var in ["question", "raw_input", "query"]:
                format_vars[var] = state["raw_input"]
        
        messages = prompt.format_messages(**format_vars)
        
        # Invoke LLM to analyze the query
        response = self.llm.invoke(messages)
        
        # Parse the JSON response
        try:
            routing_decision = json.loads(response.content)
            
            # Post-process: For SDR workflow, ensure all agents are included
            identified = routing_decision.get("identified_agents", [])
            if "company_research" in identified and "contact_discovery" in identified:
                # This is an SDR request - ensure we also qualify and personalize
                if "lead_qualification" not in identified:
                    identified.append("lead_qualification")
                if "outreach_personalization" not in identified:
                    identified.append("outreach_personalization")
                routing_decision["identified_agents"] = identified
                
                # Update execution plan for proper SDR sequencing
                routing_decision["execution_plan"] = {
                    "phase_1": {
                        "parallel": ["company_research"],
                        "sequential": []
                    },
                    "phase_2": {
                        "parallel": ["contact_discovery", "lead_qualification"],
                        "sequential": []
                    },
                    "phase_3": {
                        "parallel": ["outreach_personalization"],
                        "sequential": []
                    }
                }
                
        except json.JSONDecodeError as e:
            # Fallback to basic pattern matching if LLM fails
            task_content, output_format = self._parse_input(state["raw_input"])
            identified_agents = self._identify_agents(task_content)
            execution_plan = self._create_execution_plan(identified_agents)
            
            routing_decision = {
                "task_content": task_content,
                "output_format": output_format.model_dump() if output_format else None,
                "identified_agents": identified_agents,
                "execution_plan": execution_plan
            }
        
        # Update state with routing decision
        # Handle output_format which might be a string or a dict
        output_format = None
        if routing_decision.get("output_format"):
            if isinstance(routing_decision["output_format"], str):
                # Router returned just the format type
                output_format = OutputFormat(format=routing_decision["output_format"])
            elif isinstance(routing_decision["output_format"], dict):
                # Router returned full format spec
                output_format = OutputFormat(**routing_decision["output_format"])
        
        # Update state with routing decision
        state.update({
            "task_content": routing_decision.get("task_content", state["raw_input"]),
            "output_format": output_format,
            "identified_agents": routing_decision.get("identified_agents", []),
            "execution_plan": routing_decision.get("execution_plan", {}),
            "current_phase": "routing_complete",
            "recommended_model": recommended_model
        })
        
        return state
    
    def _parse_input(self, raw_input: str) -> Tuple[str, Optional[OutputFormat]]:
        """Parse user input to extract task and output format."""
        # Check if input is JSON with format specification
        try:
            # Try to parse as JSON
            data = json.loads(raw_input)
            if isinstance(data, dict) and "format" in data:
                # Structured output requested
                output_format = OutputFormat(
                    format=data.get("format", "json"),
                    fields=data.get("fields", {})
                )
                # Extract the actual query
                task_content = data.get("query", raw_input)
                return task_content, output_format
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Check for inline JSON format specification
        json_match = re.search(r'\{[\s\S]*"format"[\s\S]*\}', raw_input)
        if json_match:
            try:
                format_spec = json.loads(json_match.group())
                output_format = OutputFormat(
                    format=format_spec.get("format", "json"),
                    fields=format_spec.get("fields", {})
                )
                # Remove JSON from task content
                task_content = raw_input.replace(json_match.group(), "").strip()
                return task_content, output_format
            except:
                pass
        
        # Default: plain text output
        return raw_input, None
    
    def _identify_agents(self, task_content: str) -> List[str]:
        """Identify which agents are needed based on the task."""
        task_lower = task_content.lower()
        identified = []
        
        for agent, triggers in self.AGENT_TRIGGERS.items():
            if any(trigger in task_lower for trigger in triggers):
                identified.append(agent)
        
        # If no specific agents identified, default to company research
        if not identified:
            identified = ["company_research"]
        
        # SDR Enhancement: If both company and contact discovery are requested,
        # automatically add qualification and personalization for a complete SDR workflow
        if "company_research" in identified and "contact_discovery" in identified:
            # Always qualify leads and create personalized outreach for SDR tasks
            if "lead_qualification" not in identified:
                identified.append("lead_qualification")
            if "outreach_personalization" not in identified:
                identified.append("outreach_personalization")
        
        # Add dependencies
        all_agents = set(identified)
        for agent in identified:
            if agent in self.AGENT_DEPENDENCIES:
                all_agents.update(self.AGENT_DEPENDENCIES[agent])
        
        return list(all_agents)
    
    def _create_execution_plan(self, agents: List[str]) -> Dict[str, Any]:
        """Create an execution plan for the identified agents."""
        # For SDR workflow, ensure proper sequencing
        if all(agent in agents for agent in ["company_research", "contact_discovery", 
                                               "lead_qualification", "outreach_personalization"]):
            # Full SDR workflow with optimal sequencing
            return {
                "phase_1": {
                    "parallel": ["company_research"],
                    "sequential": []
                },
                "phase_2": {
                    "parallel": ["contact_discovery", "lead_qualification"],
                    "sequential": []
                },
                "phase_3": {
                    "parallel": ["outreach_personalization"],
                    "sequential": []
                }
            }
        
        # Otherwise use the original logic
        phase1_parallel = []
        phase1_sequential = []
        phase2_parallel = []
        phase2_sequential = []
        
        # Categorize agents based on dependencies
        for agent in agents:
            deps = self.AGENT_DEPENDENCIES.get(agent, [])
            
            # If agent has no dependencies, it can run in phase 1
            if not deps or not any(dep in agents for dep in deps):
                phase1_parallel.append(agent)
            else:
                # Agent has dependencies - check if all deps are in phase 1
                if all(dep in phase1_parallel for dep in deps if dep in agents):
                    phase2_parallel.append(agent)
                else:
                    phase2_sequential.append(agent)
        
        return {
            "phase_1": {
                "parallel": phase1_parallel,
                "sequential": phase1_sequential
            },
            "phase_2": {
                "parallel": phase2_parallel,
                "sequential": phase2_sequential
            }
        }
    
    def get_next_agents(self, state: SDRState) -> List[str]:
        """Determine which agents to run next based on current state."""
        execution_plan = state.get("execution_plan", {})
        completed_agents = state.get("completed_agents", [])
        
        # Handle new phased format with 'phases' array
        if "phases" in execution_plan:
            phases = execution_plan["phases"]
            for phase in phases:
                phase_agents = phase.get("agents", [])
                # Find agents in this phase that haven't been completed
                pending_agents = [a for a in phase_agents if a not in completed_agents]
                if pending_agents:
                    return pending_agents
            
            # All phases complete
            return []
        
        # Handle old format with phase_1, phase_2, etc.
        elif "phase_1" in execution_plan:
            # Process phases in order
            for phase_name in sorted(execution_plan.keys()):
                phase = execution_plan[phase_name]
                phase_parallel = phase.get("parallel", [])
                phase_sequential = phase.get("sequential", [])
                
                # Check parallel agents in this phase
                pending_parallel = [a for a in phase_parallel if a not in completed_agents]
                if pending_parallel:
                    return pending_parallel
                
                # Check sequential agents in this phase
                for agent in phase_sequential:
                    if agent not in completed_agents:
                        # Check dependencies
                        deps = state.get("agent_dependencies", {}).get(agent, [])
                        if all(dep in completed_agents for dep in deps):
                            return [agent]
            
            # All phases complete
            return []
        else:
            # Flat structure
            parallel_agents = execution_plan.get("parallel", [])
            sequential_agents = execution_plan.get("sequential", [])
        
        # First run parallel agents
        pending_parallel = [a for a in parallel_agents if a not in completed_agents]
        
        if pending_parallel:
            return pending_parallel
        
        # Then run sequential agents
        for agent in sequential_agents:
            if agent not in completed_agents:
                # Check if dependencies are met
                deps = self.AGENT_DEPENDENCIES.get(agent, [])
                if all(dep in completed_agents for dep in deps):
                    return [agent]
        
        return []  # All agents completed 

    def _detect_query_complexity(self, query: str) -> str:
        """Detect if query is complex and requires reasoning model."""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_score = 0
        matched_categories = []
        
        for category, indicators in self.COMPLEXITY_INDICATORS.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity_score += 1
                matched_categories.append(category)
        
        # Check query length (longer queries tend to be more complex)
        if len(query.split()) > 20:
            complexity_score += 1
            matched_categories.append("long_query")
        
        # Check if multiple agents are needed
        identified_agents = self._identify_agents(query)
        if len(identified_agents) > 2:
            complexity_score += 1
            matched_categories.append("multi_agent")
        
        # Determine model based on complexity
        if complexity_score >= 2:
            model = "o4-mini"
            print(f"\n[ROUTER] Complex query detected (score: {complexity_score})")
            print(f"  - Complexity indicators: {matched_categories}")
            print(f"  - Using model: {model}")
        else:
            model = "gpt-4o"
            print(f"\n[ROUTER] Simple query detected")
            print(f"  - Using model: {model}")
        
        return model 