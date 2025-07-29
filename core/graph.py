"""
LangGraph workflow for SDR Agent orchestration.
"""
import os
import asyncio
from typing import Any, Dict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

from core.state import SDRState
from agents.router import RouterAgent
from agents.company import CompanyResearchAgent
from agents.formatter import OutputFormatterAgent
from agents.contact import ContactDiscoveryAgent
from agents.qualification import LeadQualificationAgent
from agents.personalization import OutreachPersonalizationAgent
from utils.tracing import trace_agent
from tools.brightdata_mcp_subprocess import MCPSubprocessClient

# Import improved contact discovery if available
try:
    from agents.contact_improved import ImprovedContactDiscoveryAgent
    IMPROVED_CONTACT_AVAILABLE = True
except ImportError:
    IMPROVED_CONTACT_AVAILABLE = False

class SDRWorkflow:
    """Main workflow orchestrating all SDR agents."""
    
    def __init__(self):
        # Configuration loaded as needed
        
        # Initialize a single, shared MCP client
        self.mcp_client = MCPSubprocessClient()

        # Initialize agents with the shared client
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.current_model = "gpt-4o"  # Track current model
        self.router = RouterAgent(llm=self.llm)
        self.company_agent = CompanyResearchAgent(mcp_client=self.mcp_client, llm=self.llm)
        self.formatter = OutputFormatterAgent(llm=self.llm)
        
        # Use improved contact discovery if available and enabled
        use_improved = os.environ.get("USE_IMPROVED_CONTACT", "false").lower() == "true"
        if use_improved and IMPROVED_CONTACT_AVAILABLE:
            print("[INFO] Using ImprovedContactDiscoveryAgent")
            self.contact_agent = ImprovedContactDiscoveryAgent(mcp_client=self.mcp_client, llm=self.llm)
        else:
            self.contact_agent = ContactDiscoveryAgent(mcp_client=self.mcp_client, llm=self.llm)
            
        self.qualification_agent = LeadQualificationAgent(mcp_client=self.mcp_client, llm=self.llm)
        self.personalization_agent = OutreachPersonalizationAgent(mcp_client=self.mcp_client)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> Any:
        """Build the LangGraph state machine."""
        # Create the graph
        workflow = StateGraph(SDRState)
        
        # Add nodes with descriptions for better visualization
        workflow.add_node("router", self.route_query)
        workflow.add_node("company_research", self.run_company_research)
        workflow.add_node("formatter", self.format_output)
        workflow.add_node("parallel_executor", self.execute_parallel_agents)
        
        # Define the flow
        workflow.set_entry_point("router")
        
        # Router determines next steps
        workflow.add_conditional_edges(
            "router",
            self.determine_next_step,
            {
                "company_only": "company_research",
                "parallel": "parallel_executor",
                "format": "formatter"
            }
        )
        
        # After company research, check if more agents needed
        workflow.add_conditional_edges(
            "company_research",
            self.check_completion,
            {
                "formatter": "formatter",
                "continue": "parallel_executor"
            }
        )
        
        # After parallel execution, check if more agents needed
        workflow.add_conditional_edges(
            "parallel_executor",
            self.check_completion,
            {
                "formatter": "formatter",
                "continue": "parallel_executor"
            }
        )
        
        # Formatter always ends
        workflow.add_edge("formatter", END)
        
        # Compile with recursion limit
        return workflow.compile(checkpointer=None, interrupt_before=[], interrupt_after=[], debug=False)
    
    def _update_agent_models(self, model: str):
        """Update all agents to use the recommended model."""
        if model != self.current_model:
            print(f"\n[MODEL UPDATE] Switching from {self.current_model} to {model}")
            self.current_model = model
            
            # Create new LLM instance with recommended model
            # o4-mini only supports temperature=1 (default)
            if model == "o4-mini":
                self.llm = ChatOpenAI(model=model, temperature=1)
            else:
                self.llm = ChatOpenAI(model=model, temperature=0)
            
            # Update all agents with new LLM
            self.router.llm = self.llm
            self.company_agent.llm = self.llm
            self.contact_agent.llm = self.llm
            self.qualification_agent.llm = self.llm
            self.personalization_agent.llm = self.llm
            self.formatter.llm = self.llm
    
    async def route_query(self, state: SDRState) -> SDRState:
        """Router node that analyzes the query and plans execution."""
        # First run the router to get recommended model
        state = await self.router.analyze_query(state)
        
        # Update agents with recommended model
        if "recommended_model" in state:
            self._update_agent_models(state["recommended_model"])
        
        return state
    
    @trace_agent("company_research")
    async def run_company_research(self, state: SDRState) -> SDRState:
        """Company research node."""
        state = await self.company_agent.research_company(state)
        return state
    
    @trace_agent("parallel_executor")
    async def execute_parallel_agents(self, state: SDRState) -> SDRState:
        """Execute multiple agents in parallel."""
        import time
        
        # Check if we've exceeded the deadline
        if "execution_deadline" in state and time.time() > state["execution_deadline"]:
            print("[TIMEOUT CHECK] Execution deadline exceeded, stopping...")
            state["error_messages"].append("Execution timeout - stopping agent execution")
            return state
        
        # Get agents to run
        next_agents = self.router.get_next_agents(state)
        
        print(f"\n[PARALLEL EXECUTOR] Running agents: {next_agents}")
        
        if not next_agents:
            return state
        
        # Create tasks for parallel execution
        tasks = []
        agent_names = []
        
        # Check which agents haven't been run yet
        for agent_name in next_agents:
            if agent_name not in state.get("completed_agents", []):
                if agent_name == "company_research":
                    print(f"  - Adding task: {agent_name}")
                    tasks.append(self.company_agent.research_company(state.copy()))
                    agent_names.append(agent_name)
                elif agent_name == "contact_discovery":
                    print(f"  - Adding task: {agent_name}")
                    tasks.append(self.contact_agent.discover_contacts(state.copy()))
                    agent_names.append(agent_name)
                elif agent_name == "lead_qualification":
                    print(f"  - Adding task: {agent_name}")
                    tasks.append(self.qualification_agent.qualify_lead(state.copy()))
                    agent_names.append(agent_name)
                elif agent_name == "outreach_personalization":
                    print(f"  - Adding task: {agent_name}")
                    tasks.append(self.personalization_agent.create_outreach(state.copy()))
                    agent_names.append(agent_name)
            else:
                print(f"  - Skipping {agent_name} (already completed)")
        
        # Run agents in parallel with individual timeouts
        print(f"[PARALLEL EXECUTOR] Executing {len(tasks)} tasks...")
        if tasks:
            # Calculate remaining time
            remaining_time = state.get("execution_deadline", float('inf')) - time.time()
            if remaining_time <= 0:
                print("[TIMEOUT CHECK] No time remaining for agent execution")
                state["error_messages"].append("Execution timeout before agent execution")
                return state
            
            # Use a shorter timeout for individual agents
            agent_timeout = min(remaining_time, 60)  # Max 60 seconds per agent
            print(f"[PARALLEL EXECUTOR] Agent timeout: {agent_timeout:.1f}s")
            
            # Run with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=agent_timeout
                )
            except asyncio.TimeoutError:
                print("[PARALLEL EXECUTOR] Agent execution timeout!")
                state["error_messages"].append("Agent execution timeout")
                # Mark all agents as completed to avoid retry
                for agent_name in agent_names:
                    if agent_name not in state["completed_agents"]:
                        state["completed_agents"].append(agent_name)
                return state
            
            # Merge results back into state
            for i, (result, agent_name) in enumerate(zip(results, agent_names)):
                if isinstance(result, Exception):
                    print(f"[ERROR] Agent {agent_name} failed: {str(result)}")
                    state["error_messages"].append(f"{agent_name} error: {str(result)}")
                    # Still mark as completed to avoid infinite loop
                    if agent_name not in state["completed_agents"]:
                        state["completed_agents"].append(agent_name)
                else:
                    print(f"[SUCCESS] Agent {agent_name} completed")
                    # Merge the returned state
                    if isinstance(result, dict):
                        # Update agent_results
                        if "agent_results" in result:
                            state["agent_results"].update(result["agent_results"])
                        # Update completed_agents - ensure the agent is marked complete
                        if agent_name not in state["completed_agents"]:
                            state["completed_agents"].append(agent_name)
                        # Also check if agent added itself
                        if "completed_agents" in result:
                            for agent in result["completed_agents"]:
                                if agent not in state["completed_agents"]:
                                    state["completed_agents"].append(agent)
                        # Update citations
                        if "citations" in result:
                            state["citations"].extend(result["citations"])
        
        print(f"[PARALLEL EXECUTOR] Completed agents: {state['completed_agents']}")
        return state
    
    async def format_output(self, state: SDRState) -> SDRState:
        """Output formatting node."""
        # Debug: Available data
        agent_results = state.get("agent_results", {})
        
        # Call formatter
        return await self.formatter.format_output(state)
    
    def determine_next_step(self, state: SDRState) -> Literal["company_only", "parallel", "format"]:
        """Determine the next step after routing."""
        # Routing logic
        identified_agents = state.get("identified_agents", [])
        completed_agents = state.get("completed_agents", [])
        
        # If no agents identified, route to end
        if not identified_agents:
            return "format"
        
        # If only company research is needed
        if identified_agents == ["company_research"]:
            return "company_only"
        
        # Check if we have agents to run
        if identified_agents:
            # For phased plans or simple plans, go to parallel executor
            return "parallel"
        
        # Otherwise go straight to formatting
        return "format"
    
    def check_completion(self, state: SDRState) -> Literal["formatter", "continue"]:
        """Check if all agents have completed."""
        next_agents = self.router.get_next_agents(state)
        
        if next_agents:
            return "continue"
        else:
            return "formatter"
    
    async def run(self, user_input: str) -> Dict:
        """Main entry point to run the workflow with timeout and retry."""
        import time
        import asyncio
        
        MAX_EXECUTION_TIME = 120  # 2 minutes
        MAX_RETRIES = 1
        
        for attempt in range(MAX_RETRIES + 1):
            start_time = time.time()
            
            try:
                # Start the shared MCP client
                await self.mcp_client.start()
                await self.mcp_client.discover_tools()

                if attempt > 0:
                    print(f"\n[RETRY {attempt}/{MAX_RETRIES}] Retrying after timeout...")
                
                # Initialize state
                initial_state = {
                    "user_query": user_input,  # Add user_query for agents to access
                    "raw_input": user_input,
                    "task_content": "",
                    "output_format": None,
                    "identified_agents": [],
                    "execution_plan": {},
                    "agent_dependencies": {},
                    "completed_agents": [],
                    "agent_results": {},
                    "formatted_output": None,
                    "citations": [],
                    "error_messages": [],
                    "validation_attempts": 0,
                    "execution_deadline": time.time() + MAX_EXECUTION_TIME  # Add deadline
                }
                
                # Run the workflow with timeout
                try:
                    result = await asyncio.wait_for(
                        self.graph.ainvoke(initial_state, config={"recursion_limit": 10}),
                        timeout=MAX_EXECUTION_TIME
                    )
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "formatted_output": result.get("formatted_output"),
                        "output": result.get("formatted_output"),  # Add for compatibility
                        "citations": result.get("citations", []),
                        "execution_time": execution_time,
                        "errors": result.get("error_messages", []),
                        "state": result  # Include full state for debugging
                    }
                    
                except asyncio.TimeoutError:
                    execution_time = time.time() - start_time
                    print(f"\n[TIMEOUT] Execution exceeded {MAX_EXECUTION_TIME} seconds!")
                    
                    # Clean up any hanging MCP clients
                    await self._cleanup_resources()
                    
                    if attempt < MAX_RETRIES:
                        print(f"[RETRY] Will retry the request...")
                        await asyncio.sleep(2)  # Brief pause before retry
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Execution timeout after {MAX_EXECUTION_TIME} seconds. The system may be experiencing issues with web scraping or API calls.",
                            "formatted_output": None,
                            "citations": [],
                            "execution_time": execution_time,
                            "timeout": True
                        }
                
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"\n[ERROR] Execution failed: {str(e)[:200]}")
                
                # Clean up resources
                await self._cleanup_resources()
                
                if attempt < MAX_RETRIES and "timeout" not in str(e).lower():
                    print(f"[RETRY] Will retry after error...")
                    await asyncio.sleep(2)
                    continue
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "formatted_output": None,
                        "citations": [],
                        "execution_time": execution_time
                    }
        
        # Should never reach here, but just in case
        return {
            "success": False,
            "error": "Maximum retries exceeded",
            "formatted_output": None,
            "citations": [],
            "execution_time": time.time() - start_time
        }
    
    async def _cleanup_resources(self):
        """Clean up any hanging resources like MCP clients."""
        try:
            if self.mcp_client:
                await self.mcp_client.stop()
            print("[CLEANUP] Resources cleaned up")
        except Exception as e:
            print(f"[CLEANUP] Error during cleanup: {str(e)[:100]}")

def create_sdr_workflow() -> SDRWorkflow:
    """Factory function to create an SDR workflow instance."""
    return SDRWorkflow() 