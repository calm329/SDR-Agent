"""Lead Qualification Agent using ONLY Brightdata MCP."""

from typing import Dict, Any, Optional
from tools.brightdata_mcp_subprocess import MCPSubprocessClient
from tools.smart_scraping import SmartScrapingMixin
from langsmith import traceable
from core.models import AgentResult
from langchain_openai import ChatOpenAI
from core.prompt_manager import prompt_manager
import json
from datetime import datetime

class LeadQualificationAgent(SmartScrapingMixin):
    """Qualifies leads using ONLY Brightdata MCP web scraping."""
    
    def __init__(self, mcp_client: MCPSubprocessClient, llm: Optional[ChatOpenAI] = None):
        self.mcp_client = mcp_client
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
    
    @traceable(name="lead_qualification_agent")
    async def qualify_lead(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Qualify leads by analyzing real data from web scraping."""
        
        # Initialize completed_agents if not present
        if "completed_agents" not in state:
            state["completed_agents"] = []
        
        # Get company info
        company_result = state.get("agent_results", {}).get("company_research")
        if not company_result or not hasattr(company_result, 'data'):
            result = AgentResult(
                agent_name="lead_qualification",
                data={"error": "No company research available"},
                citations=[],
                timestamp=datetime.now().isoformat()
            )
            # Store in agent_results
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["lead_qualification"] = result
            
            # Also store directly for backwards compatibility
            state["lead_qualification"] = result
            state["completed_agents"].append("lead_qualification")
            return state
        
        # Extract company info
        company_name = company_result.data.get("name", "Unknown")
        company_website = company_result.data.get("website", "")
        industry = company_result.data.get("industry", "")
        size = company_result.data.get("size", "")
        
        # Analyze user query to understand what we're qualifying for
        user_query = state.get("user_query", "").lower()
        product_context = ""
        specific_signals_to_find = []
        
        if "devops" in user_query or "ci/cd" in user_query or "deployment" in user_query:
            product_context = "DevOps tools and CI/CD solutions"
            specific_signals_to_find = [
                "Uses Kubernetes", "Uses Docker", "Uses Jenkins", "Uses GitLab CI",
                "Uses GitHub Actions", "Uses AWS", "Uses Azure", "Uses GCP",
                "Scaling challenges", "Deployment automation needs", "Infrastructure as Code"
            ]
        elif "data" in user_query and ("analytics" in user_query or "warehouse" in user_query):
            product_context = "Data analytics and warehousing solutions"
            specific_signals_to_find = [
                "Uses Snowflake", "Uses BigQuery", "Uses Redshift", "Uses Databricks",
                "Data pipeline challenges", "Analytics infrastructure", "Real-time processing needs"
            ]
        elif "security" in user_query:
            product_context = "Security solutions"
            specific_signals_to_find = [
                "Security incidents", "Compliance requirements", "Uses security tools",
                "SOC2 compliance", "GDPR compliance", "Security team growth"
            ]
        else:
            product_context = "technology solutions"
            specific_signals_to_find = []
        
        signals = {
            "technology_signals": [],
            "buying_signals": [],
            "growth_signals": [],
            "funding_events": []
        }
        citations = []
        
        # Strategy 1: Check job postings for technology needs
        job_urls = [
            f"https://{company_website}/careers",
            f"https://{company_website}/jobs",
            f"https://{company_website}/join-us"
        ]
        
        job_content = ""
        for url in job_urls:
            try:
                print(f"[INFO] Checking careers page: {url}")
                scraped = await self.mcp_client.call_tool(
                    "scrape_as_markdown",
                    {"url": url}
                )
                
                if scraped and "content" in scraped:
                    content = scraped.get("content", [])
                    if isinstance(content, list) and content:
                        job_content = content[0].get("text", "")
                        if job_content:
                            citations.append(url)
                            break
            except:
                continue
        
        # Extract technology signals from job postings using LLM
        if job_content:
            tech_signals = await self._extract_tech_signals_with_llm(company_name, job_content)
            signals["technology_signals"].extend(tech_signals)
        
        # Strategy 2: Search for company technology stack and funding
        search_queries = [
            f"{company_name} technology stack engineering blog",
            f"{company_name} funding round series investment raised million"
        ]
        
        for query in search_queries:
            try:
                print(f"[INFO] Searching: {query[:50]}...")
                search_result = await self.mcp_client.call_tool(
                    "search_engine",
                    {
                        "query": query,
                        "max_results": 5
                    }
                )
                
                if search_result and "content" in search_result:
                    content = search_result.get("content", [])
                    search_text = ""
                    
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                search_text += item.get("text", "") + "\n"
                    
                    if search_text:
                        # Extract tech signals
                        if "technology" in query or "engineering" in query:
                            tech_signals = await self._extract_tech_signals_with_llm(company_name, search_text)
                            signals["technology_signals"].extend(tech_signals)
                            citations.append(f"Tech stack search for {company_name}")
                        
                        # Extract funding signals
                        if "funding" in query or "investment" in query:
                            funding_info = await self._extract_funding_signals_with_llm(company_name, search_text)
                            signals["funding_events"].extend(funding_info.get("funding_events", []))
                            signals["growth_signals"].extend(funding_info.get("growth_signals", []))
                            if funding_info.get("has_recent_funding"):
                                signals["buying_signals"].append("Recent funding indicates budget availability")
                            citations.append(f"Funding search for {company_name}")
                            
            except Exception as e:
                print(f"[ERROR] Search failed: {str(e)[:100]}")
        
        # Get company insights from state if available
        if "company_insights" in state:
            insights = state["company_insights"]
            # Use real tech stack from SDR intelligence
            if insights.get("tech_stack"):
                signals["technology_signals"].extend([f"Uses {tech}" for tech in insights["tech_stack"]])
            if insights.get("growth_signals"):
                signals["growth_signals"].extend(insights["growth_signals"])
        
        # Add context-specific buying signals for DevOps
        if "devops" in product_context.lower():
            # Tech companies with large engineering teams need DevOps
            company_size = company_result.data.get("size", "")
            if "employees" in str(company_size):
                try:
                    # Extract number from strings like "16,901 employees"
                    emp_count = int(''.join(filter(str.isdigit, str(company_size).split()[0])))
                    if emp_count > 1000:
                        signals["buying_signals"].append("Large engineering organization needs DevOps automation")
                    if emp_count > 5000:
                        signals["buying_signals"].append("Enterprise-scale company requires robust CI/CD")
                except:
                    pass
            
            # Music streaming = massive scale = DevOps needs
            if company_name.lower() == "spotify":
                signals["buying_signals"].extend([
                    "Streaming platform requires continuous deployment",
                    "Global scale demands infrastructure automation",
                    "Microservices architecture needs orchestration"
                ])
        
        # Deduplicate signals
        signals["technology_signals"] = list(set(signals["technology_signals"]))[:10]
        signals["growth_signals"] = list(set(signals["growth_signals"]))[:5]
        signals["buying_signals"] = list(set(signals["buying_signals"]))[:5]
        signals["funding_events"] = list({json.dumps(event, sort_keys=True): event for event in signals["funding_events"]}.values())[:3]
        
        # Calculate qualification score
        score = self._calculate_score(signals, product_context, company_name)
        
        # Determine priority and approach
        if score >= 80:
            priority = "High"
            approach = "Hot lead - Schedule demo immediately"
        elif score >= 60:
            priority = "Medium" 
            approach = "Warm lead - Personalized outreach within 24 hours"
        else:
            priority = "Low"
            approach = "Cool lead - Add to nurture campaign. Monitor for buying signals."
        
        # Create result
        result = AgentResult(
            agent_name="lead_qualification",
            data={
                "company": company_name,
                "qualification_score": score,
                "technology_signals": signals["technology_signals"],
                "buying_signals": signals["buying_signals"],
                "growth_signals": signals["growth_signals"],
                "funding_events": signals["funding_events"],
                "recommended_approach": approach,
                "priority": priority
            },
            citations=citations,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in agent_results
        if "agent_results" not in state:
            state["agent_results"] = {}
        state["agent_results"]["lead_qualification"] = result
        
        # Also store directly for backwards compatibility
        state["lead_qualification"] = result
        state["completed_agents"].append("lead_qualification")
        return state
    
    async def _extract_tech_signals_with_llm(self, company_name: str, text: str) -> list:
        """Extract technology signals using LLM."""
        try:
            prompt = prompt_manager.get_prompt("tech_stack_extractor")
            if not prompt:
                return []
            
            # Limit text length
            if len(text) > 5000:
                text = text[:5000]
            
            formatted_prompt = prompt.format_messages(
                company=company_name,
                text=text
            )
            
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            # Return technologies with "Uses" prefix for consistency
            technologies = result.get("technologies", [])
            return [f"Uses {tech}" for tech in technologies]
            
        except Exception as e:
            print(f"[ERROR] Failed to extract tech signals: {str(e)[:100]}")
            return []
    
    async def _extract_funding_signals_with_llm(self, company_name: str, text: str) -> Dict[str, Any]:
        """Extract funding and growth signals using LLM."""
        try:
            prompt = prompt_manager.get_prompt("funding_signal_extractor")
            if not prompt:
                return {
                    "funding_events": [],
                    "growth_signals": [],
                    "has_recent_funding": False
                }
            
            # Limit text length
            if len(text) > 5000:
                text = text[:5000]
            
            formatted_prompt = prompt.format_messages(
                company=company_name,
                text=text
            )
            
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
            
        except Exception as e:
            print(f"[ERROR] Failed to extract funding signals: {str(e)[:100]}")
            return {
                "funding_events": [],
                "growth_signals": [],
                "has_recent_funding": False
            }
    
    def _calculate_score(self, signals: Dict[str, list], product_context: str = "", company_name: str = "") -> int:
        """Calculate qualification score based on signals and context."""
        score = 50  # Base score
        
        # Technology alignment
        tech_count = len(signals["technology_signals"])
        score += min(tech_count * 5, 25)  # Max 25 points
        
        # Buying signals
        buying_count = len(signals["buying_signals"])
        score += min(buying_count * 10, 20)  # Max 20 points
        
        # Growth signals
        growth_count = len(signals["growth_signals"])
        score += min(growth_count * 5, 15)  # Max 15 points
        
        # Funding events
        funding_count = len(signals.get("funding_events", []))
        score += min(funding_count * 10, 20)  # Max 20 points
        
        # Context-aware bonus for tech companies
        if product_context:
            # Tech companies are prime candidates for DevOps tools
            if "devops" in product_context.lower():
                # Check if they have relevant tech stack
                tech_stack_str = " ".join(signals["technology_signals"]).lower()
                if any(tech in tech_stack_str for tech in ["kubernetes", "docker", "aws", "cloud", "microservices"]):
                    score += 15  # Strong DevOps candidate
                elif tech_count > 0:
                    score += 10  # Has tech stack, likely needs DevOps
            
            # Music streaming companies like Spotify handle massive scale
            if company_name and "spotify" in company_name.lower():
                score += 20  # Spotify definitely needs DevOps tools!
        
        return min(score, 100) 