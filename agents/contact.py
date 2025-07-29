"""Contact discovery agent for finding decision makers."""

import re
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from datetime import datetime
import json

from core.prompt_manager import prompt_manager
from core.state import AgentResult, SDRState
from utils.tracing import trace_agent
from tools.brightdata_mcp_subprocess import MCPSubprocessClient
from tools.smart_scraping import SmartScrapingMixin
from tools.sdr_intelligence import SDRIntelligence
from tools.email_enrichment import EmailEnrichmentService


class ContactDiscoveryAgent(SmartScrapingMixin):
    """Agent responsible for discovering contacts at target companies."""
    
    def __init__(self, mcp_client: MCPSubprocessClient, llm: Optional[ChatOpenAI]):
        """Initialize the contact discovery agent."""
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.mcp_client = mcp_client
        self.email_enrichment = EmailEnrichmentService()
    
    @trace_agent("contact_discovery")
    async def discover_contacts(self, state: SDRState) -> SDRState:
        """Discover contacts based on company research using SDR Intelligence."""
        # Get company info from previous agent
        company_result = state.get("agent_results", {}).get("company_research")
        
        if not company_result or not hasattr(company_result, 'data') or not company_result.data:
            # No company data available, return error
            result = AgentResult(
                agent_name="contact_discovery",
                data={"error": "No company research available"},
                citations=[],
                timestamp=datetime.now().isoformat()
            )
            
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["contact_discovery"] = result
            if "completed_agents" not in state:
                state["completed_agents"] = []
            state["completed_agents"].append("contact_discovery")
            return state
        
        company_info = company_result.data
        company_name = company_info.get("name", "Unknown")
        company_website = company_info.get("website", "")
        
        # Extract requested role from the user query
        user_query = state.get("task_content", state.get("raw_input", "")).lower()
        
        # SDR-optimized role extraction
        requested_role = None
        
        # Check for specific role requests in user query
        if "vp of engineering" in user_query or "vp engineering" in user_query or "vice president" in user_query and "engineering" in user_query:
            requested_role = "VP of Engineering"
        elif "cto" in user_query or "chief technology officer" in user_query:
            requested_role = "CTO"
        elif "ceo" in user_query or "chief executive officer" in user_query:
            requested_role = "CEO"
        elif "vp" in user_query and "product" in user_query:
            requested_role = "VP of Product"
        elif "director" in user_query and "engineering" in user_query:
            requested_role = "Director of Engineering"
        else:
            # Default to VP of Engineering for SDR targeting
            requested_role = "VP of Engineering"
        
        # Initialize SDR Intelligence
        sdr_intel = SDRIntelligence(self.mcp_client)
        
        contacts = []
        citations = []
        
        # Use SDR Intelligence module to find engineering leaders
        leader_result = await sdr_intel.find_engineering_leader(company_name, requested_role)
        
        if leader_result:
            # Found a real person
            contact = {
                "name": leader_result.get("name", "Unknown"),
                "title": leader_result.get("role", requested_role),
                "department": "Engineering",
                "linkedin": leader_result.get("linkedin_url", "")
            }
            contacts.append(contact)
            citations.append(f"Google Search: {company_name} {requested_role}")
        
        # If no specific role found, try alternative roles
        if not contacts:
            alternative_roles = ["CTO", "Head of Engineering", "VP Technology", "Engineering Director"]
            for alt_role in alternative_roles:
                if alt_role != requested_role:
                    alt_result = await sdr_intel.find_engineering_leader(company_name, alt_role)
                    if alt_result:
                        contact = {
                            "name": alt_result.get("name", "Unknown"),
                            "title": alt_result.get("role", alt_role),
                            "department": "Engineering",
                            "linkedin": alt_result.get("linkedin_url", "")
                        }
                        contacts.append(contact)
                        citations.append(f"Google Search: {company_name} {alt_role}")
                        break
        
        # Get company insights for better context
        company_insights = await sdr_intel.get_company_insights(company_name)
        
        # Store insights in state for other agents to use
        if "company_insights" not in state:
            state["company_insights"] = company_insights
        
        # NO FAKE DATA - If no contacts found, be honest about it
        if not contacts:
            primary_contact = None
        else:
            primary_contact = contacts[0]
        
        # Enrich contacts with email addresses using EmailEnrichmentService
        domain = self._extract_domain(company_website) if company_website else None
        enriched_contacts = []
        
        if contacts and (domain or company_name):
            
            try:
                # Ensure all contacts have company name for enrichment
                for contact in contacts:
                    if "company" not in contact:
                        contact["company"] = company_name
                
                # Enrich all contacts in parallel
                enriched_contacts = await self.email_enrichment.bulk_enrich(
                    contacts,
                    domain=domain
                )
                
                # Update contacts with enriched data
                for i, enriched in enumerate(enriched_contacts):
                    if enriched.get("email"):
                        contacts[i]["email"] = enriched["email"]
                        contacts[i]["email_confidence"] = enriched.get("email_confidence", 0)
                        contacts[i]["email_source"] = enriched.get("source", "Unknown")
                    else:
                        pass # No print statements for email enrichment
                        
            except Exception as e:
                print(f"[EMAIL ENRICHMENT ERROR] {str(e)[:100]}")
                # Continue without emails if enrichment fails
        
        # Detect email pattern based on domain and context
        email_pattern_info = None
        
        if domain:
            # Use LLM to detect email pattern
            context = f"Company: {company_name}"
            if contacts:
                context += f"\nExample contact: {contacts[0].get('name', '')}"
            
            email_pattern_info = await self._detect_email_pattern_with_llm(domain, context)
            email_pattern = email_pattern_info.get("pattern")
        else:
            # If no website found, don't guess - be honest
            email_pattern = None
        
        # Create result
        result = AgentResult(
            agent_name="contact_discovery",
            data={
                "company": company_name,
                "contacts_found": len(contacts),
                "contacts": contacts,
                "primary_contact": primary_contact,
                "email_pattern": email_pattern,
                "requested_role": requested_role,
                "company_insights": company_insights  # Include for downstream agents
            },
            citations=citations,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in agent_results
        if "agent_results" not in state:
            state["agent_results"] = {}
        state["agent_results"]["contact_discovery"] = result
        
        # Mark as completed
        if "completed_agents" not in state:
            state["completed_agents"] = []
        state["completed_agents"].append("contact_discovery")
        
        return state
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace("www.", "").strip("/")
    
    def _guess_email_pattern(self, domain: str) -> str:
        """Guess email pattern for a domain using LLM."""
        # This is now handled by the async method below
        return f"first.last@{domain}"
    
    async def _detect_email_pattern_with_llm(self, domain: str, context: str = "") -> Dict[str, Any]:
        """Detect email pattern using LLM based on domain and any context."""
        try:
            prompt = prompt_manager.get_prompt("email_pattern_detector")
            if not prompt:
                return {
                    "pattern": f"first.last@{domain}",
                    "confidence": "low",
                    "reasoning": "Using default pattern"
                }
            
            formatted_prompt = prompt.format_messages(
                domain=domain,
                context=context or f"Company domain is {domain}"
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
            return result
            
        except Exception as e:
            print(f"[ERROR] Email pattern detection failed: {str(e)[:100]}")
            return {
                "pattern": f"first.last@{domain}",
                "confidence": "low",
                "reasoning": "Error in detection, using default"
            } 