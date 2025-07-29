"""Improved contact discovery agent with robust fallback strategies."""

import re
import asyncio
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


class ImprovedContactDiscoveryAgent(SmartScrapingMixin):
    """Improved agent with robust contact discovery strategies."""
    
    def __init__(self, mcp_client: MCPSubprocessClient, llm: Optional[ChatOpenAI]):
        """Initialize the improved contact discovery agent."""
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.mcp_client = mcp_client
        self.email_enrichment = EmailEnrichmentService()
        
        # Configuration for retries and timeouts
        self.max_retries = 3
        self.base_timeout = 30  # seconds
        self.backoff_factor = 1.5
    
    @trace_agent("contact_discovery_improved")
    async def discover_contacts(self, state: SDRState) -> SDRState:
        """Discover contacts with improved error handling and fallback strategies."""
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
        
        # Extract requested role from user query
        user_query = state.get("user_query", "").lower()
        requested_role = self._extract_requested_role(user_query)
        
        print(f"\n[CONTACT DISCOVERY] Starting improved search for {company_name}")
        print(f"  Target role: {requested_role}")
        
        # Try multiple strategies with fallbacks
        contacts = []
        citations = []
        
        # Strategy 1: Try SDR Intelligence with retry logic
        try:
            sdr_result = await self._try_with_retry(
                self._search_with_sdr_intelligence,
                company_name,
                requested_role
            )
            if sdr_result:
                contacts.extend(sdr_result["contacts"])
                citations.extend(sdr_result["citations"])
        except Exception as e:
            print(f"[WARNING] SDR Intelligence failed: {str(e)[:100]}")
        
        # Strategy 2: Conference speaker search (doesn't require LinkedIn)
        if len(contacts) < 3:
            try:
                conference_result = await self._search_conference_speakers(
                    company_name,
                    requested_role
                )
                if conference_result:
                    contacts.extend(conference_result["contacts"])
                    citations.extend(conference_result["citations"])
            except Exception as e:
                print(f"[WARNING] Conference search failed: {str(e)[:100]}")
        
        # Strategy 3: Press release and news search
        if len(contacts) < 3:
            try:
                news_result = await self._search_news_mentions(
                    company_name,
                    requested_role
                )
                if news_result:
                    contacts.extend(news_result["contacts"])
                    citations.extend(news_result["citations"])
            except Exception as e:
                print(f"[WARNING] News search failed: {str(e)[:100]}")
        
        # Deduplicate contacts
        unique_contacts = self._deduplicate_contacts(contacts)
        
        # Enrich contacts with email addresses
        domain = self._extract_domain(company_website) if company_website else None
        enriched_contacts = await self._enrich_contacts_with_emails(
            unique_contacts[:5],  # Limit to top 5 contacts
            company_name,
            domain
        )
        
        # Get company insights
        company_insights = state.get("company_insights", {})
        if not company_insights:
            try:
                sdr_intel = SDRIntelligence(self.mcp_client)
                company_insights = await sdr_intel.get_company_insights(company_name)
                state["company_insights"] = company_insights
            except Exception as e:
                print(f"[WARNING] Failed to get company insights: {str(e)[:100]}")
                company_insights = {}
        
        # Prepare result
        primary_contact = enriched_contacts[0] if enriched_contacts else None
        
        # Detect email pattern
        email_pattern = None
        if domain and enriched_contacts:
            email_pattern = self._detect_email_pattern_from_contacts(enriched_contacts, domain)
        
        # Create result
        result = AgentResult(
            agent_name="contact_discovery",
            data={
                "company": company_name,
                "contacts_found": len(enriched_contacts),
                "contacts": enriched_contacts,
                "primary_contact": primary_contact,
                "email_pattern": email_pattern,
                "requested_role": requested_role,
                "company_insights": company_insights
            },
            citations=list(set(citations)),  # Deduplicate citations
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
        
        print(f"\n[CONTACT DISCOVERY] Complete! Found {len(enriched_contacts)} contacts.")
        return state
    
    async def _try_with_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries):
            timeout = self.base_timeout * (self.backoff_factor ** attempt)
            
            try:
                print(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} (timeout: {timeout}s)")
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                print(f"[TIMEOUT] {last_error}")
            except Exception as e:
                last_error = str(e)[:200]
                print(f"[ERROR] {last_error}")
            
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"[RETRY] Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed after {self.max_retries} attempts. Last error: {last_error}")
    
    async def _search_with_sdr_intelligence(self, company_name: str, requested_role: str):
        """Search using SDR Intelligence with timeout protection."""
        sdr_intel = SDRIntelligence(self.mcp_client)
        leader_result = await sdr_intel.find_engineering_leader(company_name, requested_role)
        
        if leader_result:
            contact = {
                "name": leader_result.get("name", "Unknown"),
                "title": leader_result.get("role", requested_role),
                "department": "Engineering",
                "linkedin": leader_result.get("linkedin_url", "")
            }
            return {
                "contacts": [contact],
                "citations": [f"Google Search: {company_name} {requested_role}"]
            }
        return None
    
    async def _search_conference_speakers(self, company_name: str, requested_role: str):
        """Search for conference speakers from the company."""
        queries = [
            f'"{company_name}" conference speaker "{requested_role}"',
            f'"{company_name}" tech talk presenter engineering',
            f'"{company_name}" summit keynote technology'
        ]
        
        contacts = []
        citations = []
        
        for query in queries[:2]:  # Limit queries
            try:
                result = await self.mcp_client.call_tool(
                    "search_engine",
                    {"query": query, "engine": "google"}
                )
                
                if result and "content" in result:
                    content = result.get("content", [])
                    if isinstance(content, list) and content:
                        html_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        
                        # Extract speaker information using LLM
                        speaker_info = await self._extract_speaker_info(html_text, company_name)
                        if speaker_info:
                            contacts.extend(speaker_info)
                            citations.append(f"Conference search: {query}")
                            
            except Exception as e:
                print(f"[WARNING] Conference search error: {str(e)[:100]}")
        
        return {"contacts": contacts, "citations": citations} if contacts else None
    
    async def _search_news_mentions(self, company_name: str, requested_role: str):
        """Search for people mentioned in news articles."""
        queries = [
            f'"{company_name}" announces appoints "{requested_role}"',
            f'"{company_name}" hires promotes engineering leader'
        ]
        
        contacts = []
        citations = []
        
        for query in queries[:1]:  # Limit queries
            try:
                result = await self.mcp_client.call_tool(
                    "search_engine",
                    {"query": query, "engine": "google"}
                )
                
                if result and "content" in result:
                    content = result.get("content", [])
                    if isinstance(content, list) and content:
                        html_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        
                        # Extract people from news
                        news_contacts = await self._extract_news_contacts(html_text, company_name)
                        if news_contacts:
                            contacts.extend(news_contacts)
                            citations.append(f"News search: {query}")
                            
            except Exception as e:
                print(f"[WARNING] News search error: {str(e)[:100]}")
        
        return {"contacts": contacts, "citations": citations} if contacts else None
    
    async def _extract_speaker_info(self, html_text: str, company_name: str) -> List[Dict[str, Any]]:
        """Extract speaker information from conference listings."""
        # Simple extraction - in production, use LLM-based extraction
        # For now, return empty to avoid errors
        return []
    
    async def _extract_news_contacts(self, html_text: str, company_name: str) -> List[Dict[str, Any]]:
        """Extract contact information from news articles."""
        # Simple extraction - in production, use LLM-based extraction
        # For now, return empty to avoid errors
        return []
    
    async def _enrich_contacts_with_emails(self, contacts: List[Dict[str, Any]], 
                                         company_name: str, domain: str) -> List[Dict[str, Any]]:
        """Enrich contacts with email addresses."""
        if not contacts:
            return []
        
        print(f"\n[EMAIL ENRICHMENT] Enriching {len(contacts)} contacts...")
        
        try:
            # Ensure all contacts have company name
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
                    print(f"  ✓ {enriched['name']}: {enriched['email']} (confidence: {enriched['email_confidence']:.2f})")
                else:
                    print(f"  ✗ {enriched['name']}: No email found")
                    
        except Exception as e:
            print(f"[EMAIL ENRICHMENT ERROR] {str(e)[:100]}")
            # Continue without emails if enrichment fails
        
        return contacts
    
    def _extract_requested_role(self, user_query: str) -> str:
        """Extract the requested role from user query."""
        role_patterns = {
            "VP of Engineering": ["vp of engineering", "vp engineering", "vice president.*engineering"],
            "CTO": ["cto", "chief technology officer"],
            "CEO": ["ceo", "chief executive officer"],
            "VP of Product": ["vp.*product", "vice president.*product"],
            "Director of Engineering": ["director.*engineering", "engineering director"],
            "Head of Engineering": ["head.*engineering"],
            "Engineering Manager": ["engineering manager", "manager.*engineering"]
        }
        
        for role, patterns in role_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_query, re.IGNORECASE):
                    return role
        
        # Default to VP of Engineering for SDR targeting
        return "VP of Engineering"
    
    def _deduplicate_contacts(self, contacts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate contacts by name."""
        seen_names = set()
        unique_contacts = []
        
        for contact in contacts:
            name = contact.get("name", "").lower()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_contacts.append(contact)
        
        return unique_contacts
    
    def _detect_email_pattern_from_contacts(self, contacts: List[Dict[str, Any]], domain: str) -> str:
        """Detect email pattern from enriched contacts."""
        patterns = []
        
        for contact in contacts:
            if contact.get("email") and contact.get("email_confidence", 0) > 0.7:
                email = contact["email"]
                name = contact.get("name", "").lower().split()
                
                if len(name) >= 2 and "@" in email:
                    local_part = email.split("@")[0]
                    first = name[0]
                    last = name[-1]
                    
                    # Detect pattern
                    if local_part == f"{first}.{last}":
                        patterns.append("first.last")
                    elif local_part == f"{first}{last}":
                        patterns.append("firstlast")
                    elif local_part == f"{first[0]}{last}":
                        patterns.append("flast")
                    elif local_part == f"{first}_{last}":
                        patterns.append("first_last")
        
        # Return most common pattern
        if patterns:
            most_common = max(set(patterns), key=patterns.count)
            return f"{most_common}@{domain}"
        
        return f"first.last@{domain}"  # Default pattern
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        return domain.replace("www.", "").strip("/") 