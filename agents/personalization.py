"""Outreach Personalization Agent using ONLY Brightdata MCP."""

from typing import Dict, Any, Optional
from datetime import datetime
from tools.brightdata_mcp_subprocess import MCPSubprocessClient
from tools.smart_scraping import SmartScrapingMixin
from langsmith import traceable
from core.models import AgentResult

class OutreachPersonalizationAgent(SmartScrapingMixin):
    """Creates personalized outreach using ONLY Brightdata MCP web scraping."""
    
    def __init__(self, mcp_client: MCPSubprocessClient):
        self.mcp_client = mcp_client
    
    @traceable(name="outreach_personalization_agent")
    async def create_outreach(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create personalized outreach based on real scraped data."""
        
        # Initialize completed_agents if not present
        if "completed_agents" not in state:
            state["completed_agents"] = []
        
        # Get data from previous agents
        company_result = state.get("agent_results", {}).get("company_research")
        contact_result = state.get("agent_results", {}).get("contact_discovery") 
        qualification_result = state.get("agent_results", {}).get("lead_qualification")
        
        if not company_result:
            print("[ERROR] No company research data available")
            return state
        
        # Extract data properly from AgentResult objects
        company_data = company_result.data if hasattr(company_result, 'data') else company_result
        contact_data = contact_result.data if hasattr(contact_result, 'data') else contact_result if contact_result else {}
        qualification_data = qualification_result.data if hasattr(qualification_result, 'data') else qualification_result if qualification_result else {}
        
        company_name = company_data.get("name", "the company")
        
        # Check if we have a real contact
        contacts = contact_data.get("contacts", [])
        primary_contact = contact_data.get("primary_contact", {})
        
        if not contacts or not primary_contact:
            # NO FAKE PERSONALIZATION - Be honest about missing contact
            print("[INFO] No contact found - cannot generate personalized outreach")
            
            result = AgentResult(
                agent_name="outreach_personalization",
                data={
                    "error": "No contact information available",
                    "message": f"Unable to generate personalized outreach for {company_name} - no engineering contacts found in search results",
                    "suggestions": [
                        "Complete Brightdata KYC to access LinkedIn data",
                        "Try alternative contact discovery tools (Apollo.io, Hunter.io)",
                        "Search for conference speaker lists or GitHub profiles"
                    ]
                },
                citations=[],
                timestamp=datetime.now().isoformat()
            )
            
            # Store result
            if "agent_results" not in state:
                state["agent_results"] = {}
            state["agent_results"]["outreach_personalization"] = result
            
            # Mark as completed
            if "completed_agents" not in state:
                state["completed_agents"] = []
            state["completed_agents"].append("outreach_personalization")
            
            return state
        
        # Get real contact information
        target_name = primary_contact.get("name", "")
        target_title = primary_contact.get("title", "Engineering Leader")
        
        # Only proceed if we have a real name (not "Unknown" or "Engineering Leader")
        if not target_name or target_name in ["Unknown", "Engineering Leader", ""]:
            print(f"[INFO] Contact name is unknown - limited personalization possible")
            target_name = None
        
        personalization_data = {
            "company_insights": [],
            "contact_insights": [],
            "hooks": []
        }
        citations = []
        
        # If we have a LinkedIn URL, scrape it for personalization
        if primary_contact.get("linkedin"):
            linkedin_url = f"https://{primary_contact['linkedin']}"
            try:
                print(f"[INFO] Scraping LinkedIn profile: {linkedin_url}")
                scraped = await self.mcp_client.call_tool(
                    "scrape_as_markdown",
                    {"url": linkedin_url}
                )
                
                if scraped and "content" in scraped:
                    content = scraped.get("content", [])
                    if isinstance(content, list) and content:
                        linkedin_text = content[0].get("text", "")
                        
                        # Extract insights from LinkedIn
                        contact_insights = self._extract_linkedin_insights(linkedin_text)
                        if contact_insights:
                            personalization_data["contact_insights"].extend(contact_insights)
                            citations.append(linkedin_url)
            except:
                pass
        
        # Search for recent company news for personalization
        news_query = f"{company_data.get('name', '')} latest news announcement"
        try:
            print(f"[INFO] Searching for recent company news...")
            search_result = await self.mcp_client.call_tool(
                "search_engine",
                {
                    "query": news_query,
                    "max_results": 3
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
                    company_insights = self._extract_company_insights(search_text, company_data.get("name", ""))
                    if company_insights:
                        personalization_data["company_insights"].extend(company_insights)
                        citations.append(f"Recent news for {company_data.get('name', '')}")
        except:
            pass
        
        # Generate personalized outreach
        outreach = self._generate_outreach(
            company_data,
            primary_contact,
            qualification_data,
            personalization_data
        )
        
        # Create result
        result = AgentResult(
            agent_name="outreach_personalization",
            data={
                "target": primary_contact.get("name", "There"),
                "company": company_data.get("name", "your company"),
                "email_subject_lines": outreach["subject_lines"],
                "email_opening": outreach["opening"],
                "value_proposition": outreach["value_prop"],
                "social_proof": outreach["social_proof"],
                "call_to_action": outreach["cta"],
                "personalization_hooks": outreach["hooks"],
                "linkedin_message": outreach["linkedin_message"],
                "talk_track": outreach["talk_track"]
            },
            citations=citations,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in agent_results
        if "agent_results" not in state:
            state["agent_results"] = {}
        state["agent_results"]["outreach_personalization"] = result
        
        # Also store directly for backwards compatibility
        state["outreach_personalization"] = result
        state["completed_agents"].append("outreach_personalization")
        return state
    
    def _extract_linkedin_insights(self, text: str) -> list:
        """Extract personalization insights from LinkedIn profile."""
        insights = []
        text_lower = text.lower()
        
        # Look for recent activity
        if "posted" in text_lower or "shared" in text_lower:
            insights.append("Active on LinkedIn")
        
        # Look for common interests/topics
        tech_interests = ["ai", "machine learning", "devops", "cloud", "automation", "digital transformation"]
        for interest in tech_interests:
            if interest in text_lower:
                insights.append(f"Interested in {interest}")
        
        # Look for career progression
        if "promoted" in text_lower or "new role" in text_lower:
            insights.append("Recent role change")
        
        return insights
    
    def _extract_company_insights(self, text: str, company_name: str) -> list:
        """Extract company insights for personalization."""
        insights = []
        text_lower = text.lower()
        
        # Look for recent events
        if "announced" in text_lower or "launches" in text_lower:
            insights.append("Recent product/service announcement")
        if "funding" in text_lower or "raised" in text_lower:
            insights.append("Recent funding activity")
        if "partnership" in text_lower or "collaboration" in text_lower:
            insights.append("New partnership announced")
        if "expansion" in text_lower or "growth" in text_lower:
            insights.append("Company expansion")
        
        return insights
    
    def _generate_outreach(self, company_data: Dict, contact: Dict, qualification: Dict, personalization: Dict) -> Dict:
        """Generate personalized outreach based on all data."""
        company_name = company_data.get("name", "your company")
        contact_name = contact.get("name", "there").split()[0] if contact.get("name") else "there"
        contact_title = contact.get("title", "")
        
        # Generate subject lines
        subject_lines = []
        
        # Based on qualification score
        if qualification.get("qualification_score", 0) >= 80:
            subject_lines.append(f"Quick question about {company_name}'s DevOps initiatives")
            subject_lines.append(f"{contact_name} - solving deployment bottlenecks at {company_name}")
        else:
            subject_lines.append(f"{contact_name} - quick question about {company_name}'s engineering")
            subject_lines.append(f"Ideas for {company_name}'s tech stack optimization")
        
        # Based on recent insights
        if personalization.get("company_insights"):
            insight = personalization["company_insights"][0]
            subject_lines.append(f"Congrats on {company_name}'s {insight.lower()}")
        
        # Generate opening based on personalization
        hooks = []
        if personalization.get("company_insights"):
            hooks.append(f"I noticed {company_name}'s {personalization['company_insights'][0].lower()}")
        if qualification.get("technology_signals"):
            tech = qualification["technology_signals"][0].replace("Uses ", "")
            hooks.append(f"I see you're using {tech}")
        if company_data.get("recent_news"):
            hooks.append(f"Saw the news about {company_data['recent_news'][0].get('title', '')[:50]}...")
        
        opening = hooks[0] if hooks else f"I came across {company_name} and was impressed by what you're building"
        
        # Value proposition based on pain points
        value_props = []
        if qualification.get("technology_signals"):
            if any("CI/CD" in sig or "deployment" in sig for sig in qualification["technology_signals"]):
                value_props.append("automate your deployment pipeline and reduce release cycles by 70%")
            if any("Kubernetes" in sig or "Docker" in sig for sig in qualification["technology_signals"]):
                value_props.append("optimize your container orchestration and reduce infrastructure costs")
        else:
            value_props.append("streamline your engineering workflows and accelerate delivery")
        
        value_prop = value_props[0] if value_props else "help scale your engineering efficiency"
        
        # Social proof
        social_proof = f"We've helped companies similar to {company_name} {value_prop}"
        
        # Call to action
        cta = "Would you be open to a brief 15-minute call next week to explore if we could help?"
        
        # LinkedIn message
        linkedin_msg = f"Hi {contact_name}, {opening}. I'd love to connect and share some ideas on how we could {value_prop}. Open to a quick chat?"
        
        # Talk track key points
        talk_track = {
            "opening": f"Thanks for taking my call, {contact_name}. {opening}.",
            "qualifying_questions": [
                f"How is {company_name} currently handling deployments?",
                "What's the biggest bottleneck in your engineering workflow?",
                "How much time does your team spend on manual processes?"
            ],
            "value_points": [
                f"We can {value_prop}",
                "Most clients see ROI within 90 days",
                "No disruption to your existing workflow"
            ]
        }
        
        return {
            "subject_lines": subject_lines[:3],
            "opening": opening,
            "value_prop": f"We can {value_prop}",
            "social_proof": social_proof,
            "cta": cta,
            "hooks": hooks[:3],
            "linkedin_message": linkedin_msg,
            "talk_track": talk_track
        } 