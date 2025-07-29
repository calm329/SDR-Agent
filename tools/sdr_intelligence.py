"""SDR Intelligence module for extracting actionable sales insights."""

from typing import Dict, Any, List, Optional
import re
from .serp_parser import SERPParser

class SDRIntelligence:
    """Extract SDR-relevant insights using Brightdata tools intelligently."""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.serp_parser = SERPParser()
    
    async def get_company_insights(self, company_name: str) -> Dict[str, Any]:
        """Get actionable company insights for SDR outreach."""
        insights = {
            "tech_stack": [],
            "recent_news": [],
            "growth_signals": [],
            "pain_points": [],
            "initiatives": []
        }
        
        # Try Crunchbase first for funding/growth data
        try:
            print(f"[SDR Intel] Checking Crunchbase for {company_name}")
            crunchbase_result = await self.mcp_client.call_tool(
                "web_data_crunchbase_company",
                {"query": company_name}
            )
            
            if crunchbase_result and "content" in crunchbase_result:
                content = crunchbase_result.get("content", [])
                if isinstance(content, list) and content:
                    data = content[0] if isinstance(content[0], dict) else {}
                    
                    # Extract funding info
                    if data.get("last_funding_type"):
                        insights["growth_signals"].append(f"Recent {data.get('last_funding_type')} funding")
                    if data.get("num_employees_enum"):
                        insights["growth_signals"].append(f"Company size: {data.get('num_employees_enum')}")
        except Exception as e:
            print(f"[SDR Intel] Crunchbase error: {str(e)[:100]}")
        
        # Search for tech stack using SERP
        tech_queries = [
            f'"{company_name}" "tech stack" "engineering blog" "how we built"',
            f'"{company_name}" "uses Kubernetes" OR "uses AWS" OR "uses Python" OR "uses React"',
            f'site:stackshare.io "{company_name}"'
        ]
        
        for query in tech_queries[:2]:  # Limit to avoid timeout
            try:
                result = await self.mcp_client.call_tool(
                    "search_engine",
                    {"query": query, "engine": "google"}
                )
                
                if result and "content" in result:
                    content = result.get("content", [])
                    if isinstance(content, list) and content:
                        html_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        
                        # Use SERP parser to extract tech stack
                        tech_found = await self.serp_parser.extract_tech_stack(html_text, company_name)
                        insights["tech_stack"].extend(tech_found)
                        
                        # Also parse search results for insights
                        search_results = self.serp_parser.parse_search_results(html_text)
                        for result in search_results[:3]:
                            if 'engineering' in result.get('title', '').lower():
                                insights["initiatives"].append(result.get('snippet', '')[:200])
                                
            except Exception as e:
                print(f"[SDR Intel] Tech search error: {str(e)[:100]}")
        
        # Search for recent news and pain points
        news_query = f'"{company_name}" "announced" OR "launches" OR "partners with" 2023 2024'
        try:
            result = await self.mcp_client.call_tool(
                "search_engine", 
                {"query": news_query, "engine": "google"}
            )
            
            if result and "content" in result:
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    html_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                    
                    # Parse search results
                    search_results = self.serp_parser.parse_search_results(html_text)
                    for result in search_results[:5]:
                        snippet = result.get('snippet', '')
                        if any(word in snippet.lower() for word in ['announced', 'launches', 'partners', 'raises']):
                            insights["recent_news"].append(snippet[:200])
                            
        except Exception as e:
            print(f"[SDR Intel] News search error: {str(e)[:100]}")
        
        # Deduplicate
        insights["tech_stack"] = list(set(insights["tech_stack"]))[:10]
        insights["recent_news"] = insights["recent_news"][:5]
        insights["initiatives"] = insights["initiatives"][:3]
        
        return insights
    
    async def find_engineering_leader(self, company_name: str, target_role: str = "VP of Engineering") -> Optional[Dict[str, Any]]:
        """Find engineering leaders using SERP parsing."""
        print(f"[SDR Intel] Searching for {target_role} at {company_name}")
        
        # Search queries optimized for finding people
        queries = [
            f'"{company_name}" "{target_role}" site:linkedin.com/in/',
            f'"{company_name}" "{target_role}" "announces" OR "appoints" OR "promotes"',
            f'"{target_role}" "{company_name}" -jobs -careers'
        ]
        
        all_people = []
        
        for query in queries:
            try:
                result = await self.mcp_client.call_tool(
                    "search_engine",
                    {"query": query, "engine": "google"}
                )
                
                if result and "content" in result:
                    content = result.get("content", [])
                    if isinstance(content, list) and content:
                        html_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        
                        # Use SERP parser to extract person info
                        people = await self.serp_parser.extract_person_info(html_text, company_name, target_role)
                        all_people.extend(people)
                        
            except Exception as e:
                print(f"[SDR Intel] Search failed: {str(e)[:100]}")
        
        # Deduplicate and return best match
        seen_names = set()
        unique_people = []
        for person in all_people:
            name = person.get('name', 'Unknown')
            if name and name not in seen_names:
                seen_names.add(name)
                unique_people.append(person)
        
        # Return the first person with a LinkedIn URL, or the first person found
        for person in unique_people:
            if person.get('linkedin_url'):
                return person
        
        return unique_people[0] if unique_people else None
    
    def generate_personalization_hooks(self, company_insights: Dict[str, Any], contact_info: Dict[str, Any]) -> List[str]:
        """Generate specific personalization hooks based on insights."""
        hooks = []
        
        # Tech stack hooks
        tech_stack = company_insights.get("tech_stack", [])
        if "Kubernetes" in tech_stack:
            hooks.append("I noticed you're using Kubernetes - we help companies optimize their container orchestration and reduce costs by 40%")
        if "AWS" in tech_stack:
            hooks.append("Since you're on AWS, you might be interested in our cloud cost optimization tools")
        if any(tech in tech_stack for tech in ["Python", "Go", "Java"]):
            hooks.append(f"Your {[t for t in ['Python', 'Go', 'Java'] if t in tech_stack][0]} engineering team might benefit from our developer productivity tools")
        
        # Growth signal hooks
        for signal in company_insights.get("growth_signals", []):
            if "funding" in signal.lower():
                hooks.append(f"Congratulations on your {signal} - perfect timing to scale your engineering infrastructure")
            if "size:" in signal:
                hooks.append(f"With your growing team ({signal}), maintaining engineering velocity becomes crucial")
        
        # Initiative hooks
        for initiative in company_insights.get("initiatives", [])[:2]:
            if len(initiative) > 50:
                hooks.append(f"I saw your team's work on {initiative[:100]}... impressive!")
        
        # Recent news hooks
        for news in company_insights.get("recent_news", [])[:2]:
            if "announced" in news.lower() or "launches" in news.lower():
                hooks.append(f"Congrats on {news[:100]}...")
        
        return hooks[:5]  # Return top 5 hooks 