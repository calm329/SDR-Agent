"""Company research agent for gathering comprehensive company information."""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import re

from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from core.prompt_manager import prompt_manager
from core.state import AgentResult, SDRState
from utils.tracing import trace_agent
from tools.brightdata_mcp_subprocess import MCPSubprocessClient
from tools.smart_scraping import SmartScrapingMixin


class CompanyInfo(BaseModel):
    """Structured company information."""
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    size: str = Field(description="Company size (employees)")
    location: str = Field(description="Headquarters location") 
    website: str = Field(description="Company website URL")
    description: str = Field(description="Brief company description")
    products: List[str] = Field(default_factory=list, description="Main products/services")
    recent_news: List[Dict[str, str]] = Field(default_factory=list, description="Recent news items")


class CompanyResearchAgent(SmartScrapingMixin):
    """Agent specialized in researching company information."""
    
    def __init__(self, mcp_client: MCPSubprocessClient, llm: ChatOpenAI):
        """Initialize the company research agent."""
        self.llm = llm
        self.mcp_client = mcp_client
    
    @trace_agent("company_research")
    async def research_company(self, state: SDRState) -> SDRState:
        """Research a company based on the task content."""
        task = state["task_content"]
        citations = []
        
        try:
            # Extract company name from task using an LLM
            company_name = await self._extract_company_name_with_llm(task)
            print(f"[INFO] Researching company: {company_name}")
            
            # Try to find the company's official website
            website_url = None
            citations = []
            
            # Common URL patterns to try
            possible_urls = [
                f"https://www.{company_name.lower().replace(' ', '')}.com",
                f"https://{company_name.lower().replace(' ', '')}.com",
                f"https://www.{company_name.lower().replace(' ', '-')}.com"
            ]
            
            for url in possible_urls:
                try:
                    print(f"[INFO] Testing URL: {url}")
                    
                    # Use smart scraping with fallback
                    scraped = await self.scrape_with_fallback(
                        url, 
                        {"company_name": company_name}
                    )
                    
                    if scraped and scraped.get("success") and scraped.get("content"):
                        website_url = url
                        print(f"[SUCCESS] Found valid website: {url}")
                        break
                except Exception as e:
                    print(f"[FAILED] URL failed: {url} - {str(e)[:50]}")
                    continue
            
            # Initialize data collection
            company_data = {
                "name": company_name,
                "website": website_url or "Unknown"
            }
            
            # Gather information from multiple sources
            all_content = []
            
            # First, try LinkedIn for comprehensive company info
            linkedin_url = await self._find_linkedin_url(company_name)
            linkedin_scraped = False
            has_employee_count = False
            
            # Try to get LinkedIn data
            if linkedin_url:
                print(f"[INFO] Found LinkedIn URL: {linkedin_url}")
                
                # First try the specialized LinkedIn company profile tool
                try:
                    print(f"[INFO] Using LinkedIn company profile tool for {company_name}")
                    linkedin_result = await self.mcp_client.call_tool(
                        "web_data_linkedin_company_profile",
                        {"url": linkedin_url}
                    )
                    
                    if linkedin_result and "content" in linkedin_result:
                        content = linkedin_result.get("content", [])
                        if isinstance(content, list) and content:
                            # Extract structured LinkedIn data
                            linkedin_data = content[0] if isinstance(content[0], dict) else {}
                            
                            # Extract basic info from LinkedIn data
                            if linkedin_data:
                                if linkedin_data.get("description"):
                                    company_data["description"] = linkedin_data["description"]
                                if linkedin_data.get("employee_count"):
                                    company_data["size"] = f"{linkedin_data['employee_count']} employees"
                                    has_employee_count = True
                                if linkedin_data.get("industry"):
                                    company_data["industry"] = linkedin_data["industry"]
                                if linkedin_data.get("headquarters"):
                                    company_data["location"] = linkedin_data["headquarters"]
                                    
                                # Also add as content for LLM
                                linkedin_content = f"""
                                Company: {linkedin_data.get('name', company_name)}
                                Description: {linkedin_data.get('description', '')}
                                Industry: {linkedin_data.get('industry', '')}
                                Employees: {linkedin_data.get('employee_count', '')}
                                Headquarters: {linkedin_data.get('headquarters', '')}
                                """
                                all_content.append(linkedin_content)
                                linkedin_scraped = True
                                print("[INFO] Successfully extracted LinkedIn company data")
                except Exception as e:
                    print(f"[ERROR] LinkedIn company profile tool failed: {str(e)[:100]}")
                
                # If specialized tool fails, try regular scraping
                if not linkedin_scraped:
                    try:
                        result = await self.scrape_with_fallback(linkedin_url, {"company_name": company_name})
                        
                        if result and result.get('success'):
                            linkedin_content = result.get('content', '')
                            if isinstance(linkedin_content, list) and linkedin_content:
                                linkedin_content = linkedin_content[0].get("text", "") if isinstance(linkedin_content[0], dict) else str(linkedin_content[0])
                            
                            all_content.append(linkedin_content)
                            citations.append(f"LinkedIn: {linkedin_url}")
                            linkedin_scraped = True
                            
                            # Check if LinkedIn has employee count
                            import re
                            employee_patterns = [
                                r'View all ([\d,]+) employees',
                                r'([\d,]+)\s*employees',
                                r'Company size\s*[\n\r]*([\d,\-]+)',
                            ]
                            
                            for pattern in employee_patterns:
                                if re.search(pattern, linkedin_content, re.IGNORECASE):
                                    has_employee_count = True
                                    break
                            
                            if has_employee_count:
                                print("[INFO] Found employee count in LinkedIn data")
                    except Exception as e:
                        print(f"[ERROR] LinkedIn scraping failed: {str(e)[:100]}")
            
            # Try to scrape the company website (if we didn't get complete LinkedIn data)
            if website_url and not (linkedin_scraped and has_employee_count):
                print(f"[INFO] Scraping company website: {website_url}")
                
                try:
                    # Scrape the main website  
                    result = await self.scrape_with_fallback(website_url, {"company_name": company_name})
                    
                    if result and result.get('success'):
                        content = result.get('content', '')
                        if isinstance(content, list) and content:
                            content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        
                        all_content.append(content)
                        citations.append(f"Website: {website_url}")
                except Exception as e:
                    print(f"[ERROR] Failed to scrape {website_url}: {str(e)}")
            
            # If we still need more info, search for information
            if not all_content or not has_employee_count:
                print(f"[INFO] Searching for more info about {company_name}")
                
                # First, do a general company search to get basic info
                from tools.web_search import WebSearchTool
                web_search = WebSearchTool()
                
                try:
                    general_query = f"{company_name} company description employees products services"
                    search_result = await web_search.search(general_query, max_results=5)
                    
                    if search_result.get("success") and search_result.get("text_summary"):
                        all_content.append(search_result['text_summary'])
                        citations.append(f"Web search for {company_name}")
                        print(f"[WEB SEARCH] Added general company info")
                except Exception as e:
                    print(f"[ERROR] General web search failed: {str(e)[:100]}")
                
                # Then do specific employee count search if needed
                search_queries = [
                    f"{company_name} number of employees how many people work at {company_name}",
                    f'"{company_name}" "employees" "headcount" "staff size" 2024 2023'
                ]
                
                for query in search_queries[:1]:  # Start with first query
                    search_result = await self.search_with_context(
                        query,
                        {"company_name": company_name}
                    )
                    
                    if search_result and search_result.get("content"):
                        content = search_result.get("content", [])
                        search_text = ""
                        
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    search_text += item.get("text", "") + "\n"
                        elif isinstance(content, str):
                            search_text = content
                        
                        if search_text:
                            all_content.append(search_text)
                            citations.append(f"Web search for {company_name}")
                            break
            
            # Check if user is asking for funding information specifically
            user_query = state.get("user_query", "").lower()
            funding_data = None
            
            if any(keyword in user_query for keyword in ["funding", "raised", "investment", "investor", "series", "valuation"]):
                print(f"[INFO] User is asking for funding information - searching...")
                
                # Use our WebSearchTool
                from tools.web_search import WebSearchTool
                web_search = WebSearchTool()
                
                funding_queries = [
                    f"{company_name} funding round raised million billion series 2023 2024",
                    f'"{company_name}" "raised" "funding" "series" "led by" "valuation"',
                    f"{company_name} latest funding investors venture capital"
                ]
                
                # Collect all funding search results
                all_funding_results = []
                funding_citations = []
                
                for query in funding_queries[:2]:  # Search with first two queries
                    try:
                        print(f"[WEB SEARCH] Searching: {query[:50]}...")
                        search_result = await web_search.search(query, max_results=5)
                        
                        if search_result.get("success") and search_result.get("text_summary"):
                            all_funding_results.append(search_result['text_summary'])
                            # Add URLs as citations
                            for r in search_result.get("results", [])[:3]:
                                if r.get("url"):
                                    funding_citations.append(r["url"])
                            print(f"[WEB SEARCH] Found {search_result['num_results']} results")
                    except Exception as e:
                        print(f"[ERROR] Web search failed: {str(e)[:100]}")
                
                # Extract funding info using dedicated method
                if all_funding_results:
                    combined_results = "\n\n".join(all_funding_results)
                    funding_data = await self._extract_funding_info_with_llm(company_name, combined_results)
                    
                    # Add funding citations
                    citations.extend(funding_citations[:5])  # Limit to 5 citations
            
            # Use LLM to extract structured information from all content
            if all_content:
                extracted_info = await self._extract_company_info_with_llm(
                    company_name,
                    "\n\n".join(all_content)
                )
                
                # Merge extracted info with company_data
                company_data.update(extracted_info)
                
                # Add funding data if available
                if funding_data:
                    company_data["funding_rounds"] = funding_data.get("funding_rounds", [])
                    
                    # Add latest round info to recent_news if available
                    latest_round = funding_data.get("latest_round", {})
                    if latest_round and latest_round.get("amount"):
                        funding_summary = f"{company_name} raised {latest_round.get('amount', 'undisclosed amount')} in {latest_round.get('round_type', 'funding round')} at {latest_round.get('valuation', 'undisclosed valuation')} ({latest_round.get('date', 'recently')})"
                        if "recent_news" not in company_data:
                            company_data["recent_news"] = []
                        company_data["recent_news"].insert(0, funding_summary)
                
                # If still missing critical info, try Wikipedia
                missing_keys = [k for k in ("industry", "size", "location") if company_data.get(k, "Unknown") in ("Unknown", "")]
                if missing_keys:
                    try:
                        print(f"[INFO] Attempting Wikipedia API for: {company_name}")
                        import httpx
                        
                        # Try different Wikipedia page names for disambiguation
                        wiki_names = [company_name.replace(' ', '_')]
                        
                        # Add common disambiguation suffixes for known companies
                        if company_name.lower() == "stripe":
                            wiki_names = ["Stripe_(company)", "Stripe,_Inc."]
                        elif company_name.lower() == "apple":
                            wiki_names = ["Apple_Inc.", "Apple_(company)"]
                        elif company_name.lower() == "amazon":
                            wiki_names = ["Amazon_(company)", "Amazon.com"]
                        
                        wiki_text = ""
                        successful_url = None
                        
                        for wiki_name in wiki_names:
                            try:
                                # Use Wikipedia API to get clean text
                                api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"
                                async with httpx.AsyncClient(timeout=10) as client:
                                    resp = await client.get(api_url, headers={"User-Agent": "SDRAgent/1.0"})
                                
                                if resp.status_code == 200:
                                    wiki_data = resp.json()
                                    wiki_text = wiki_data.get('extract', '')
                                    
                                    # Check if it's not a disambiguation page
                                    if wiki_text and "may refer to:" not in wiki_text:
                                        successful_url = f"https://en.wikipedia.org/wiki/{wiki_name}"
                                        break
                            except:
                                continue
                        
                        if wiki_text and successful_url:
                            # For companies like Stripe/Walmart, we need more than just the extract
                            # Let's search for specific information in the page
                            search_query = f"{company_name} employees headquarters industry revenue"
                            search_result = await self.search_with_context(
                                search_query,
                                {"company_name": company_name}
                            )
                            
                            combined_text = wiki_text
                            if search_result and search_result.get("content"):
                                search_content = search_result.get("content", [])
                                if isinstance(search_content, list) and search_content:
                                    for item in search_content[:2]:  # Take first 2 results
                                        if isinstance(item, dict) and "text" in item:
                                            combined_text += "\n\n" + item.get("text", "")
                            
                            if combined_text and len(combined_text) > 100:
                                # Use a more specific prompt for extraction
                                enhanced_content = f"Company: {company_name}\n\n{combined_text[:3000]}"
                                wiki_info = await self._extract_company_info_with_llm(company_name, enhanced_content)
                                
                                # Update only missing keys
                                for k, v in wiki_info.items():
                                    if k in missing_keys and v not in ("Unknown", ""):
                                        company_data[k] = v
                                
                                if any(company_data.get(k, "Unknown") != "Unknown" for k in missing_keys):
                                    citations.append(f"Wikipedia: {successful_url}")
                    except Exception as e:
                        print(f"[WARN] Wikipedia API fallback failed: {str(e)[:100]}")
            else:
                # Fallback to basic info if no content available
                company_data.update({
                    "industry": "Unknown",
                    "size": "Unknown",
                    "location": "Unknown",
                    "description": "",
                    "products": [],
                    "recent_news": []
                })
            
            # Create the result
            result = AgentResult(
                agent_name="company_research",
                data=company_data,
                citations=citations if citations else ["No web sources available - using cached data"],
                timestamp=datetime.now().isoformat(),
                error=None
            )
            
            # Update state
            current_results = state.get("agent_results", {})
            current_results["company_research"] = result
            state["agent_results"] = current_results
            state["current_phase"] = "company_research_complete"
            
            # Mark as completed
            if "completed_agents" not in state:
                state["completed_agents"] = []
            state["completed_agents"].append("company_research")
            
            print(f"[SUCCESS] Company research completed for: {company_name}")
            
        except Exception as e:
            print(f"[ERROR] Company research error: {str(e)}")
            error_result = AgentResult(
                agent_name="company_research",
                data={},
                citations=[],
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )
            current_results = state.get("agent_results", {})
            current_results["company_research"] = error_result
            state["agent_results"] = current_results
            state["current_phase"] = "company_research_error"
        
        return state
    
    async def _extract_company_name_with_llm(self, task: str) -> str:
        """Extracts the company name from a user query using an LLM."""
        prompt = prompt_manager.get_prompt("company_name_extractor")
        
        if not prompt:
            raise ValueError("Company name extractor prompt not found.")
            
        formatted_prompt = prompt.format(user_query=task)
        
        response = await self.llm.ainvoke(formatted_prompt)
        
        # The response should be just the company name
        company_name = response.content.strip()
        
        # Basic cleaning
        if company_name.endswith((".", ",")):
            company_name = company_name[:-1]
            
        return company_name 
    
    async def _extract_company_info_with_llm(self, company_name: str, scraped_content: str) -> Dict[str, Any]:
        """Extract structured company information using LLM."""
        prompt = prompt_manager.get_prompt("company_extraction")
        
        if not prompt:
            raise ValueError("Company extraction prompt not found.")
        
        # Truncate content if too long (keep first 3000 chars)
        if len(scraped_content) > 3000:
            scraped_content = scraped_content[:3000] + "..."
        
        # Also print if we see any employee-related keywords
        employee_patterns = [
            r'\b\d{1,3}(?:,\d{3})*\s*(?:employees?|people|staff|workforce)\b',
            r'\bteam\s*(?:of|size)?\s*\d+',
            r'\b(?:employs?|workforce\s*of)\s*\d+',
        ]
        
        import re
        matches = []
        for pattern in employee_patterns:
            found = re.findall(pattern, scraped_content, re.IGNORECASE)
            matches.extend(found)
        
        formatted_prompt = prompt.format_messages(
            company_name=company_name,
            scraped_content=scraped_content
        )
        
        response = await self.llm.ainvoke(formatted_prompt)
        
        try:
            # Parse JSON response
            response_text = response.content.strip()
            
            # Handle case where LLM wraps JSON in markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
                
            extracted_info = json.loads(response_text.strip())
            
            # Ensure all required fields are present
            default_fields = {
                "description": "Unknown",
                "industry": "Unknown", 
                "size": "Unknown",
                "location": "Unknown",
                "products": [],
                "recent_news": [],
                "funding_rounds": []
            }
            
            for key, default_value in default_fields.items():
                if key not in extracted_info:
                    extracted_info[key] = default_value
            
            # Ensure all required fields exist
            return {
                "description": extracted_info.get("description", ""),
                "industry": extracted_info.get("industry", "Unknown"),
                "size": extracted_info.get("size", "Unknown"),
                "location": extracted_info.get("location", "Unknown"),
                "products": extracted_info.get("products", []),
                "recent_news": extracted_info.get("recent_news", []),
                "funding_rounds": extracted_info.get("funding_rounds", [])
            }
        except json.JSONDecodeError:
            print(f"[WARNING] Failed to parse LLM response as JSON: {response.content[:200]}")
            return {
                "description": "",
                "industry": "Unknown",
                "size": "Unknown",
                "location": "Unknown",
                "products": [],
                "recent_news": []
            } 

    async def _extract_funding_info_with_llm(self, company_name: str, search_results: str) -> Dict[str, Any]:
        """Extract funding information using LLM."""
        funding_prompt = """Extract funding information from these search results about {company_name}.

Search Results:
{search_results}

Extract ALL funding rounds mentioned and return a JSON object with:
- funding_rounds: array of objects with:
  - date: when the funding happened
  - amount: how much was raised
  - round_type: Series A/B/C, Seed, etc.
  - valuation: company valuation if mentioned
  - lead_investors: array of investor names
  - purpose: what the funding will be used for
- latest_round: the most recent funding round details

Return ONLY valid JSON. If no funding information found, return empty arrays."""
        
        try:
            formatted_prompt = funding_prompt.format(
                company_name=company_name,
                search_results=search_results[:5000]  # Limit to 5000 chars
            )
            
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            funding_data = json.loads(response_text.strip())
            
            return funding_data
            
        except Exception as e:
            print(f"[ERROR] Failed to extract funding info: {str(e)[:100]}")
            return {
                "funding_rounds": [],
                "latest_round": {}
            }

    async def _find_linkedin_url(self, company_name: str) -> Optional[str]:
        """Find the LinkedIn URL for a company."""
        try:
            # Search for the company's LinkedIn page
            query = f"site:linkedin.com/company {company_name}"
            result = await self.mcp_client.call_tool(
                "search_engine",
                {"query": query, "engine": "google"}
            )
            
            if result and "content" in result:
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    text = content[0].get("text", "")
                    
                    # Extract LinkedIn company URL from search results
                    import re
                    linkedin_pattern = r'https?://(?:www\.)?linkedin\.com/company/[a-zA-Z0-9-]+/?'
                    matches = re.findall(linkedin_pattern, text)
                    
                    if matches:
                        # Clean up the URL
                        linkedin_url = matches[0].rstrip('/')
                        return linkedin_url
        except Exception as e:
            print(f"[ERROR] Failed to find LinkedIn URL: {str(e)}")
        
        return None
    
    async def _scrape_linkedin(self, linkedin_url: str) -> Optional[str]:
        """Scrape LinkedIn company page."""
        try:
            result = await self.mcp_client.call_tool(
                "scrape_as_markdown",
                {"url": linkedin_url}
            )
            
            if result and "content" in result:
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    return content[0].get("text", "")
        except Exception as e:
            print(f"[ERROR] Failed to scrape LinkedIn: {str(e)}")
        
        return None
    
 