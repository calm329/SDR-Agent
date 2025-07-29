"""Smart scraping utilities for handling Brightdata policy blocks and tool selection."""
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

class SmartScrapingMixin:
    """
    Mixin for smart web scraping with intelligent fallback strategies.
    """
    
    # JavaScript-heavy sites that need Browser API
    JS_HEAVY_DOMAINS = {
        'greenhouse.io', 'lever.co', 'workday.com', 'ashbyhq.com', 
        'jobvite.com', 'breezy.hr', 'smartrecruiters.com'
    }
    
    async def scrape_with_fallback(self, url: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Smart scraping with fallback strategies.
        Uses Web Unlocker for most sites, Browser API for JS-heavy sites.
        """
        if not hasattr(self, 'mcp_client'):
            raise AttributeError("SmartScrapingMixin requires 'mcp_client' attribute")
        
        domain = self._extract_domain(url)
        
        try:
            # 1. Handle JS-heavy sites with Browser API
            if domain in self.JS_HEAVY_DOMAINS:
                print(f"[SMART SCRAPE] Using Browser API for JS-heavy site: {domain}")
                await self.mcp_client.call_tool(
                    "scraping_browser_navigate", 
                    {"url": url}
                )
                result = await self.mcp_client.call_tool(
                    "scraping_browser_get_text", 
                    {}
                )
            
            # 2. Use Web Unlocker for everything else (including LinkedIn, Spotify, etc.)
            else:
                print(f"[SMART SCRAPE] Using Web Unlocker (scrape_as_markdown) for {domain}")
                result = await self.mcp_client.call_tool(
                    "scrape_as_markdown", 
                    {"url": url}
                )
            
            # Process the result from the successful scrape
            if result and "content" in result:
                content = result.get("content")
                
                # Check if the content is actually an error message
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        text = first_item.get("text", "")
                        
                        # Check for BrightData policy errors
                        if "policy_20050" in text or "requires special permission" in text:
                            print(f"[SMART SCRAPE] Site requires KYC permission from BrightData")
                            # Fall back to search
                            entity_name = context.get("company_name") or context.get("person_name") or self._extract_domain(url)
                            return await self._generic_fallback(url, entity_name)
                        
                        # Check for other errors
                        if text.lower().startswith("tool") and "failed" in text.lower():
                            print(f"[SMART SCRAPE] Error in response: {text[:100]}")
                            # Fall back to search
                            entity_name = context.get("company_name") or context.get("person_name") or self._extract_domain(url)
                            return await self._generic_fallback(url, entity_name)
                
                # If we got here, the scraping was successful
                return {
                    "content": content,
                    "citation": {"method": "direct_scrape", "url": url},
                    "success": True
                }
            else:
                raise Exception("Empty result from scraping, initiating fallback")
                
        except Exception as e:
            print(f"[SMART SCRAPE] Initial scrape failed: {str(e)[:100]}")
            # If scraping fails, try a simple search fallback
            entity_name = context.get("company_name") or context.get("person_name") or self._extract_domain(url)
            return await self._generic_fallback(url, entity_name)
    
    async def _linkedin_person_fallback(self, person_name: str, company_name: str) -> Dict[str, Any]:
        """Fallback strategy for LinkedIn person profiles using SERP API."""
        print(f"[FALLBACK] LinkedIn person blocked - using SERP API search")
        
        search_queries = [
            f'"{person_name}" "{company_name}" site:linkedin.com',
            f'"{person_name}" "{company_name}" "VP" OR "Chief" OR "Head of"',
            f'"{person_name}" {company_name} conference speaker biography'
        ]
        
        all_results = []
        for query in search_queries[:2]:  # Use first 2 queries
            try:
                # Use SERP API for better search results
                result = await self.mcp_client.call_tool(
                    "search_engine",
                    {
                        "query": query, 
                        "engine": "google"
                    }
                )
                
                if result and "content" in result:
                    content = result.get("content", [])
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            all_results.append(item.get("text", ""))
            except Exception as e:
                print(f"[FALLBACK] SERP search failed: {str(e)[:50]}")
        
        combined_text = "\n\n".join(all_results)
        return {
            "content": combined_text or "No information found through alternative sources",
            "citation": {"method": "serp_fallback", "queries": search_queries[:2]},
            "success": bool(combined_text),
            "fallback": True
        }
    
    async def _linkedin_company_fallback(self, company_name: str) -> Dict[str, Any]:
        """Fallback strategy for LinkedIn company profiles using multiple sources."""
        print(f"[FALLBACK] LinkedIn company blocked - trying alternative sources")
        
        # Try multiple sources
        sources = []
        
        # 1. Try company website directly (but skip if it's a blocked domain)
        try:
            company_domain = company_name.lower().replace(" ", "") + ".com"
            # Try to scrape the company website directly
            website_result = await self.mcp_client.call_tool(
                "scrape_as_markdown",
                {"url": f"https://www.{company_domain}/about"}
            )
            if website_result and "content" in website_result:
                sources.append(("Company website", website_result.get("content")))
        except:
            pass
        
        # 2. Use SERP API for company info and executives
        try:
            # Search for company info
            print(f"[FALLBACK] Searching for {company_name} company info...")
            search_result = await self.mcp_client.call_tool(
                "search_engine",
                {
                    "query": f"{company_name} company profile employees funding headquarters",
                    "engine": "google"
                }
            )
            if search_result and "content" in search_result:
                sources.append(("Google search", search_result.get("content")))
                print(f"[FALLBACK] Found company info")
            
            # Also search specifically for executives
            print(f"[FALLBACK] Searching for {company_name} executives...")
            exec_result = await self.mcp_client.call_tool(
                "search_engine",
                {
                    "query": f'"{company_name}" "VP of Engineering" OR "Vice President Engineering" -jobs -careers',
                    "engine": "google"
                }
            )
            if exec_result and "content" in exec_result:
                sources.append(("Executive search", exec_result.get("content")))
                print(f"[FALLBACK] Found executive info")
        except Exception as e:
            print(f"[FALLBACK ERROR] SERP search failed: {str(e)[:100]}")
            pass
        
        # 3. Try Crunchbase/Bloomberg via SERP
        try:
            print(f"[FALLBACK] Searching business directories for {company_name}...")
            news_result = await self.mcp_client.call_tool(
                "search_engine",
                {
                    "query": f'"{company_name}" site:crunchbase.com OR site:bloomberg.com',
                    "engine": "google"
                }
            )
            if news_result and "content" in news_result:
                sources.append(("Business directories", news_result.get("content")))
                print(f"[FALLBACK] Found business directory info")
        except Exception as e:
            print(f"[FALLBACK ERROR] Business directory search failed: {str(e)[:100]}")
            pass
        
        # Combine results
        combined_content = []
        for source_name, content in sources:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        combined_content.append(f"[{source_name}] {item.get('text', '')}")
            else:
                combined_content.append(f"[{source_name}] {content}")
        
        print(f"[FALLBACK] LinkedIn company fallback completed with {len(sources)} sources")
        
        return {
            "content": "\n\n".join(combined_content) or "No company information found",
            "citation": {"method": "multi_source_fallback", "sources": [s[0] for s in sources]},
            "success": bool(combined_content),
            "fallback": True
        }
    
    async def _generic_fallback(self, url: str, entity_name: str) -> Dict[str, Any]:
        """Generic fallback for any blocked site using SERP."""
        print(f"[FALLBACK] Generic fallback for {url}")
        
        # Try to search for information about the entity using SERP
        try:
            result = await self.mcp_client.call_tool(
                "search_engine",
                {
                    "query": f'"{entity_name}" -site:{urlparse(url).netloc}',
                    "engine": "google"
                }
            )
            
            if result and "content" in result:
                return {
                    "content": result.get("content"),
                    "citation": {"method": "serp_fallback", "original_url": url},
                    "success": True,
                    "fallback": True
                }
        except:
            pass
        
        return {
            "content": f"Unable to access {url} due to policy restrictions",
            "citation": {"method": "blocked", "url": url},
            "success": False,
            "fallback": True,
            "policy_blocked": True
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    async def search_with_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced search using SERP API for better results."""
        try:
            # Try direct Google search URL scraping
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            
            print(f"[SEARCH] Trying direct Google search for: {query[:50]}...")
            
            # Use scrape_as_markdown directly
            result = await self.mcp_client.call_tool(
                "scrape_as_markdown",
                {"url": search_url}
            )
            
            if result and "content" in result:
                content = result.get("content", [])
                
                if isinstance(content, list) and content:
                    # Get the text content
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_content += item.get("text", "") + "\n"
                        elif isinstance(item, str):
                            text_content += item + "\n"
                    
                    if text_content:
                        # Parse search results from the text
                        from .serp_parser import SERPParser
                        serp_parser = SERPParser()
                        
                        # Try to extract meaningful content from the search results
                        lines = text_content.split('\n')
                        search_results = []
                        
                        # Simple extraction of search results
                        for i, line in enumerate(lines):
                            line = line.strip()
                            # Look for patterns that indicate search results
                            if line and len(line) > 20 and not line.startswith('[') and 'Google' not in line:
                                # Check if next line might be a URL
                                if i + 1 < len(lines):
                                    next_line = lines[i + 1].strip()
                                    if next_line.startswith('http') or next_line.startswith('www.'):
                                        search_results.append({
                                            'title': line,
                                            'url': next_line,
                                            'snippet': lines[i + 2].strip() if i + 2 < len(lines) else ""
                                        })
                        
                        if search_results:
                            print(f"[SEARCH] Found {len(search_results)} results from direct scraping")
                            return {
                                "success": True,
                                "content": [{
                                    "text": "\n\n".join([
                                        f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                                        for r in search_results[:5]
                                    ])
                                }]
                            }
                        else:
                            # Return raw content if parsing fails
                            return {
                                "success": True,
                                "content": [{"text": text_content[:3000]}]  # Limit content size
                            }
                
            # If direct scraping fails, try the search_engine tool with timeout handling
            try:
                import asyncio
                print(f"[SEARCH] Falling back to search_engine tool...")
                
                # Set a shorter timeout for search_engine
                result = await asyncio.wait_for(
                    self.mcp_client.call_tool(
                        "search_engine",
                        {
                            "query": query,
                            "engine": "google"
                        }
                    ),
                    timeout=10.0  # 10 second timeout
                )
                
                if result:
                    return result
                    
            except asyncio.TimeoutError:
                print(f"[SEARCH] search_engine tool timed out after 10s")
            except Exception as e:
                print(f"[SEARCH] search_engine tool failed: {str(e)[:100]}")
                
        except Exception as e:
            print(f"[SEARCH ERROR] Search failed: {str(e)[:100]}")
        
        # Return empty result on failure
        return {
            "success": False,
            "content": []
        } 