"""Web search tool using DuckDuckGo for real-time information."""

import asyncio
from typing import List, Dict, Any
from ddgs import DDGS

class WebSearchTool:
    """Web search tool for real-time information retrieval."""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict with search results and metadata
        """
        try:
            # Run the sync search in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.text(query, max_results=max_results))
            )
            
            if results:
                # Format results
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "snippet": r.get("body", ""),
                        "source": "DuckDuckGo"
                    })
                
                # Create a text summary for LLM processing
                text_summary = "\n\n".join([
                    f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                    for r in formatted_results
                ])
                
                return {
                    "success": True,
                    "query": query,
                    "results": formatted_results,
                    "text_summary": text_summary,
                    "num_results": len(formatted_results)
                }
            else:
                return {
                    "success": False,
                    "query": query,
                    "results": [],
                    "text_summary": "No results found",
                    "num_results": 0
                }
                
        except Exception as e:
            print(f"[WEB SEARCH ERROR] {str(e)[:200]}")
            return {
                "success": False,
                "query": query,
                "results": [],
                "text_summary": f"Search error: {str(e)[:100]}",
                "num_results": 0,
                "error": str(e)
            }
    
    async def search_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for news articles.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dict with news results
        """
        try:
            # Run the sync search in an executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(self.ddgs.news(query, max_results=max_results))
            )
            
            if results:
                # Format news results
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", "DuckDuckGo News")
                    })
                
                # Create text summary
                text_summary = "\n\n".join([
                    f"Title: {r['title']}\nDate: {r['date']}\nSource: {r['source']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                    for r in formatted_results
                ])
                
                return {
                    "success": True,
                    "query": query,
                    "results": formatted_results,
                    "text_summary": text_summary,
                    "num_results": len(formatted_results)
                }
            else:
                return {
                    "success": False,
                    "query": query,
                    "results": [],
                    "text_summary": "No news results found",
                    "num_results": 0
                }
                
        except Exception as e:
            print(f"[WEB SEARCH NEWS ERROR] {str(e)[:200]}")
            return {
                "success": False,
                "query": query,
                "results": [],
                "text_summary": f"News search error: {str(e)[:100]}",
                "num_results": 0,
                "error": str(e)
            } 