"""
Brightdata MCP Client using subprocess to run the MCP server.

This implementation runs the Brightdata MCP server as a Node.js subprocess
and communicates with it using the Model Context Protocol over stdio.
"""
import os
import json
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langsmith import traceable
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BrightDataMCPConfig(BaseModel):
    """Configuration for Brightdata MCP client."""
    api_token: str = Field(default_factory=lambda: os.getenv("BRIGHTDATA_API_KEY", ""))
    unlocker_zone: str = Field(
        default_factory=lambda: os.getenv("BRIGHTDATA_WEB_UNLOCKER_ZONE", "web_unlocker1")
    )
    browser_zone: str = Field(
        default_factory=lambda: os.getenv("BRIGHTDATA_BROWSER_ZONE", "scraping_browser3")
    )
    serp_zone: str = Field(
        default_factory=lambda: os.getenv("BRIGHTDATA_SERP_ZONE", "serp_api1")
    )
    rate_limit: Optional[str] = Field(
        default=None, 
        description="Rate limit for requests (e.g., '10 per minute')"
    )


class MCPSubprocessClient:
    """Client that runs Brightdata MCP server as a subprocess."""
    
    def __init__(self, config: Optional[BrightDataMCPConfig] = None):
        """Initialize the MCP subprocess client."""
        self.config = config or BrightDataMCPConfig()
        self.process: Optional[subprocess.Popen] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.request_id = 0
        self.initialized = False
        self.tools = []
        self._read_buffer = ""
        self._response_futures = {}
        self._next_id = 1
        self._stop_event = asyncio.Event()
        self._reader_task = None
        # Add semaphore to limit concurrent requests
        self._request_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        
    async def __aenter__(self):
        """Start the MCP server subprocess."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop the MCP server subprocess."""
        await self.stop()
    
    async def start(self):
        """Start the MCP server subprocess."""
        if self.process:
            return  # Already started
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env['API_TOKEN'] = self.config.api_token
            env['WEB_UNLOCKER_ZONE'] = self.config.unlocker_zone
            env['BROWSER_ZONE'] = self.config.browser_zone
            env['SERP_ZONE'] = self.config.serp_zone
            
            # Command to run the MCP server
            cmd = ['npx', '-y', '@brightdata/mcp']
            
            # On Windows, try to find npx path
            if sys.platform == "win32":
                npx_result = subprocess.run(
                    ["where", "npx"], 
                    capture_output=True, 
                    text=True,
                    shell=True
                )
                if npx_result.returncode == 0:
                    npx_path = npx_result.stdout.strip().split('\n')[0]
                    cmd[0] = npx_path
            
            print("[INFO] Starting Brightdata MCP server...")
            print("[INFO] Configuration:")
            print(f"  API Token: {self.config.api_token[:10]}...")
            print(f"  Unlocker Zone: {self.config.unlocker_zone}")
            print(f"  Browser Zone: {self.config.browser_zone}")
            print(f"  SERP Zone: {self.config.serp_zone}")
            
            # Start the subprocess with UTF-8 encoding
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding='utf-8',  # Explicitly use UTF-8
                errors='replace',  # Replace invalid chars instead of failing
                shell=(sys.platform == "win32")  # Use shell on Windows
            )
            
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                raise Exception(f"MCP server failed to start: {stderr}")
            
            # Initialize the connection
            await self.initialize()
            
            # Discover tools
            await self.discover_tools()
            
            print("[SUCCESS] MCP server started successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to start MCP server: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the MCP server subprocess."""
        if self.process:
            self.process.terminate()
            await asyncio.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
            self.process = None
    
    async def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a request to the MCP server and wait for response."""
        async with self._request_semaphore:  # Limit concurrent requests
            if not self.process:
                raise Exception("Not connected to MCP server")
            
            REQUEST_TIMEOUT = 30  # 30 seconds timeout per request
            
            request_id = self.request_id
            self.request_id += 1
            
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method
            }
            
            if params:
                request["params"] = params
            
            try:
                # Write request
                request_str = json.dumps(request) + '\n'
                self.process.stdin.write(request_str)
                self.process.stdin.flush()
                
                # Read response with timeout using asyncio
                async def read_with_timeout():
                    import threading
                    result = [None, None]  # [response, exception]
                    
                    def read_line():
                        try:
                            result[0] = self.process.stdout.readline()
                        except Exception as e:
                            result[1] = e
                    
                    # Read in a separate thread
                    thread = threading.Thread(target=read_line)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait for the thread with timeout
                    start_time = asyncio.get_event_loop().time()
                    while thread.is_alive():
                        if asyncio.get_event_loop().time() - start_time > REQUEST_TIMEOUT:
                            # Timeout occurred
                            raise asyncio.TimeoutError(f"MCP request timeout after {REQUEST_TIMEOUT}s")
                        await asyncio.sleep(0.1)
                    
                    # Check result
                    if result[1]:
                        raise result[1]
                    return result[0]
                
                response_str = await read_with_timeout()
                
                if not response_str:
                    raise Exception("No response from MCP server")
                
                try:
                    response = json.loads(response_str.strip())
                except json.JSONDecodeError as e:
                    # Log more details about the parsing error
                    print(f"[MCP JSON ERROR] Failed to parse response")
                    print(f"[MCP JSON ERROR] Response length: {len(response_str)}")
                    print(f"[MCP JSON ERROR] First 100 chars: {response_str[:100]}")
                    print(f"[MCP JSON ERROR] Last 100 chars: {response_str[-100:]}")
                    
                    # Check if it's a truncated response
                    if response_str.strip() and not response_str.strip().endswith("}"):
                        print(f"[MCP JSON ERROR] Response appears truncated (doesn't end with }})")
                        # Try to read more data
                        try:
                            additional = self.process.stdout.read(1000)  # Read up to 1000 more bytes
                            if additional:
                                response_str += additional
                                response = json.loads(response_str.strip())
                                print(f"[MCP JSON ERROR] Successfully recovered by reading more data")
                            else:
                                raise Exception(f"MCP returned truncated JSON: {response_str[:200]}")
                        except:
                            raise Exception(f"MCP returned invalid JSON: {response_str[:200]}")
                    else:
                        # Check if it might be base64 or other encoding
                        if all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in response_str.strip()[:50]):
                            print(f"[MCP JSON ERROR] Response looks like base64 encoding")
                        
                        # Try to extract JSON from within the response
                        # Sometimes the response has JSON embedded in text
                        import re
                        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                        if json_match:
                            try:
                                response = json.loads(json_match.group())
                                print(f"[MCP JSON ERROR] Successfully extracted JSON from response")
                            except:
                                raise Exception(f"MCP returned invalid JSON: {response_str[:200]}")
                        else:
                            raise Exception(f"MCP returned invalid JSON: {response_str[:200]}")

                if "error" in response:
                    raise Exception(f"MCP Error: {response['error']}")
                
                return response.get("result", {})
                
            except asyncio.TimeoutError:
                print(f"[MCP TIMEOUT] Request '{method}' timed out after {REQUEST_TIMEOUT}s")
                # Do not stop the server, just raise the exception
                raise Exception(f"MCP request timeout: {method}")
            except Exception as e:
                print(f"[MCP ERROR] Request failed: {str(e)[:100]}")
                raise
    
    async def initialize(self):
        """Initialize the MCP connection."""
        result = await self._send_request("initialize", {
            "protocolVersion": "0.1.0",
            "capabilities": {},
            "clientInfo": {
                "name": "SDR Agent",
                "version": "1.0.0"
            }
        })
        self.initialized = True
        return result
    
    async def discover_tools(self):
        """Discover available tools from the MCP server."""
        try:
            result = await self._send_request("tools/list")
            self.tools = result.get("tools", [])
            print(f"[INFO] Discovered {len(self.tools)} MCP tools")
            return self.tools
        except Exception as e:
            print(f"⚠️ Error discovering tools: {e}")
            return []
    
    @traceable(name="brightdata_mcp_call_tool")
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific MCP tool."""
        result = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return result
    
    @traceable(name="brightdata_mcp_search")
    async def search(self, query: str, engine: str = "google", max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using the search_engine tool."""
        try:
            result = await self.call_tool(
                "search_engine",
                {
                    "query": query,
                    "engine": engine,
                    "max_results": max_results
                }
            )
            
            print(f"🔍 Raw search result: {result}")
            
            # Parse the result
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    results = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            # Parse the text content for search results
                            text = item["text"]
                            # Simple parsing - in production, use proper parsing
                            results.append({
                                "title": f"Result for: {query}",
                                "content": text,
                                "source": "brightdata_mcp"
                            })
                    return results
            
            return [{
                "title": "Search Results",
                "content": str(result),
                "source": "brightdata_mcp"
            }]
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    @traceable(name="brightdata_mcp_scrape")
    async def scrape(self, url: str, format: str = "markdown") -> Dict[str, Any]:
        """Scrape a webpage using MCP tools."""
        try:
            tool_name = "scrape_as_markdown" if format == "markdown" else "scrape_as_html"
            result = await self.call_tool(tool_name, {"url": url})
            
            # Extract content
            content = ""
            if isinstance(result, dict) and "content" in result:
                content_blocks = result["content"]
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if isinstance(block, dict) and "text" in block:
                            content += block["text"]
                elif isinstance(content_blocks, str):
                    content = content_blocks
            
            return {
                "url": url,
                "content": content,
                "format": format,
                "source": "brightdata_mcp"
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "source": "brightdata_mcp"
            }


# Create LangChain tools
def create_brightdata_mcp_tools():
    """Create LangChain tools using the MCP subprocess client."""
    from langchain.tools import Tool
    
    # Initialize client
    client = MCPSubprocessClient()
    
    async def search_wrapper(query: str) -> str:
        """Wrapper for search tool."""
        async with client:
            results = await client.search(query)
            return json.dumps(results, indent=2)
    
    async def scrape_wrapper(url: str) -> str:
        """Wrapper for scrape tool."""
        async with client:
            result = await client.scrape(url)
            return json.dumps(result, indent=2)
    
    return [
        Tool(
            name="brightdata_web_search",
            description="Search the web using Brightdata MCP. Input: search query",
            func=lambda q: asyncio.run(search_wrapper(q))
        ),
        Tool(
            name="brightdata_scrape_page",
            description="Scrape a webpage using Brightdata MCP. Input: URL",
            func=lambda url: asyncio.run(scrape_wrapper(url))
        )
    ] 