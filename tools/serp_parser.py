"""SERP HTML Parser - Extracts real data from Google search results using LLM."""

import json
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from core.prompt_manager import prompt_manager

class SERPParser:
    """Parse Google SERP HTML to extract actual search results using LLM-based extraction."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def parse_search_results(self, html_content: str) -> List[Dict[str, Any]]:
        """Extract search results from Google SERP HTML."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Method 1: Look for search result divs
            # Google uses various classes like g, rc, etc.
            for result_div in soup.find_all('div', class_=['g', 'rc', 'Gx5Zad']):
                result = self._extract_result_from_div(result_div)
                if result and result.get('title'):
                    results.append(result)
            
            # Method 2: If no results, try to extract from any <a> tags with titles
            if not results:
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.startswith('http') and 'google' not in href:
                        title = link.get_text(strip=True)
                        if len(title) > 20:  # Meaningful title
                            # Look for snippet near this link
                            parent = link.parent
                            snippet = ''
                            if parent:
                                snippet_elem = parent.find(['span', 'div'], class_=['st', 'aCOpRe', 'lEBKkf'])
                                if snippet_elem:
                                    snippet = snippet_elem.get_text(strip=True)
                            
                            results.append({
                                'title': title,
                                'url': href,
                                'snippet': snippet or self._extract_nearby_text(link, soup)
                            })
            
            # Return clean results
            return results
            
        except Exception as e:
            print(f"[ERROR SERP] Failed to parse HTML: {str(e)[:100]}")
        
        return results[:10]  # Limit to top 10 results
    
    def _extract_result_from_div(self, div) -> Optional[Dict[str, Any]]:
        """Extract a single search result from a div."""
        try:
            # Find the title/link
            link = div.find('a')
            if not link:
                return None
            
            title = link.get_text(strip=True)
            url = link.get('href', '')
            
            # Find snippet
            snippet = ''
            snippet_elem = div.find(['span', 'div'], class_=['st', 'aCOpRe', 'lEBKkf'])
            if snippet_elem:
                snippet = snippet_elem.get_text(strip=True)
            
            if title and url:
                return {
                    'title': title,
                    'url': url,
                    'snippet': snippet
                }
        except:
            pass
        
        return None
    
    def _extract_nearby_text(self, element, soup, max_distance=3) -> str:
        """Extract text near an element."""
        text_parts = []
        
        # Look at siblings
        for sibling in element.find_next_siblings()[:max_distance]:
            text = sibling.get_text(strip=True)
            if text and len(text) > 20:
                text_parts.append(text)
        
        return ' '.join(text_parts)[:200]
    
    async def extract_person_info(self, html_content: str, company: str, role: str) -> List[Dict[str, Any]]:
        """Extract person information from search results using LLM."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()[:5000]  # Limit text length
            
            # Use LLM to extract person information
            prompt = prompt_manager.get_prompt("serp_person_extractor")
            if not prompt:
                print("[ERROR] Person extractor prompt not found")
                return []
            
            formatted_prompt = prompt.format_messages(
                company=company,
                role=role,
                search_text=text
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
            
            people = json.loads(response_text.strip())
            
            # Validate and clean results
            validated_people = []
            for person in people:
                if isinstance(person, dict) and person.get('name'):
                    validated_people.append({
                        'name': person.get('name'),
                        'role': person.get('role', role),
                        'company': person.get('company', company),
                        'linkedin_url': person.get('linkedin_url')
                    })
            
            return validated_people
            
        except Exception as e:
            print(f"[ERROR] Failed to extract person info: {str(e)[:100]}")
            return []
    
    async def extract_tech_stack(self, html_content: str, company: str) -> List[str]:
        """Extract technology mentions from search results using LLM."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()[:5000]  # Limit text length
            
            # Use LLM to extract tech stack
            prompt = prompt_manager.get_prompt("tech_stack_extractor")
            if not prompt:
                print("[ERROR] Tech stack extractor prompt not found")
                return []
            
            formatted_prompt = prompt.format_messages(
                company=company,
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
            
            # Return technologies if confidence is medium or high
            if result.get('confidence') in ['high', 'medium']:
                return result.get('technologies', [])
            
            return []
            
        except Exception as e:
            print(f"[ERROR] Failed to extract tech stack: {str(e)[:100]}")
            return [] 