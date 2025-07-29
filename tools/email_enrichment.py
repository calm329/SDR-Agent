"""Email enrichment module with multiple fallback strategies."""

import os
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

class EmailEnrichmentService:
    """Service for enriching contact information with email addresses using multiple providers."""
    
    def __init__(self):
        self.apollo_api_key = os.getenv("APOLLO_API_KEY", "")
        self.hunter_api_key = os.getenv("HUNTER_API_KEY", "")
        self.clearbit_api_key = os.getenv("CLEARBIT_API_KEY", "")
        
        # Rate limiting
        self.apollo_semaphore = asyncio.Semaphore(2)  # Apollo has strict rate limits
        self.hunter_semaphore = asyncio.Semaphore(3)
        
    async def enrich_contact(self, 
                           name: str, 
                           company: str, 
                           domain: str = None,
                           title: str = None,
                           linkedin_url: str = None) -> Dict[str, Any]:
        """
        Enrich contact with email using multiple strategies.
        Returns enriched contact data with confidence scores.
        """
        print(f"\n[EMAIL ENRICHMENT] Starting enrichment for {name} at {company}")
        
        results = {
            "name": name,
            "company": company,
            "title": title,
            "domain": domain,
            "email": None,
            "email_confidence": 0,
            "source": None,
            "alternate_emails": [],
            "verification_status": "unverified"
        }
        
        # Strategy 1: Try Apollo first (highest quality data)
        if self.apollo_api_key:
            apollo_result = await self._apollo_enrich(name, company, domain, title)
            if apollo_result and apollo_result.get("email"):
                results.update(apollo_result)
                results["source"] = "Apollo.io"
                print(f"[APOLLO] Found email: {results['email']} (confidence: {results['email_confidence']})")
                return results
        
        # Strategy 2: Try Hunter.io (good for domain-based search)
        if self.hunter_api_key and domain:
            hunter_result = await self._hunter_enrich(name, domain, company)
            if hunter_result and hunter_result.get("email"):
                results.update(hunter_result)
                results["source"] = "Hunter.io"
                print(f"[HUNTER] Found email: {results['email']} (confidence: {results['email_confidence']})")
                return results
        
        # Strategy 3: Try pattern-based email generation with verification
        if domain:
            pattern_result = await self._pattern_based_enrichment(name, domain, company)
            if pattern_result and pattern_result.get("email"):
                results.update(pattern_result)
                results["source"] = "Pattern-based"
                print(f"[PATTERN] Generated email: {results['email']} (confidence: {results['email_confidence']})")
                return results
        
        # Strategy 4: LinkedIn-based enrichment (if we have LinkedIn URL)
        if linkedin_url:
            linkedin_result = await self._linkedin_based_enrichment(linkedin_url, domain)
            if linkedin_result and linkedin_result.get("email"):
                results.update(linkedin_result)
                results["source"] = "LinkedIn-based"
                return results
        
        print(f"[EMAIL ENRICHMENT] No email found for {name}")
        return results
    
    async def _apollo_enrich(self, name: str, company: str, domain: str = None, title: str = None) -> Optional[Dict[str, Any]]:
        """Enrich using Apollo.io API."""
        try:
            async with self.apollo_semaphore:
                url = "https://api.apollo.io/v1/people/search"
                
                headers = {
                    "api_key": self.apollo_api_key,
                    "Content-Type": "application/json"
                }
                
                # Build search query
                params = {
                    "person_name": name,
                    "organization_name": company,
                    "page_size": 1
                }
                
                if title:
                    params["person_titles"] = [title]
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("people") and len(data["people"]) > 0:
                                person = data["people"][0]
                                return {
                                    "email": person.get("email"),
                                    "email_confidence": person.get("email_confidence", 0) / 100,  # Convert to 0-1
                                    "linkedin_url": person.get("linkedin_url"),
                                    "title": person.get("title"),
                                    "verified": person.get("email_status") == "verified"
                                }
        except Exception as e:
            print(f"[APOLLO ERROR] {str(e)[:100]}")
        return None
    
    async def _hunter_enrich(self, name: str, domain: str, company: str) -> Optional[Dict[str, Any]]:
        """Enrich using Hunter.io API."""
        try:
            async with self.hunter_semaphore:
                # First, try to find email directly
                url = "https://api.hunter.io/v2/email-finder"
                params = {
                    "domain": domain,
                    "full_name": name,
                    "api_key": self.hunter_api_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("data") and data["data"].get("email"):
                                return {
                                    "email": data["data"]["email"],
                                    "email_confidence": data["data"].get("confidence", 0) / 100,
                                    "sources": data["data"].get("sources", []),
                                    "pattern": data["data"].get("pattern")
                                }
                        
                        # If direct search fails, try domain search for pattern
                        elif response.status == 404:
                            domain_url = "https://api.hunter.io/v2/domain-search"
                            domain_params = {
                                "domain": domain,
                                "api_key": self.hunter_api_key,
                                "limit": 5
                            }
                            
                            async with session.get(domain_url, params=domain_params) as domain_response:
                                if domain_response.status == 200:
                                    domain_data = await domain_response.json()
                                    if domain_data.get("data", {}).get("pattern"):
                                        # Use the pattern to generate email
                                        pattern = domain_data["data"]["pattern"]
                                        email = self._apply_email_pattern(name, domain, pattern)
                                        return {
                                            "email": email,
                                            "email_confidence": 0.7,  # Medium confidence for pattern-based
                                            "pattern": pattern
                                        }
        except Exception as e:
            print(f"[HUNTER ERROR] {str(e)[:100]}")
        return None
    
    async def _pattern_based_enrichment(self, name: str, domain: str, company: str) -> Optional[Dict[str, Any]]:
        """Generate email using common patterns and verify if possible."""
        common_patterns = [
            "{first}.{last}",
            "{first}{last}",
            "{f}{last}",
            "{first}_{last}",
            "{last}.{first}",
            "{first}",
            "{f}.{last}"
        ]
        
        # Parse name
        name_parts = name.lower().split()
        if len(name_parts) < 2:
            return None
            
        first_name = name_parts[0]
        last_name = name_parts[-1]
        first_initial = first_name[0]
        
        generated_emails = []
        
        for pattern in common_patterns:
            email = pattern.format(
                first=first_name,
                last=last_name,
                f=first_initial
            ) + "@" + domain
            generated_emails.append(email)
        
        # For now, return the most common pattern with medium confidence
        # In production, you'd verify these emails
        return {
            "email": generated_emails[0],  # first.last@ is most common
            "email_confidence": 0.6,
            "alternate_emails": generated_emails[1:3],
            "pattern_used": common_patterns[0]
        }
    
    async def _linkedin_based_enrichment(self, linkedin_url: str, domain: str = None) -> Optional[Dict[str, Any]]:
        """Extract email hints from LinkedIn URL (would need LinkedIn API in production)."""
        # This is a placeholder - in production, you'd use LinkedIn API
        # or a service that can extract contact info from LinkedIn
        return None
    
    def _apply_email_pattern(self, name: str, domain: str, pattern: str) -> str:
        """Apply email pattern to generate email address."""
        name_parts = name.lower().split()
        if len(name_parts) < 2:
            return f"{name_parts[0]}@{domain}"
            
        first_name = name_parts[0]
        last_name = name_parts[-1]
        first_initial = first_name[0]
        last_initial = last_name[0]
        
        # Map Hunter.io patterns to email format
        pattern_map = {
            "{first}": first_name,
            "{last}": last_name,
            "{f}": first_initial,
            "{l}": last_initial,
            "{first}.{last}": f"{first_name}.{last_name}",
            "{first}{last}": f"{first_name}{last_name}",
            "{f}{last}": f"{first_initial}{last_name}",
            "{first}_{last}": f"{first_name}_{last_name}",
            "{f}.{last}": f"{first_initial}.{last_name}"
        }
        
        email_local = pattern
        for key, value in pattern_map.items():
            email_local = email_local.replace(key, value)
            
        return f"{email_local}@{domain}"
    
    async def bulk_enrich(self, contacts: List[Dict[str, Any]], domain: str = None) -> List[Dict[str, Any]]:
        """Enrich multiple contacts in parallel with rate limiting."""
        tasks = []
        for contact in contacts:
            task = self.enrich_contact(
                name=contact.get("name"),
                company=contact.get("company"),
                domain=domain or contact.get("domain"),
                title=contact.get("title"),
                linkedin_url=contact.get("linkedin_url")
            )
            tasks.append(task)
        
        # Process in batches to respect rate limits
        batch_size = 3
        enriched_contacts = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[ENRICHMENT ERROR] {str(result)[:100]}")
                    enriched_contacts.append(contacts[i + batch_results.index(result)])
                else:
                    enriched_contacts.append(result)
            
            # Small delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)
        
        return enriched_contacts 