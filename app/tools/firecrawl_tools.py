"""
Firecrawl Tools - Web scraping, search, map, crawl, and extract via Firecrawl API.

These tools call a self-hosted Firecrawl instance (v2 API).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)


def _build_firecrawl_url(path: str) -> str:
    settings = get_settings()
    base = (settings.firecrawl_api_url or "").rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


def _firecrawl_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.firecrawl_api_key:
        headers["Authorization"] = f"Bearer {settings.firecrawl_api_key}"
    return headers


async def _firecrawl_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = _build_firecrawl_url(path)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload, headers=_firecrawl_headers())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.error(f"Firecrawl request failed ({path}): {exc}")
        return {"success": False, "error": str(exc)}


class FirecrawlScrapeInput(BaseModel):
    """Input schema for Firecrawl scrape."""

    url: str = Field(..., description="URL to scrape")
    formats: list[str] = Field(
        default_factory=lambda: ["markdown"],
        description="Output formats (e.g. ['markdown', 'html'])",
    )
    only_main_content: bool = Field(
        default=True, description="Only return the main content of the page"
    )
    include_tags: Optional[list[str]] = Field(
        default=None, description="HTML tags to include"
    )
    exclude_tags: Optional[list[str]] = Field(
        default=None, description="HTML tags to exclude"
    )
    max_age: Optional[int] = Field(
        default=None, description="Cache max age in ms"
    )
    wait_for: Optional[int] = Field(
        default=None, description="Delay in ms before scraping"
    )
    mobile: Optional[bool] = Field(
        default=None, description="Emulate mobile device"
    )
    timeout: Optional[int] = Field(
        default=None, description="Timeout in ms (max 300000)"
    )
    actions: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Browser actions to perform before scraping"
    )
    location: Optional[dict[str, Any]] = Field(
        default=None, description="Location settings for proxy/locale"
    )
    remove_base64_images: Optional[bool] = Field(
        default=None, description="Remove base64 images from output"
    )
    block_ads: Optional[bool] = Field(
        default=None, description="Enable ad/cookie blocking"
    )
    proxy: Optional[str] = Field(
        default=None, description="Proxy mode: basic|stealth|auto"
    )
    store_in_cache: Optional[bool] = Field(
        default=None, description="Store results in cache"
    )
    zero_data_retention: Optional[bool] = Field(
        default=None, description="Enable zero data retention"
    )


@tool(args_schema=FirecrawlScrapeInput)
async def firecrawl_scrape(
    url: str,
    formats: list[str] = ["markdown"],
    only_main_content: bool = True,
    include_tags: Optional[list[str]] = None,
    exclude_tags: Optional[list[str]] = None,
    max_age: Optional[int] = None,
    wait_for: Optional[int] = None,
    mobile: Optional[bool] = None,
    timeout: Optional[int] = None,
    actions: Optional[list[dict[str, Any]]] = None,
    location: Optional[dict[str, Any]] = None,
    remove_base64_images: Optional[bool] = None,
    block_ads: Optional[bool] = None,
    proxy: Optional[str] = None,
    store_in_cache: Optional[bool] = None,
    zero_data_retention: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Scrape a single URL using Firecrawl (v2 API).

    Returns key fields (markdown/summary/metadata/links) to keep responses compact.
    """
    payload: dict[str, Any] = {
        "url": url,
        "formats": formats,
        "onlyMainContent": only_main_content,
    }
    if include_tags is not None:
        payload["includeTags"] = include_tags
    if exclude_tags is not None:
        payload["excludeTags"] = exclude_tags
    if max_age is not None:
        payload["maxAge"] = max_age
    if wait_for is not None:
        payload["waitFor"] = wait_for
    if mobile is not None:
        payload["mobile"] = mobile
    if timeout is not None:
        payload["timeout"] = timeout
    if actions is not None:
        payload["actions"] = actions
    if location is not None:
        payload["location"] = location
    if remove_base64_images is not None:
        payload["removeBase64Images"] = remove_base64_images
    if block_ads is not None:
        payload["blockAds"] = block_ads
    if proxy is not None:
        payload["proxy"] = proxy
    if store_in_cache is not None:
        payload["storeInCache"] = store_in_cache
    if zero_data_retention is not None:
        payload["zeroDataRetention"] = zero_data_retention

    result = await _firecrawl_post("/v2/scrape", payload)
    data = result.get("data") or {}

    return {
        "url": url,
        "markdown": data.get("markdown"),
        "summary": data.get("summary"),
        "metadata": data.get("metadata"),
        "links": data.get("links"),
        "warning": data.get("warning"),
        "success": result.get("success", True),
    }


class FirecrawlSearchInput(BaseModel):
    """Input schema for Firecrawl search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, description="Max results to return")
    lang: Optional[str] = Field(default=None, description="Language (e.g. en)")
    country: Optional[str] = Field(default=None, description="Country code (e.g. US)")
    tbs: Optional[str] = Field(default=None, description="Google TBS filter string")
    scrape_options: Optional[dict[str, Any]] = Field(
        default=None, description="Optional scrape options for search results"
    )


@tool(args_schema=FirecrawlSearchInput)
async def firecrawl_search(
    query: str,
    limit: int = 5,
    lang: Optional[str] = None,
    country: Optional[str] = None,
    tbs: Optional[str] = None,
    scrape_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Perform a web search via Firecrawl (v2 API).
    """
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
    }
    if lang is not None:
        payload["lang"] = lang
    if country is not None:
        payload["country"] = country
    if tbs is not None:
        payload["tbs"] = tbs
    if scrape_options is not None:
        payload["scrapeOptions"] = scrape_options

    result = await _firecrawl_post("/v2/search", payload)
    data = result.get("data") or result
    web = data.get("web") or data.get("results") or []

    results = []
    for item in web:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description") or item.get("snippet"),
            }
        )

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "success": result.get("success", True),
    }


class FirecrawlMapInput(BaseModel):
    """Input schema for Firecrawl map."""

    url: str = Field(..., description="Base URL to map")
    limit: Optional[int] = Field(default=None, description="Max URLs to return")
    include_subdomains: Optional[bool] = Field(
        default=None, description="Include subdomains"
    )
    ignore_query_parameters: Optional[bool] = Field(
        default=None, description="Ignore query parameters"
    )
    search: Optional[str] = Field(
        default=None, description="Filter URLs by search term"
    )
    sitemap: Optional[str] = Field(
        default=None, description="Sitemap handling: include|ignore"
    )


@tool(args_schema=FirecrawlMapInput)
async def firecrawl_map(
    url: str,
    limit: Optional[int] = None,
    include_subdomains: Optional[bool] = None,
    ignore_query_parameters: Optional[bool] = None,
    search: Optional[str] = None,
    sitemap: Optional[str] = None,
) -> dict[str, Any]:
    """
    Map a website and return discovered URLs.
    """
    payload: dict[str, Any] = {"url": url}
    if limit is not None:
        payload["limit"] = limit
    if include_subdomains is not None:
        payload["includeSubdomains"] = include_subdomains
    if ignore_query_parameters is not None:
        payload["ignoreQueryParameters"] = ignore_query_parameters
    if search is not None:
        payload["search"] = search
    if sitemap is not None:
        payload["sitemap"] = sitemap

    result = await _firecrawl_post("/v2/map", payload)
    data = result.get("data") or result
    urls = data.get("urls") or data.get("links") or []

    return {
        "url": url,
        "urls": urls,
        "count": len(urls) if isinstance(urls, list) else 0,
        "success": result.get("success", True),
    }


class FirecrawlCrawlInput(BaseModel):
    """Input schema for Firecrawl crawl."""

    url: str = Field(..., description="Base URL to crawl")
    limit: Optional[int] = Field(default=None, description="Max pages to crawl")
    max_discovery_depth: Optional[int] = Field(
        default=None, description="Max crawl depth"
    )
    allow_external_links: Optional[bool] = Field(
        default=None, description="Allow external links"
    )
    allow_subdomains: Optional[bool] = Field(
        default=None, description="Allow subdomains"
    )
    scrape_options: Optional[dict[str, Any]] = Field(
        default=None, description="Scrape options for each page"
    )


@tool(args_schema=FirecrawlCrawlInput)
async def firecrawl_crawl(
    url: str,
    limit: Optional[int] = None,
    max_discovery_depth: Optional[int] = None,
    allow_external_links: Optional[bool] = None,
    allow_subdomains: Optional[bool] = None,
    scrape_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Start a crawl job via Firecrawl (v2 API).
    """
    payload: dict[str, Any] = {"url": url}
    if limit is not None:
        payload["limit"] = limit
    if max_discovery_depth is not None:
        payload["maxDiscoveryDepth"] = max_discovery_depth
    if allow_external_links is not None:
        payload["allowExternalLinks"] = allow_external_links
    if allow_subdomains is not None:
        payload["allowSubdomains"] = allow_subdomains
    if scrape_options is not None:
        payload["scrapeOptions"] = scrape_options

    result = await _firecrawl_post("/v2/crawl", payload)
    data = result.get("data") or result

    return {
        "url": url,
        "job": data,
        "success": result.get("success", True),
    }


class FirecrawlExtractInput(BaseModel):
    """Input schema for Firecrawl extract."""

    urls: list[str] = Field(..., description="URLs to extract from")
    prompt: Optional[str] = Field(
        default=None, description="Extraction prompt"
    )
    schema: Optional[dict[str, Any]] = Field(
        default=None, description="JSON schema for extraction"
    )
    allow_external_links: Optional[bool] = Field(
        default=None, description="Allow external links"
    )
    enable_web_search: Optional[bool] = Field(
        default=None, description="Enable web search"
    )
    include_subdomains: Optional[bool] = Field(
        default=None, description="Include subdomains"
    )


@tool(args_schema=FirecrawlExtractInput)
async def firecrawl_extract(
    urls: list[str],
    prompt: Optional[str] = None,
    schema: Optional[dict[str, Any]] = None,
    allow_external_links: Optional[bool] = None,
    enable_web_search: Optional[bool] = None,
    include_subdomains: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Extract structured information from web pages via Firecrawl (v2 API).
    """
    payload: dict[str, Any] = {"urls": urls}
    if prompt is not None:
        payload["prompt"] = prompt
    if schema is not None:
        payload["schema"] = schema
    if allow_external_links is not None:
        payload["allowExternalLinks"] = allow_external_links
    if enable_web_search is not None:
        payload["enableWebSearch"] = enable_web_search
    if include_subdomains is not None:
        payload["includeSubdomains"] = include_subdomains

    result = await _firecrawl_post("/v2/extract", payload)
    data = result.get("data") or result

    return {
        "urls": urls,
        "results": data,
        "success": result.get("success", True),
    }
