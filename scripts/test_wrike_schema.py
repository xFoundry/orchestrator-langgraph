#!/usr/bin/env python3
"""
Check what schema the LLM sees for wrike_get_tasks.
"""

import asyncio
import json
import sys
import aiohttp


async def get_tool_schemas(token: str):
    """Get full schemas for Wrike tools."""

    base_url = "https://www.wrike.com"
    headers = {"Authorization": f"Bearer {token}"}

    async with aiohttp.ClientSession() as session:
        # Connect to SSE
        sse_response = await session.get(
            f"{base_url}/app/mcp/sse",
            headers={**headers, "Accept": "text/event-stream"},
        )

        messages_url = None
        request_id = 0

        async def send_request(method, params=None):
            nonlocal request_id
            request_id += 1
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params:
                payload["params"] = params

            async with session.post(
                messages_url,
                headers={**headers, "Content-Type": "application/json"},
                json=payload
            ):
                pass

            async for line in sse_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data:"):
                    try:
                        parsed = json.loads(line[5:].strip())
                        if parsed.get("id") == request_id:
                            return parsed
                    except:
                        pass
            return None

        # Get session endpoint
        async for line in sse_response.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data:") and "/messages?sessionId=" in line:
                messages_url = f"{base_url}{line[5:].strip()}"
                break

        # Initialize
        await send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        })

        # Get tools
        tools_response = await send_request("tools/list", {})

        if tools_response and "result" in tools_response:
            tools = tools_response["result"].get("tools", [])

            # Find specific tools
            target_tools = ["wrike_get_tasks", "wrike_convert_numeric_id", "wrike_search_tasks"]

            for tool in tools:
                if tool.get("name") in target_tools:
                    print(f"\n{'='*70}")
                    print(f"TOOL: {tool.get('name')}")
                    print(f"{'='*70}")
                    print(f"\nDescription: {tool.get('description', 'N/A')}")

                    schema = tool.get("inputSchema", {})
                    print(f"\nInput Schema:")
                    print(json.dumps(schema, indent=2))

                    # Check for required fields
                    required = schema.get("required", [])
                    print(f"\nRequired fields: {required if required else 'NONE SPECIFIED'}")

                    # List all properties
                    props = schema.get("properties", {})
                    print(f"\nProperties ({len(props)}):")
                    for prop_name, prop_def in props.items():
                        prop_type = prop_def.get("type", "unknown")
                        desc = prop_def.get("description", "")[:60]
                        print(f"  - {prop_name} ({prop_type}): {desc}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_wrike_schema.py <token>")
        sys.exit(1)
    asyncio.run(get_tool_schemas(sys.argv[1]))
