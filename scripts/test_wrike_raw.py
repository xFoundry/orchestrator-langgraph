#!/usr/bin/env python3
"""
Raw test of Wrike MCP SSE endpoint - no LangChain/LangGraph.
Tests search_tasks filters and compares with sanitization.
"""

import asyncio
import json
import sys
import re
import aiohttp


# Copy of our sanitization logic to compare
CONTROL_CHAR_REGEX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

def sanitize_for_json(value):
    """Our sanitization function - copy from app/tools/sanitize.py"""
    if value is None:
        return None
    if isinstance(value, str):
        return CONTROL_CHAR_REGEX.sub('', value)
    if isinstance(value, (dict, list)):
        try:
            json_str = json.dumps(value, default=str)
            clean_str = CONTROL_CHAR_REGEX.sub('', json_str)
            return json.loads(clean_str)
        except Exception as e:
            print(f"Sanitization fallback: {e}")
            if isinstance(value, dict):
                return {k: sanitize_for_json(v) for k, v in value.items()}
            return [sanitize_for_json(item) for item in value]
    return value


async def test_wrike_raw(token: str):
    """Test Wrike MCP with proper bidirectional SSE protocol."""

    base_url = "https://www.wrike.com"
    headers = {
        "Authorization": f"Bearer {token}",
    }

    print("=" * 70)
    print("WRIKE MCP RAW TEST - Search Filters & Sanitization Check")
    print("=" * 70)

    async with aiohttp.ClientSession() as session:

        # Connect to SSE
        sse_response = await session.get(
            f"{base_url}/app/mcp/sse",
            headers={**headers, "Accept": "text/event-stream"},
        )

        messages_url = None
        request_id = 0

        async def send_request(method: str, params: dict = None):
            """Send request and get response from SSE."""
            nonlocal request_id
            request_id += 1

            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params:
                payload["params"] = params

            async with session.post(
                messages_url,
                headers={**headers, "Content-Type": "application/json"},
                json=payload
            ) as resp:
                pass

            # Read response from SSE
            async for line in sse_response.content:
                line = line.decode('utf-8').strip()
                if not line or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                    try:
                        parsed = json.loads(data)
                        if parsed.get("id") == request_id:
                            return parsed
                        elif "error" in parsed and not parsed.get("id"):
                            continue  # Skip unrelated errors
                    except:
                        pass
            return None

        # Get session endpoint
        async for line in sse_response.content:
            line = line.decode('utf-8').strip()
            if line.startswith("data:") and "/messages?sessionId=" in line:
                endpoint = line[5:].strip()
                messages_url = f"{base_url}{endpoint}"
                print(f"Connected: {messages_url}\n")
                break

        # Initialize
        await send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        })

        # ============================================================
        # TEST 1: Get wrike_search_tasks schema
        # ============================================================
        print("=" * 70)
        print("TEST 1: wrike_search_tasks INPUT SCHEMA")
        print("=" * 70)

        tools_response = await send_request("tools/list", {})

        search_schema = None
        if tools_response and "result" in tools_response:
            for tool in tools_response["result"].get("tools", []):
                if tool.get("name") == "wrike_search_tasks":
                    search_schema = tool.get("inputSchema", {})
                    print(f"\nwrike_search_tasks inputSchema:")
                    print(json.dumps(search_schema, indent=2))

                    # List all properties
                    props = search_schema.get("properties", {})
                    print(f"\n>>> Available filter parameters ({len(props)}):")
                    for prop_name, prop_def in props.items():
                        desc = prop_def.get("description", "")[:80]
                        print(f"  - {prop_name}: {desc}...")
                    break

        # ============================================================
        # TEST 2: Get current user's contact ID
        # ============================================================
        print("\n" + "=" * 70)
        print("TEST 2: Get current user's contact ID")
        print("=" * 70)

        contact_response = await send_request("tools/call", {
            "name": "wrike_get_my_contact_id",
            "arguments": {}
        })

        my_contact_id = None
        if contact_response and "result" in contact_response:
            content = contact_response["result"].get("content", [])
            for item in content:
                if item.get("type") == "text":
                    my_contact_id = item.get("text", "").strip()
                    print(f"\nMy contact ID: {my_contact_id}")
                    break

        # ============================================================
        # TEST 3: Search tasks with responsibles filter
        # ============================================================
        print("\n" + "=" * 70)
        print("TEST 3: Search tasks filtered by RESPONSIBLE (assignee)")
        print("=" * 70)

        if my_contact_id:
            # Try different filter parameter names
            filter_attempts = [
                {"responsibles": [my_contact_id], "limit": 5},
                {"responsible": my_contact_id, "limit": 5},
                {"assignee": my_contact_id, "limit": 5},
                {"responsibleIds": [my_contact_id], "limit": 5},
            ]

            for filters in filter_attempts:
                print(f"\n>>> Trying filter: {filters}")

                search_response = await send_request("tools/call", {
                    "name": "wrike_search_tasks",
                    "arguments": filters
                })

                if search_response:
                    if "error" in search_response:
                        print(f"  Error: {search_response['error'].get('message', search_response['error'])}")
                    elif "result" in search_response:
                        result = search_response["result"]
                        if result.get("isError"):
                            content = result.get("content", [])
                            for item in content:
                                if item.get("type") == "text":
                                    print(f"  Tool error: {item.get('text', '')[:200]}")
                        else:
                            content = result.get("content", [])
                            for item in content:
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    try:
                                        tasks = json.loads(text)
                                        print(f"  SUCCESS! Found {len(tasks)} tasks")
                                        if tasks:
                                            print(f"  First task: {tasks[0].get('title', 'N/A')}")
                                        break
                                    except:
                                        print(f"  Response: {text[:200]}")
        else:
            print("Skipped - no contact ID")

        # ============================================================
        # TEST 4: Raw response vs Sanitized response comparison
        # ============================================================
        print("\n" + "=" * 70)
        print("TEST 4: SANITIZATION COMPARISON")
        print("=" * 70)

        # Get a task with full details
        search_response = await send_request("tools/call", {
            "name": "wrike_search_tasks",
            "arguments": {"limit": 1}
        })

        task_id = None
        if search_response and "result" in search_response:
            for item in search_response["result"].get("content", []):
                if item.get("type") == "text":
                    try:
                        tasks = json.loads(item.get("text", "[]"))
                        if tasks:
                            task_id = tasks[0].get("id")
                    except:
                        pass

        if task_id:
            print(f"\nGetting full details for task: {task_id}")

            get_response = await send_request("tools/call", {
                "name": "wrike_get_tasks",
                "arguments": {"taskIds": [task_id]}
            })

            if get_response and "result" in get_response:
                for item in get_response["result"].get("content", []):
                    if item.get("type") == "text":
                        raw_text = item.get("text", "")

                        # Parse raw response
                        try:
                            raw_data = json.loads(raw_text)
                        except Exception as e:
                            print(f"Failed to parse raw JSON: {e}")
                            print(f"Raw text: {raw_text[:500]}")
                            continue

                        # Apply our sanitization
                        sanitized_data = sanitize_for_json(raw_data)

                        # Compare
                        print(f"\n>>> RAW data type: {type(raw_data)}")
                        if isinstance(raw_data, list) and raw_data:
                            raw_task = raw_data[0]
                            san_task = sanitized_data[0] if sanitized_data else {}

                            print(f"\nRAW task keys ({len(raw_task)}): {sorted(raw_task.keys())}")
                            print(f"SANITIZED task keys ({len(san_task)}): {sorted(san_task.keys())}")

                            # Check for missing keys
                            raw_keys = set(raw_task.keys())
                            san_keys = set(san_task.keys())

                            if raw_keys != san_keys:
                                missing = raw_keys - san_keys
                                extra = san_keys - raw_keys
                                if missing:
                                    print(f"\n!!! MISSING after sanitization: {missing}")
                                if extra:
                                    print(f"\n!!! EXTRA after sanitization: {extra}")
                            else:
                                print(f"\n✓ All keys preserved after sanitization")

                            # Check specific important fields
                            important_fields = ['id', 'title', 'responsibles', 'authors', 'status', 'description']
                            print(f"\nImportant field comparison:")
                            for field in important_fields:
                                raw_val = raw_task.get(field)
                                san_val = san_task.get(field)

                                raw_str = json.dumps(raw_val, default=str) if raw_val else "None"
                                san_str = json.dumps(san_val, default=str) if san_val else "None"

                                match = "✓" if raw_str == san_str else "✗ DIFFERENT"
                                print(f"  {field}: {match}")

                                if raw_str != san_str:
                                    print(f"    RAW: {raw_str[:100]}")
                                    print(f"    SAN: {san_str[:100]}")

                            # Check for control characters in raw data
                            raw_json = json.dumps(raw_data)
                            control_chars = CONTROL_CHAR_REGEX.findall(raw_json)
                            if control_chars:
                                print(f"\n>>> Found {len(control_chars)} control characters in raw data")
                                print(f"    Characters: {[hex(ord(c)) for c in control_chars[:10]]}")
                            else:
                                print(f"\n>>> No control characters found in raw data")

        # ============================================================
        # TEST 5: Check what fields search_tasks returns
        # ============================================================
        print("\n" + "=" * 70)
        print("TEST 5: Fields returned by search_tasks vs get_tasks")
        print("=" * 70)

        search_response = await send_request("tools/call", {
            "name": "wrike_search_tasks",
            "arguments": {"limit": 1}
        })

        search_fields = set()
        get_fields = set()

        if search_response and "result" in search_response:
            for item in search_response["result"].get("content", []):
                if item.get("type") == "text":
                    try:
                        tasks = json.loads(item.get("text", "[]"))
                        if tasks:
                            search_fields = set(tasks[0].keys())
                            task_id = tasks[0].get("id")
                    except:
                        pass

        if task_id:
            get_response = await send_request("tools/call", {
                "name": "wrike_get_tasks",
                "arguments": {"taskIds": [task_id]}
            })

            if get_response and "result" in get_response:
                for item in get_response["result"].get("content", []):
                    if item.get("type") == "text":
                        try:
                            tasks = json.loads(item.get("text", "[]"))
                            if tasks:
                                get_fields = set(tasks[0].keys())
                        except:
                            pass

        print(f"\nsearch_tasks returns {len(search_fields)} fields:")
        print(f"  {sorted(search_fields)}")

        print(f"\nget_tasks returns {len(get_fields)} fields:")
        print(f"  {sorted(get_fields)}")

        missing_in_search = get_fields - search_fields
        print(f"\nFields ONLY in get_tasks ({len(missing_in_search)}):")
        print(f"  {sorted(missing_in_search)}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_wrike_raw.py <wrike_token>")
        sys.exit(1)

    token = sys.argv[1]
    asyncio.run(test_wrike_raw(token))
