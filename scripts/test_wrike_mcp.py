#!/usr/bin/env python3
"""
Test script to directly call Wrike MCP and inspect responses.
Checks if task IDs are being returned correctly.
"""

import asyncio
import json
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient


async def test_wrike_mcp(token: str):
    """Test Wrike MCP connection and inspect responses."""

    # Build connection config matching our integration setup
    connection_config = {
        "transport": "sse",
        "url": "https://www.wrike.com/app/mcp/sse",
        "headers": {
            "Authorization": f"Bearer {token}"
        },
        "timeout": 30.0,
    }

    print("=" * 60)
    print("Connecting to Wrike MCP...")
    print("=" * 60)

    try:
        # Create client WITHOUT sanitization interceptor to see raw data
        client = MultiServerMCPClient(
            {"wrike": connection_config},
            tool_name_prefix=True,
        )

        # Get available tools
        tools = await client.get_tools()
        print(f"\nFound {len(tools)} tools")

        # Find specific tools we need
        search_tasks_tool = None
        get_my_contact_tool = None
        get_tasks_tool = None

        for tool in tools:
            if "search_tasks" in tool.name:
                search_tasks_tool = tool
                print(f"\nFound: {tool.name}")
                print(f"  Description: {tool.description[:100]}...")
            elif "get_my_contact" in tool.name:
                get_my_contact_tool = tool
                print(f"\nFound: {tool.name}")
                print(f"  Description: {tool.description[:100]}...")
            elif tool.name == "wrike_get_tasks":
                get_tasks_tool = tool
                print(f"\nFound: {tool.name}")
                print(f"  Description: {tool.description[:100]}...")

        # Test 1: Get my contact ID
        print("\n" + "=" * 60)
        print("TEST 1: Get my contact ID")
        print("=" * 60)

        if get_my_contact_tool:
            result = await get_my_contact_tool.ainvoke({})
            print(f"\nRaw result type: {type(result)}")
            print(f"Raw result: {result}")

            # Try to parse contact ID
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    print(f"\nParsed JSON: {json.dumps(data, indent=2)}")
                except:
                    print(f"\nNot JSON, raw string: {result}")

        # Test 2: Search tasks (no filters - get all accessible tasks)
        print("\n" + "=" * 60)
        print("TEST 2: Search tasks (limited)")
        print("=" * 60)

        if search_tasks_tool:
            # Get input schema to see what parameters are available
            try:
                schema = search_tasks_tool.get_input_schema()
                print(f"\nInput schema: {schema}")
            except Exception as e:
                print(f"\nCould not get schema: {e}")

            # Call with minimal params
            result = await search_tasks_tool.ainvoke({"limit": 5})
            print(f"\nRaw result type: {type(result)}")
            print(f"\nRaw result (first 2000 chars):")
            result_str = str(result)
            print(result_str[:2000])

            # Check for task IDs in the result
            if "id" in result_str.lower():
                print("\n[OK] Result contains 'id' field")
            else:
                print("\n[WARNING] Result may not contain 'id' field!")

            # Try to parse and find task IDs
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    print(f"\n\nParsed JSON structure:")
                    if isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())}")
                        if "data" in data and isinstance(data["data"], list):
                            print(f"  Number of tasks: {len(data['data'])}")
                            if data["data"]:
                                first_task = data["data"][0]
                                print(f"  First task keys: {list(first_task.keys())}")
                                if "id" in first_task:
                                    print(f"  First task ID: {first_task['id']}")
                                else:
                                    print("  [WARNING] No 'id' in first task!")
                except Exception as e:
                    print(f"\nFailed to parse as JSON: {e}")

        # Test 3: If we got task IDs, try get_tasks
        print("\n" + "=" * 60)
        print("TEST 3: Get tasks by ID (if we have IDs)")
        print("=" * 60)

        if get_tasks_tool and search_tasks_tool:
            # First get a task ID from search
            search_result = await search_tasks_tool.ainvoke({"limit": 1})
            task_id = None

            if isinstance(search_result, str):
                try:
                    data = json.loads(search_result)
                    if isinstance(data, dict) and "data" in data and data["data"]:
                        task_id = data["data"][0].get("id")
                except:
                    pass

            if task_id:
                print(f"\nUsing task ID: {task_id}")
                try:
                    result = await get_tasks_tool.ainvoke({"taskIds": [task_id]})
                    print(f"\nget_tasks result: {str(result)[:1000]}")
                except Exception as e:
                    print(f"\nget_tasks failed: {e}")
            else:
                print("\nCould not extract task ID from search results")

        print("\n" + "=" * 60)
        print("TESTS COMPLETE")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_wrike_mcp.py <wrike_token>")
        sys.exit(1)

    token = sys.argv[1]
    asyncio.run(test_wrike_mcp(token))
