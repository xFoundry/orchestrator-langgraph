#!/usr/bin/env python3
"""
Test sanitization logic to ensure no data is accidentally dropped.
"""

import json
import re

# Copy of the sanitization logic
CONTROL_CHAR_REGEX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_for_json(value):
    """Our sanitization function."""
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
            print(f"JSON round-trip failed: {e}")
            if isinstance(value, dict):
                return {k: sanitize_for_json(v) for k, v in value.items()}
            return [sanitize_for_json(item) for item in value]
    return value


def test_case(name, input_val, expected_keys=None, should_preserve=None):
    """Test a sanitization case."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    result = sanitize_for_json(input_val)

    if expected_keys:
        if isinstance(result, dict):
            result_keys = set(result.keys())
            expected_set = set(expected_keys)
            if result_keys == expected_set:
                print(f"✓ All {len(expected_keys)} keys preserved")
            else:
                missing = expected_set - result_keys
                extra = result_keys - expected_set
                if missing:
                    print(f"✗ MISSING keys: {missing}")
                if extra:
                    print(f"✗ EXTRA keys: {extra}")
        elif isinstance(result, list) and result:
            result_keys = set(result[0].keys()) if isinstance(result[0], dict) else set()
            expected_set = set(expected_keys)
            if result_keys == expected_set:
                print(f"✓ All {len(expected_keys)} keys preserved in list item")
            else:
                missing = expected_set - result_keys
                extra = result_keys - expected_set
                if missing:
                    print(f"✗ MISSING keys: {missing}")
                if extra:
                    print(f"✗ EXTRA keys: {extra}")

    if should_preserve:
        for key, expected_val in should_preserve.items():
            if isinstance(result, dict):
                actual = result.get(key)
            elif isinstance(result, list) and result:
                actual = result[0].get(key) if isinstance(result[0], dict) else None
            else:
                actual = None

            if actual == expected_val:
                print(f"✓ '{key}' preserved: {repr(actual)[:50]}")
            else:
                print(f"✗ '{key}' CHANGED!")
                print(f"  Expected: {repr(expected_val)[:100]}")
                print(f"  Got:      {repr(actual)[:100]}")

    return result


# ============================================================
# TEST CASES
# ============================================================

print("\n" + "="*70)
print("SANITIZATION EDGE CASE TESTS")
print("="*70)

# Test 1: Normal Wrike-like task data
test_case(
    "Normal task data (Wrike-like)",
    {
        "id": "MAAAAABpbNS6",
        "title": "Plan out video workflows",
        "status": "Active",
        "responsibles": [{"id": "KUAWERPW", "name": "Kevin Roach"}],
        "description": "Some description with <a href=\"url\">link</a>",
    },
    expected_keys=["id", "title", "status", "responsibles", "description"],
    should_preserve={
        "id": "MAAAAABpbNS6",
        "title": "Plan out video workflows",
    }
)

# Test 2: Data with control characters
test_case(
    "Data with control characters",
    {
        "id": "TEST123",
        "title": "Title with\x00null\x01and\x02control chars",
        "description": "Normal text\x0bwith vertical tab",
    },
    expected_keys=["id", "title", "description"],
    should_preserve={
        "id": "TEST123",
        "title": "Title withnullandcontrol chars",  # Control chars removed
        "description": "Normal textwith vertical tab",
    }
)

# Test 3: Nested structures
test_case(
    "Deeply nested data",
    {
        "task": {
            "subtasks": [
                {"id": "SUB1", "assignee": {"name": "John", "email": "john@test.com"}},
                {"id": "SUB2", "assignee": {"name": "Jane", "email": "jane@test.com"}},
            ]
        }
    },
    expected_keys=["task"],
    should_preserve={}
)

# Test 4: Unicode characters (should be preserved)
test_case(
    "Unicode characters (should preserve)",
    {
        "title": "Task with émojis 🎉 and ñ characters",
        "description": "日本語テスト",
        "author": "José García",
    },
    expected_keys=["title", "description", "author"],
    should_preserve={
        "title": "Task with émojis 🎉 and ñ characters",
        "description": "日本語テスト",
        "author": "José García",
    }
)

# Test 5: Empty values
test_case(
    "Empty values",
    {
        "id": "TEST",
        "title": "",
        "items": [],
        "metadata": {},
        "nullable": None,
    },
    expected_keys=["id", "title", "items", "metadata", "nullable"],
    should_preserve={
        "title": "",
        "items": [],
        "metadata": {},
    }
)

# Test 6: Special JSON characters in values
test_case(
    "Special JSON characters",
    {
        "id": "TEST",
        "title": 'String with "quotes" and \\backslashes',
        "path": "C:\\Users\\test\\file.txt",
        "json_like": '{"nested": "json"}',
    },
    expected_keys=["id", "title", "path", "json_like"],
    should_preserve={
        "title": 'String with "quotes" and \\backslashes',
        "path": "C:\\Users\\test\\file.txt",
        "json_like": '{"nested": "json"}',
    }
)

# Test 7: Numbers and booleans
test_case(
    "Numbers and booleans",
    {
        "count": 42,
        "price": 19.99,
        "active": True,
        "deleted": False,
        "big_num": 9999999999999,
        "negative": -100,
    },
    expected_keys=["count", "price", "active", "deleted", "big_num", "negative"],
    should_preserve={
        "count": 42,
        "price": 19.99,
        "active": True,
        "deleted": False,
    }
)

# Test 8: List of tasks (like Wrike search response)
test_case(
    "List of tasks (search response)",
    [
        {"id": "TASK1", "title": "First task", "status": "Active"},
        {"id": "TASK2", "title": "Second task", "status": "Completed"},
    ],
    expected_keys=["id", "title", "status"],
    should_preserve={}
)

# Test 9: MCP response structure
mcp_response_text = '[{"id":"MAAAAABpbNS6","title":"Test","responsibles":[{"id":"KUAWERPW","name":"Kevin"}]}]'
result = sanitize_for_json(mcp_response_text)
print(f"\n{'='*60}")
print("TEST: MCP response (JSON string)")
print("="*60)
if result == mcp_response_text:
    print(f"✓ JSON string preserved exactly")
else:
    print(f"✗ JSON string CHANGED!")
    print(f"  Original: {mcp_response_text[:100]}")
    print(f"  Result:   {result[:100]}")

# Test 10: Verify no data loss in round-trip
print(f"\n{'='*60}")
print("TEST: Full round-trip verification")
print("="*60)

original = {
    "metadata": [],
    "importance": "Normal",
    "customFields": [],
    "parentIds": ["IEAGVDJUI7777777"],
    "description": '[<a href="https://example.com">link</a>]<br />',
    "updatedDate": "2025-10-07T16:41:52Z",
    "title": "Plan out video workflows",
    "followedByMe": False,
    "scope": "WsTask",
    "id": "MAAAAABpbNS6",
    "hasAttachments": False,
    "subTaskIds": [],
    "shareds": [{"id": "KUAWEFU3", "name": "Isabella Laurel"}],
    "dates": {"type": "Planned", "duration": 1440, "start": "2025-10-07T09:00:00"},
    "superTaskIds": ["MAAAAABpbNRM"],
    "priority": "01d920007fffffffffff8c00",
    "superParentIds": [],
    "accountId": "IEAGVDJU",
    "dependencyIds": [],
    "createdDate": "2025-10-07T16:21:48Z",
    "followers": [{"id": "KUAWEFU3", "name": "Isabella Laurel"}],
    "customStatusId": "IEAGVDJUJMGIH7CG",
    "responsibles": [{"id": "KUAWERPW", "name": "Kevin Roach"}],
    "permalink": "https://www.wrike.com/open.htm?id=1768740026",
    "briefDescription": "[figjam]",
    "status": "Active",
    "authors": [{"id": "KUAWEFU3", "name": "Isabella Laurel"}],
}

sanitized = sanitize_for_json(original)

# Check all keys
orig_keys = set(original.keys())
san_keys = set(sanitized.keys())

if orig_keys == san_keys:
    print(f"✓ All {len(orig_keys)} keys preserved")
else:
    print(f"✗ Keys mismatch!")
    print(f"  Missing: {orig_keys - san_keys}")
    print(f"  Extra: {san_keys - orig_keys}")

# Check all values
all_match = True
for key in original:
    orig_json = json.dumps(original[key], sort_keys=True)
    san_json = json.dumps(sanitized[key], sort_keys=True)
    if orig_json != san_json:
        all_match = False
        print(f"✗ Value mismatch for '{key}'")
        print(f"  Original: {orig_json[:100]}")
        print(f"  Sanitized: {san_json[:100]}")

if all_match:
    print(f"✓ All values preserved exactly")

print("\n" + "="*70)
print("TESTS COMPLETE")
print("="*70)
