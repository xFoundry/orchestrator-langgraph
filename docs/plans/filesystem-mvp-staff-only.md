# Filesystem Persistence MVP - Staff Only

## Overview

Simplified filesystem persistence for internal xFoundry staff users. No teams, no complex RBAC - just basic persistence and sharing at the tenant level.

## Simplified Architecture

### Namespace Hierarchy (No Teams)

```
tenant_id/
├── shared/                    # All staff can read/write (tenant-wide)
├── users/
│   └── {user_id}/
│       ├── saved/             # User's saved artifacts (cross-thread)
│       ├── memories/          # User's memories (cross-thread)
│       └── threads/
│           └── {thread_id}/
│               ├── context/   # Thread working files
│               └── artifacts/ # Thread artifacts
```

### Path → Namespace Mapping

| Virtual Path | Scope | Namespace Tuple |
|--------------|-------|-----------------|
| `/context/*` | Thread | `(tenant, user, thread, "context")` |
| `/artifacts/*` | Thread | `(tenant, user, thread, "artifacts")` |
| `/artifacts/saved/*` | User | `(tenant, user, "saved")` |
| `/memories/*` | User | `(tenant, user, "memories")` |
| `/shared/*` | Tenant | `(tenant, "shared")` |

### Simplified Permissions

- **Staff role** = full read/write access to everything
- No granular permissions needed
- Simple check: `if "staff" in roles: allow`

---

## What We're Skipping (For Now)

| Feature | Why Skip |
|---------|----------|
| Teams hierarchy | Staff share at tenant level directly |
| Complex RBAC (8+ permissions) | Staff = full access |
| Team selector UI | No teams concept |
| Database for teams (NocoDB/Postgres) | No team lookups needed |
| Auth0 API with fine-grained permissions | Just check `staff` role |
| Multiple roles (guest/student/mentor/curator) | Just `staff` for now |

---

## Implementation Phases

### Phase 1: Core Persistence (Backend)

**Goal**: Thread and user scoped persistence works

#### 1.1 Create `SimpleScopedStoreBackend`

```python
# app/backends/simple_scoped_store_backend.py
from deepagents.backends.store import StoreBackend

class SimpleScopedStoreBackend(StoreBackend):
    """Store backend with simplified namespace scoping (no teams)."""

    SCOPE_THREAD = "thread"
    SCOPE_USER = "user"
    SCOPE_TENANT = "tenant"

    PATH_SCOPES = {
        "/context/": SCOPE_THREAD,
        "/artifacts/saved/": SCOPE_USER,  # Must be before /artifacts/
        "/artifacts/": SCOPE_THREAD,
        "/memories/": SCOPE_USER,
        "/shared/": SCOPE_TENANT,
    }

    def _get_scope_for_path(self, path: str) -> str:
        for prefix, scope in sorted(self.PATH_SCOPES.items(),
                                     key=lambda x: -len(x[0])):
            if path.startswith(prefix):
                return scope
        return self.SCOPE_THREAD

    def _get_namespace(self, path: str = None) -> tuple[str, ...]:
        cfg = get_config().get("configurable", {})

        tenant_id = cfg.get("tenant_id", "default")
        user_id = cfg.get("user_id", "anonymous")
        thread_id = cfg.get("thread_id")

        scope = self._get_scope_for_path(path or "/")

        if scope == self.SCOPE_TENANT:
            return (tenant_id, "shared")

        if scope == self.SCOPE_USER:
            return (tenant_id, "users", user_id)

        # Thread scope
        if not thread_id:
            raise ValueError("Thread scope requires thread_id")
        return (tenant_id, "users", user_id, thread_id)
```

#### 1.2 Update Backend Factory

```python
# app/graphs/orchestrator_deep_agent.py
def make_backend(runtime):
    from deepagents.backends import CompositeBackend, StateBackend
    from app.backends.simple_scoped_store_backend import SimpleScopedStoreBackend

    ephemeral = StateBackend(runtime)

    if not store:
        return ephemeral

    scoped = SimpleScopedStoreBackend(runtime)

    return CompositeBackend(
        default=ephemeral,
        routes={
            "/context/": scoped,
            "/artifacts/": scoped,
            "/artifacts/saved/": scoped,
            "/memories/": scoped,
            "/shared/": scoped,
        },
    )
```

#### 1.3 Remove `/context/` from Hidden Paths

```python
# app/streaming/deep_agent_event_mapper.py
INTERNAL_PATHS = {
    "/.cache/",
    "/.tmp/",
    # "/context/" removed - now visible in UI
}
```

---

### Phase 2: File Promotion Tools (Backend)

**Goal**: Staff can save and share files

#### 2.1 Create Promotion Tool

```python
# app/tools/file_promotion.py
from langchain_core.tools import tool

@tool
def promote_file(
    source_path: str,
    target_scope: str,  # "user" or "tenant"
    new_name: str = None,
) -> str:
    """
    Promote a file to a higher scope for persistence/sharing.

    Args:
        source_path: Path to the file (e.g., /artifacts/report.md)
        target_scope: "user" (save to my workspace) or "tenant" (share with all staff)
        new_name: Optional new filename

    Examples:
        - promote_file("/artifacts/report.md", "user") → saves to /artifacts/saved/report.md
        - promote_file("/artifacts/saved/template.md", "tenant") → shares to /shared/template.md
    """
    # Staff can do anything - no permission check needed for MVP
    pass

@tool
def list_saved_files() -> str:
    """List files in user's saved workspace (/artifacts/saved/)."""
    pass

@tool
def list_shared_files() -> str:
    """List files shared by all staff (/shared/)."""
    pass
```

---

### Phase 3: Frontend Updates

**Goal**: UI shows saved/shared resources and allows promotion

#### 3.1 File Tree Component (Right Sidebar)

Using `npx ai-elements@latest add file-tree`:

```tsx
// components/sidebars/context-panel.tsx
import { FileTree } from "@/components/ui/file-tree";

export function ContextPanel() {
  const { contextFiles } = useChatContext();

  return (
    <CollapsibleSection title="Context" icon={<FolderTree />}>
      <FileTree
        files={contextFiles}
        onFileClick={(file) => openPreview(file)}
      />
    </CollapsibleSection>
  );
}
```

#### 3.2 API Endpoint for File Listing

```typescript
// app/api/files/route.ts
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const scope = searchParams.get("scope"); // "thread", "user", "tenant"
  const threadId = searchParams.get("thread_id");

  // Call LangGraph backend to list files
  const response = await fetch(`${LANGGRAPH_URL}/files/list`, {
    method: "POST",
    body: JSON.stringify({ scope, thread_id: threadId, ... }),
  });

  return Response.json(await response.json());
}
```

#### 3.3 Artifact Actions

```tsx
// components/chat/artifact-card.tsx
<DropdownMenu>
  <DropdownMenuItem onClick={() => promoteFile(artifact.path, "user")}>
    <Save className="mr-2 h-4 w-4" />
    Save to workspace
  </DropdownMenuItem>
  <DropdownMenuItem onClick={() => promoteFile(artifact.path, "tenant")}>
    <Share className="mr-2 h-4 w-4" />
    Share with team
  </DropdownMenuItem>
</DropdownMenu>
```

---

## Infrastructure Requirements

### Current State (Already Implemented)

The orchestrator-langgraph **already has** the persistence infrastructure:

```python
# app/persistence/redis.py - get_store() function
# Priority order:
# 1. AsyncPostgresStore (if POSTGRES_URL set)
# 2. AsyncRedisStore (if REDIS_URL set)
# 3. InMemoryStore (fallback)
```

```python
# app/graphs/orchestrator_deep_agent.py - make_backend()
CompositeBackend(
    default=StateBackend(runtime),  # Ephemeral
    routes={
        "/memories/": StoreBackend(runtime),   # Persistent
        "/artifacts/": StoreBackend(runtime),  # Persistent
    },
)
```

### What's Missing

1. **Hierarchical namespacing** - Currently flat: `deepagent:memories:{user_id}`
2. **REST API for file access** - No endpoints to list/read files from UI
3. **`/context/` persistence** - Currently ephemeral (StateBackend)
4. **`/shared/` path** - Tenant-wide sharing not implemented

### Railway Services Needed

| Service | Status | Purpose |
|---------|--------|---------|
| **Postgres** | **Need to add** | PostgresStore for file persistence |
| **Redis** | Already have | Checkpointer (session state) |
| **FastAPI** | Already have | orchestrator-langgraph |

### Railway Environment Variables

```bash
# Required for file persistence
POSTGRES_URL=postgresql://user:password@host:5432/dbname

# Already configured
REDIS_URL=redis://...
```

The `get_store()` function in `app/persistence/redis.py` already checks for `POSTGRES_URL` and initializes `AsyncPostgresStore` with auto table creation via `await store.setup()`.

---

## LangGraph Store API (For UI File Access)

The store supports these operations **outside the agent context**:

```python
from app.persistence.redis import get_store

store = await get_store()

# List files in a namespace
items = await store.search(namespace_prefix=("tenant", "user", "saved"))
files = [{"key": item.key, "value": item.value} for item in items]

# Read a file
result = await store.get(namespace=("tenant", "user", "saved"), key="report.md")
content = result.value.get("content") if result else None

# Write a file
await store.put(
    namespace=("tenant", "user", "saved"),
    key="report.md",
    value={"content": "# Report\n...", "path": "/artifacts/saved/report.md"}
)

# List namespaces
namespaces = await store.list_namespaces(prefix=("tenant",))
```

---

## API Endpoints Needed (Backend)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/files/list` | GET | List files in a scope (thread/user/tenant) |
| `/api/files/read` | GET | Read file content by path |
| `/api/files/promote` | POST | Promote file to higher scope |
| `/api/files/tree` | GET | Get hierarchical file tree for UI |

### Example: List Files Endpoint

```python
# app/api/routes/files.py
from fastapi import APIRouter, Depends, Query
from app.persistence.redis import get_store

router = APIRouter(prefix="/api/files", tags=["files"])

@router.get("/list")
async def list_files(
    scope: str = Query("thread", enum=["thread", "user", "tenant"]),
    thread_id: str = Query(None),
    user_id: str = Query(...),
    tenant_id: str = Query("default"),
):
    """List files in the specified scope."""
    store = await get_store()

    # Build namespace based on scope
    if scope == "tenant":
        namespace = (tenant_id, "shared")
    elif scope == "user":
        namespace = (tenant_id, "users", user_id, "saved")
    else:  # thread
        namespace = (tenant_id, "users", user_id, thread_id, "artifacts")

    items = await store.search(namespace_prefix=namespace)
    return {
        "files": [
            {"key": item.key, "path": item.value.get("path"), "size": len(item.value.get("content", ""))}
            for item in items
        ]
    }
```

---

## Verification Checklist

- [ ] Create artifact in thread → shows in right sidebar
- [ ] Close browser, return → artifact still there
- [ ] "Save to workspace" → file appears in /artifacts/saved/
- [ ] Start new thread → can access saved file
- [ ] "Share with team" → file appears in /shared/
- [ ] Other staff user → can see shared file

---

## Future Enhancements (Post-MVP)

1. **Teams**: Add team hierarchy when needed
2. **Permissions**: Add granular RBAC for student/mentor roles
3. **S3 Storage**: For large binary files
4. **File versioning**: Track changes to promoted files
5. **Expiration**: Auto-cleanup old thread files
