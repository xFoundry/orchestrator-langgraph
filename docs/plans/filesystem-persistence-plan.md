# Filesystem Persistence & Access Control Plan

## Overview

This plan outlines the implementation of a hierarchical, role-based virtual filesystem for Deep Agents with proper persistence, access control, and file promotion/demotion capabilities.

### Key Decisions Summary

| Decision | Resolution |
|----------|------------|
| **Team Assignment** | Database lookup (NocoDB or Postgres - TBD) |
| **Multiple Teams** | Supported - user selects active team, one at a time |
| **User Roles** | Auth0 RBAC - roles as first-class objects with permissions |
| **Permissions** | Fine-grained Auth0 permissions tied to xFoundry API |
| **Namespace** | `tenant → team (optional) → user → thread` |
| **Persistence** | PostgresStore for Phase 1, S3 for large files later |
| **Access Model** | PolicyBackend checks Auth0 permissions from token |

---

## Current State

### Storage Architecture
- **Store**: `BaseStore` from LangGraph (Postgres in production, Redis/InMemory for dev)
- **Checkpointer**: Thread-scoped state persistence (conversation history)
- **Backend**: `CompositeBackend` routing paths to `StateBackend` (ephemeral) or `StoreBackend` (persistent)

### Current Config Structure (`get_thread_config`)
```python
{
    "configurable": {
        "thread_id": thread_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        # Missing: team_id, user_role
    }
}
```

### Current Namespace (StoreBackend)
```python
# Only uses assistant_id - flat namespace, no hierarchy
namespace = (assistant_id, "filesystem")  # or just ("filesystem",)
```

### Problems
1. No thread-scoped persistence for `/context/` and `/artifacts/`
2. No tenant → team → user → thread hierarchy
3. No role-based access control
4. No file promotion/demotion between scopes
5. `/context/` files hidden from UI and lost on session end

---

## Target Architecture

### Namespace Hierarchy

```
tenant_id/
├── shared/                    # Tenant-wide resources (super_admin can write)
│   └── templates/
├── teams/
│   └── {team_id}/
│       ├── shared/            # Team-shared resources (admin/curator can write)
│       └── members/
│           └── {user_id}/
│               ├── saved/     # User's saved artifacts (cross-thread)
│               ├── memories/  # User's memories (cross-thread)
│               └── threads/
│                   └── {thread_id}/
│                       ├── context/    # Tool results, working files
│                       └── artifacts/  # Conversation artifacts
└── users/                     # Users without a team
    └── {user_id}/
        └── (same structure as team members)
```

### Namespace Tuple Mapping

| Virtual Path | Scope | Namespace Tuple |
|--------------|-------|-----------------|
| `/context/*` | Thread | `(tenant, team?, user, thread, "context")` |
| `/artifacts/*` | Thread | `(tenant, team?, user, thread, "artifacts")` |
| `/artifacts/saved/*` | User | `(tenant, team?, user, "saved")` |
| `/memories/*` | User | `(tenant, team?, user, "memories")` |
| `/team/*` | Team | `(tenant, team, "shared")` |
| `/shared/*` | Tenant | `(tenant, "shared")` |

**Note**: `team?` is optional - users may not belong to a team.

---

## Role-Based Access Control (Auth0 RBAC)

### Auth0 RBAC Architecture

Auth0 RBAC uses a proper role-permission model:

```
Auth0 Dashboard
├── APIs
│   └── xFoundry Filesystem API
│       └── Permissions (defined here)
│           ├── read:thread
│           ├── write:thread
│           ├── read:user
│           ├── write:user
│           ├── read:team
│           ├── write:team
│           ├── read:tenant
│           ├── write:tenant
│           ├── promote:to_user
│           ├── promote:to_team
│           └── promote:to_tenant
│
└── Roles (assigned permissions from APIs)
    ├── guest        → [read:thread, read:tenant]
    ├── student      → [read:*, write:thread, write:user, promote:to_user]
    ├── mentor       → [read:*, write:thread, write:user, write:team, promote:to_team]
    ├── curator      → [read:*, write:thread, write:user, write:team, promote:to_team]
    ├── admin        → [read:*, write:*, promote:to_tenant]
    └── super_admin  → [*:*]
```

### Auth0 API Permissions (xFoundry Filesystem API)

| Permission | Description |
|------------|-------------|
| `read:thread` | Read files in current thread (`/context/`, `/artifacts/`) |
| `write:thread` | Write files in current thread |
| `read:user` | Read user-scoped files (`/artifacts/saved/`, `/memories/`) |
| `write:user` | Write to user-scoped paths |
| `read:team` | Read team-shared files (`/team/`) |
| `write:team` | Write to team-shared paths |
| `read:tenant` | Read tenant-shared files (`/shared/`) |
| `write:tenant` | Write to tenant-shared paths |
| `promote:to_user` | Promote thread files to user scope |
| `promote:to_team` | Promote files to team scope |
| `promote:to_tenant` | Promote files to tenant scope |

### Roles & Permissions Matrix

| Role | Permissions Assigned |
|------|---------------------|
| `guest` | `read:thread`, `read:tenant` |
| `student` | `read:thread`, `read:user`, `read:team`, `read:tenant`, `write:thread`, `write:user`, `promote:to_user` |
| `mentor` | All `read:*`, `write:thread`, `write:user`, `write:team`, `promote:to_user`, `promote:to_team` |
| `curator` | Same as mentor |
| `admin` | All `read:*`, all `write:*`, all `promote:*` |
| `super_admin` | All permissions (`*`) |

### Permission Scope Definitions

| Scope | Description | Required Permission to Write |
|-------|-------------|------------------------------|
| **Thread** | Current conversation only | `write:thread` |
| **User** | Cross-thread, single user | `write:user` |
| **Team** | All team members | `write:team` |
| **Tenant** | All users in tenant | `write:tenant` |

---

## Implementation Components

### 1. Scoped Store Backend (`app/backends/scoped_store_backend.py`)

```python
from deepagents.backends.store import StoreBackend
from langgraph.config import get_config

class ScopedStoreBackend(StoreBackend):
    """Store backend with hierarchical namespace scoping."""

    SCOPE_THREAD = "thread"
    SCOPE_USER = "user"
    SCOPE_TEAM = "team"
    SCOPE_TENANT = "tenant"

    # Path prefix → scope mapping
    PATH_SCOPES = {
        "/context/": SCOPE_THREAD,
        "/artifacts/saved/": SCOPE_USER,  # Must be before /artifacts/
        "/artifacts/": SCOPE_THREAD,
        "/memories/": SCOPE_USER,
        "/team/": SCOPE_TEAM,
        "/shared/": SCOPE_TENANT,
    }

    def _get_scope_for_path(self, path: str) -> str:
        """Determine scope level from path prefix."""
        for prefix, scope in sorted(self.PATH_SCOPES.items(),
                                     key=lambda x: -len(x[0])):  # Longest first
            if path.startswith(prefix):
                return scope
        return self.SCOPE_THREAD  # Default

    def _get_namespace(self, path: str = None) -> tuple[str, ...]:
        """Build namespace tuple based on scope and hierarchy."""
        cfg = get_config().get("configurable", {})

        tenant_id = cfg.get("tenant_id", "default")
        team_id = cfg.get("team_id")  # Optional
        user_id = cfg.get("user_id", "anonymous")
        thread_id = cfg.get("thread_id")

        scope = self._get_scope_for_path(path or "/")

        if scope == self.SCOPE_TENANT:
            return (tenant_id, "shared")

        if scope == self.SCOPE_TEAM:
            if not team_id:
                raise PermissionError("Team scope requires team_id")
            return (tenant_id, team_id, "shared")

        # Build user-level namespace (with optional team)
        if team_id:
            base = (tenant_id, team_id, user_id)
        else:
            base = (tenant_id, "users", user_id)

        if scope == self.SCOPE_USER:
            return base + ("user",)

        # Thread scope
        if not thread_id:
            raise ValueError("Thread scope requires thread_id")
        return base + (thread_id, "thread")
```

### 2. Policy Wrapper Backend (`app/backends/policy_backend.py`)

Uses Auth0 permissions from the access token instead of role-based lookups.

```python
from deepagents.backends.protocol import BackendProtocol, WriteResult, EditResult
from langgraph.config import get_config

class PolicyBackend(BackendProtocol):
    """Wrapper that enforces Auth0 RBAC permissions."""

    # Map path scopes to Auth0 permission names
    SCOPE_TO_PERMISSION = {
        "thread": "thread",
        "user": "user",
        "team": "team",
        "tenant": "tenant",
    }

    def __init__(self, inner: BackendProtocol, scoped_backend: ScopedStoreBackend):
        self.inner = inner
        self.scoped = scoped_backend

    def _get_user_permissions(self) -> set[str]:
        """Get permissions from config (passed from Auth0 token)."""
        cfg = get_config().get("configurable", {})
        # Permissions come as list from token, e.g., ["read:thread", "write:thread", ...]
        return set(cfg.get("user_permissions", []))

    def _has_permission(self, action: str, scope: str) -> bool:
        """Check if user has permission for action on scope."""
        permissions = self._get_user_permissions()

        # Check for wildcard permission
        if f"{action}:*" in permissions or "*:*" in permissions:
            return True

        # Check for specific permission
        permission_name = self.SCOPE_TO_PERMISSION.get(scope, scope)
        return f"{action}:{permission_name}" in permissions

    def _can_read(self, path: str) -> bool:
        scope = self.scoped._get_scope_for_path(path)
        return self._has_permission("read", scope)

    def _can_write(self, path: str) -> bool:
        scope = self.scoped._get_scope_for_path(path)
        return self._has_permission("write", scope)

    def _can_promote(self, target_scope: str) -> bool:
        return self._has_permission("promote", f"to_{target_scope}")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        if not self._can_read(file_path):
            return f"Error: Permission denied to read {file_path}"
        return self.inner.read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        if not self._can_write(file_path):
            return WriteResult(error=f"Permission denied to write {file_path}")
        return self.inner.write(file_path, content)

    def edit(self, file_path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        if not self._can_write(file_path):
            return EditResult(error=f"Permission denied to edit {file_path}")
        return self.inner.edit(file_path, old, new, replace_all)

    # Delegate read-only methods with filtering
    def ls_info(self, path: str):
        if not self._can_read(path):
            return []
        return self.inner.ls_info(path)

    def glob_info(self, pattern: str, path: str = "/"):
        return [f for f in self.inner.glob_info(pattern, path)
                if self._can_read(f.get("path", ""))]

    def grep_raw(self, pattern: str, path: str = None, glob: str = None):
        results = self.inner.grep_raw(pattern, path, glob)
        if isinstance(results, str):
            return results
        return [r for r in results if self._can_read(r.get("path", ""))]
```

### 3. File Promotion Tool (`app/tools/file_promotion.py`)

Uses Auth0 `promote:to_*` permissions for authorization.

```python
from langchain_core.tools import tool
from langgraph.config import get_config

def _has_promote_permission(target_scope: str) -> bool:
    """Check if user has permission to promote to target scope."""
    cfg = get_config().get("configurable", {})
    permissions = set(cfg.get("user_permissions", []))

    # Check for wildcard or specific promote permission
    if "promote:*" in permissions or "*:*" in permissions:
        return True
    return f"promote:to_{target_scope}" in permissions

@tool
def promote_file(
    source_path: str,
    target_scope: str,  # "user", "team", or "tenant"
    new_name: str = None,
) -> str:
    """
    Promote a file to a higher scope for sharing.

    Args:
        source_path: Path to the file to promote (e.g., /artifacts/report.md)
        target_scope: Target scope - "user" (saved), "team", or "tenant"
        new_name: Optional new filename

    Returns:
        Success message with new path, or error message

    Required permissions:
        - promote:to_user - promote to /artifacts/saved/
        - promote:to_team - promote to /team/
        - promote:to_tenant - promote to /shared/
    """
    if not _has_promote_permission(target_scope):
        return f"Error: You don't have permission to promote files to {target_scope} scope"

    # Implementation uses store.get() from source namespace
    # and store.put() to target namespace
    pass

@tool
def demote_file(
    source_path: str,
    target_scope: str,  # "thread", "user", or "team"
    target_thread_id: str = None,  # Required if demoting to thread
) -> str:
    """
    Copy a file to a lower scope (more restricted access).
    Requires write permission on target scope.

    Args:
        source_path: Path to the file to demote
        target_scope: Target scope - "thread", "user", or "team"
        target_thread_id: Thread ID if copying to thread scope

    Returns:
        Success message with new path, or error message
    """
    # Check write permission on target scope
    cfg = get_config().get("configurable", {})
    permissions = set(cfg.get("user_permissions", []))

    if f"write:{target_scope}" not in permissions and "write:*" not in permissions:
        return f"Error: You don't have permission to write to {target_scope} scope"

    pass
```

### 4. Updated Config Flow

```python
# app/persistence/redis.py
def get_thread_config(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
    team_id: str = None,              # Active team (from user selection)
    user_roles: list[str] = None,     # Auth0 roles (e.g., ["mentor", "curator"])
    user_permissions: list[str] = None,  # Auth0 permissions (e.g., ["read:team", "write:user"])
    recursion_limit: int = None,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "team_id": team_id,
            "user_roles": user_roles or [],
            "user_permissions": user_permissions or [],  # Used by PolicyBackend
        },
        "recursion_limit": recursion_limit or settings.recursion_limit,
    }
```

**Auth0 + Team Resolution Flow**:
```
1. User authenticates via Auth0
2. Auth0 returns token with:
   - Roles in custom claim (https://xfoundry.io/roles)
   - Permissions in scope claim (if RBAC enabled on API)
3. Frontend extracts roles and permissions from token
4. Frontend calls /api/user/teams to get team memberships (from database)
5. Frontend stores active_team_id in context
6. All chat requests include: team_id, roles, permissions
7. Backend passes to PolicyBackend for enforcement
```

### 5. Updated Backend Factory

```python
# app/graphs/orchestrator_deep_agent.py
def make_backend(runtime):
    from deepagents.backends import CompositeBackend, StateBackend
    from app.backends.scoped_store_backend import ScopedStoreBackend
    from app.backends.policy_backend import PolicyBackend

    ephemeral = StateBackend(runtime)

    if not store:
        return ephemeral

    # Create scoped backend for persistent paths
    scoped = ScopedStoreBackend(runtime)

    # Wrap with policy enforcement
    policy_scoped = PolicyBackend(scoped, scoped)

    return CompositeBackend(
        default=ephemeral,
        routes={
            # Thread-scoped (per conversation)
            "/context/": policy_scoped,
            "/artifacts/": policy_scoped,

            # User-scoped (cross-thread)
            "/artifacts/saved/": policy_scoped,
            "/memories/": policy_scoped,

            # Team-scoped
            "/team/": policy_scoped,

            # Tenant-scoped
            "/shared/": policy_scoped,
        },
    )
```

---

## API Changes

### ChatRequest Model Updates

```python
# app/models/chat.py
class UserContext(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    auth0_id: Optional[str] = None
    roles: list[str] = Field(
        default_factory=list,
        description="User roles from Auth0 RBAC (e.g., ['mentor', 'curator'])"
    )
    permissions: list[str] = Field(
        default_factory=list,
        description="User permissions from Auth0 RBAC (e.g., ['read:team', 'write:user'])"
    )

class ChatRequest(BaseModel):
    # ... existing fields ...

    team_id: Optional[str] = Field(
        default=None,
        description="Active team ID for team-scoped resources"
    )
    user_context: Optional[UserContext] = Field(
        default=None,
        description="User context including roles and permissions from Auth0 RBAC"
    )
```

### Chat Stream Route Updates

```python
# app/api/routes/chat_stream.py
config = get_thread_config(
    thread_id=thread_id,
    user_id=request.user_id or "anonymous",
    tenant_id=request.tenant_id,
    team_id=request.team_id,
    user_roles=request.user_context.roles if request.user_context else [],
    user_permissions=request.user_context.permissions if request.user_context else [],
)
```

### Frontend API: User Teams Endpoint

```typescript
// xfoundry-chat/src/app/api/user/teams/route.ts
import { getAuthSession } from "@/lib/auth0";
import { getUserTeams } from "@/lib/database";  // TBD: NocoDB or Postgres

export async function GET() {
  const session = await getAuthSession();
  if (!session?.user) {
    return new Response("Unauthorized", { status: 401 });
  }

  // Query database for user's team memberships
  const teams = await getUserTeams(session.user.sub);  // Use Auth0 ID

  // Get active team from session/cookie or default to first team
  const activeTeamId = getActiveTeamFromSession() || teams[0]?.id || null;

  return Response.json({
    teams,
    active_team_id: activeTeamId,
  });
}
```

```typescript
// xfoundry-chat/src/app/api/user/teams/active/route.ts
export async function POST(req: Request) {
  const session = await getAuthSession();
  if (!session?.user) {
    return new Response("Unauthorized", { status: 401 });
  }

  const { team_id } = await req.json();

  // Verify user belongs to this team
  const teams = await getUserTeams(session.user.sub);
  if (!teams.some(t => t.id === team_id)) {
    return new Response("Not a member of this team", { status: 403 });
  }

  // Store active team in session/cookie
  setActiveTeamInSession(team_id);

  return Response.json({ active_team_id: team_id });
}
```

### Database Query: Get User Teams (Placeholder)

**Database TBD**: NocoDB or separate Postgres database

```typescript
// xfoundry-chat/src/lib/database.ts
// TODO: Implement based on chosen database (NocoDB or Postgres)

interface TeamMembership {
  id: string;
  name: string;
}

export async function getUserTeams(auth0Id: string): Promise<TeamMembership[]> {
  // Option A: NocoDB REST API
  // const response = await fetch(`${NOCODB_URL}/api/v1/db/data/noco/${PROJECT}/team_members`, {
  //   headers: { "xc-token": NOCODB_TOKEN },
  // });

  // Option B: Direct Postgres
  // const result = await db.query(`
  //   SELECT t.id, t.name FROM teams t
  //   JOIN team_members tm ON t.id = tm.team_id
  //   JOIN users u ON tm.user_id = u.id
  //   WHERE u.auth0_id = $1
  // `, [auth0Id]);

  throw new Error("Database not configured - implement getUserTeams()");
}
```

---

## Frontend Changes

### Auth0 RBAC Token Extraction

```typescript
// xfoundry-chat/src/lib/auth0.ts
import { getAccessToken } from "@auth0/nextjs-auth0";
import { jwtDecode } from "jwt-decode";

interface Auth0AccessToken {
  sub: string;
  scope?: string;  // Permissions as space-separated string (when RBAC enabled)
  "https://xfoundry.io/roles"?: string[];  // Roles from custom Action
}

// Helper to extract roles from access token
export function getUserRoles(accessToken: string): string[] {
  const decoded = jwtDecode<Auth0AccessToken>(accessToken);
  return decoded["https://xfoundry.io/roles"] || [];
}

// Helper to extract permissions from access token
export function getUserPermissions(accessToken: string): string[] {
  const decoded = jwtDecode<Auth0AccessToken>(accessToken);
  // Permissions come from scope claim, filter to only include our API permissions
  const scope = decoded.scope || "";
  return scope.split(" ").filter(s => s.includes(":"));
}

// Server-side helper to get roles and permissions
export async function getAuthContext() {
  const { accessToken } = await getAccessToken();
  if (!accessToken) return { roles: [], permissions: [] };

  return {
    roles: getUserRoles(accessToken),
    permissions: getUserPermissions(accessToken),
  };
}
```

### User Teams Hook

```typescript
// xfoundry-chat/src/hooks/use-user-teams.ts
import useSWR from "swr";

interface TeamMembership {
  id: string;
  name: string;
}

interface UserTeamsData {
  teams: TeamMembership[];
  active_team_id: string | null;
}

export function useUserTeams() {
  const { data, error, mutate } = useSWR<UserTeamsData>("/api/user/teams");

  const setActiveTeam = async (teamId: string) => {
    await fetch("/api/user/teams/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_id: teamId }),
    });
    mutate();
  };

  return {
    teams: data?.teams || [],
    activeTeamId: data?.active_team_id,
    setActiveTeam,
    isLoading: !data && !error,
    error,
  };
}
```

### Auth Context Hook (for roles/permissions)

```typescript
// xfoundry-chat/src/hooks/use-auth-context.ts
import useSWR from "swr";

interface AuthContextData {
  roles: string[];
  permissions: string[];
}

export function useAuthContext() {
  // Fetch from API route that extracts from access token
  const { data, error } = useSWR<AuthContextData>("/api/auth/context");

  return {
    roles: data?.roles || [],
    permissions: data?.permissions || [],
    isLoading: !data && !error,
    error,
  };
}
```

### Chat Context Updates

```typescript
// xfoundry-chat/src/contexts/chat-context.tsx
import { useUserTeams } from "@/hooks/use-user-teams";
import { useAuthContext } from "@/hooks/use-auth-context";

// Inside ChatProvider
const { activeTeamId } = useUserTeams();
const { roles, permissions } = useAuthContext();

const chatRequest = {
    message: userText,
    tenant_id: tenantId,
    team_id: activeTeamId,
    user_id: session.user.sub,
    user_context: {
        name: session.user.name,
        email: session.user.email,
        auth0_id: session.user.sub,
        roles: roles,        // From Auth0 RBAC
        permissions: permissions,  // From Auth0 RBAC
    },
};
```

### Team Selector Component

```typescript
// xfoundry-chat/src/components/sidebars/team-selector.tsx
import { useUserTeams } from "@/hooks/use-user-teams";

export function TeamSelector() {
  const { teams, activeTeamId, setActiveTeam, isLoading } = useUserTeams();

  if (isLoading || teams.length <= 1) return null;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          {teams.find(t => t.id === activeTeamId)?.name || "Select Team"}
          <ChevronDown className="ml-2 h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        {teams.map(team => (
          <DropdownMenuItem
            key={team.id}
            onClick={() => setActiveTeam(team.id)}
          >
            {team.name}
            {team.id === activeTeamId && <Check className="ml-auto h-4 w-4" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
```

### Event Mapper Updates

```python
# app/streaming/deep_agent_event_mapper.py
# Remove /context/ from INTERNAL_PATHS so tool results are visible
INTERNAL_PATHS = {
    "/.cache/",
    "/.tmp/",
    # "/context/" removed - now shown in UI
}
```

---

## Storage Considerations

### Current: PostgresStore
- **Pros**: Already configured, SQL queries, ACID compliance
- **Cons**: Not ideal for large binary files, potential bottleneck

### Alternative: S3-Compatible Storage
For large artifacts (images, PDFs, exports), consider:

```python
class S3StoreBackend(BackendProtocol):
    """Store backend using S3 for file content, Postgres for metadata."""

    def __init__(self, s3_client, bucket: str, metadata_store: BaseStore):
        self.s3 = s3_client
        self.bucket = bucket
        self.metadata = metadata_store

    def write(self, file_path: str, content: str) -> WriteResult:
        # Store content in S3
        s3_key = self._path_to_key(file_path)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=content)

        # Store metadata in Postgres
        namespace = self._get_namespace(file_path)
        self.metadata.put(namespace, file_path, {
            "s3_key": s3_key,
            "size": len(content),
            "created_at": datetime.utcnow().isoformat(),
        })
        return WriteResult(path=file_path, files_update=None)
```

### Recommendation
1. **Phase 1**: Use PostgresStore with namespace hierarchy (simpler, faster to ship)
2. **Phase 2**: Add S3 backend for large files (if needed based on usage patterns)

---

## Implementation Phases

### Phase 1: Auth0 RBAC Setup (Priority: High)
1. **Create API in Auth0**: `xFoundry Filesystem API` with identifier
2. **Define Permissions**: `read:thread`, `write:thread`, `read:user`, etc.
3. **Create Roles**: `guest`, `student`, `mentor`, `curator`, `admin`, `super_admin`
4. **Assign Permissions to Roles**: Per the matrix defined above
5. **Enable RBAC on API**: Toggle "Enable RBAC" and "Add Permissions in Access Token"
6. **Create Post-Login Action**: Add roles to token custom claim
7. **Frontend Auth**: Create `useAuthContext` hook to extract roles/permissions from token
8. **Test**: Verify roles and permissions appear in tokens

### Phase 2: Team Infrastructure (Priority: High)
1. **Choose Database**: Finalize NocoDB vs Postgres for team data
2. **Create Schema**: `users`, `teams`, `team_members` tables
3. **Team API**: Create `/api/user/teams` endpoint
4. **Team Context**: Create `useUserTeams` hook
5. **Team Selector**: Add team selector component to sidebar
6. **Active Team Storage**: Store active team in session/cookie

### Phase 3: Core Persistence (Priority: High)
1. Create `ScopedStoreBackend` with namespace hierarchy
2. Update `get_thread_config()` with team_id, user_roles, user_permissions
3. Update `ChatRequest` model with team_id and user_context (roles/permissions)
4. Update chat stream route to pass new config
5. Update backend factory in orchestrator
6. Remove `/context/` from hidden paths in event mapper
7. Update frontend chat context to pass team_id, roles, permissions in requests

### Phase 4: Access Control (Priority: High)
1. Create `PolicyBackend` wrapper using Auth0 permissions
2. Implement permission checking for all operations
3. Add permission denied error handling in UI
4. Test permission boundaries for all role combinations

### Phase 5: File Promotion (Priority: Medium)
1. Create `promote_file` and `demote_file` tools
2. Add `promote:to_*` permission checks
3. Add to Deep Agent tool registry
4. Update system prompt with promotion instructions
5. Add "Save to workspace" UI action for artifacts

### Phase 6: Frontend Polish (Priority: Medium)
1. Show scope badges on artifacts in UI (thread/user/team/tenant)
2. Filter artifacts panel by scope
3. Show shared team/tenant resources in sidebar section
4. Add visual feedback for permission denied operations

### Phase 7: S3 Integration (Priority: Low - Deferred)
1. Create `S3StoreBackend`
2. Add S3 configuration to settings
3. Migrate large files to S3
4. Update download/preview endpoints

---

## Testing Strategy

### Unit Tests
- Namespace generation for all path/scope combinations
- Permission checks for all role/operation combinations
- File promotion/demotion validation

### Integration Tests
- End-to-end: Create artifact → Save → Access from new thread
- Cross-user: Team member accessing shared files
- Permission denial: Student trying to write to team scope

### Manual Testing
1. Create artifact in thread, close browser, return - verify persistence
2. Save artifact, start new thread - verify access
3. Mentor promotes to team - verify team member access
4. Guest tries to write - verify denial

---

## Migration Notes

### Existing Data
- Current `/memories/` and `/artifacts/` data uses flat namespace
- Migration script needed to move to new hierarchy
- Consider: Keep old data accessible during transition

### Backward Compatibility
- Support both old and new namespace formats initially
- Fallback to old namespace if new returns empty
- Deprecation warning for old format access

---

## Resolved Decisions

### 1. Team Assignment
**Decision**: Team IDs are determined via database lookup based on user.

**Database**: TBD - likely NocoDB or separate Postgres database (not BaseQL/Airtable)

- When a user authenticates, query the database to get their team memberships
- Database schema has `users` and `teams` tables with a junction table
- Return team_id(s) as part of user context

```typescript
// Frontend: /api/user/teams endpoint
interface UserTeams {
  user_id: string;
  teams: Array<{
    id: string;
    name: string;
  }>;
  active_team_id: string | null;  // Currently selected team
}
```

**Database query (placeholder - adapt to actual database)**:
```sql
-- Example for Postgres/NocoDB
SELECT t.id, t.name
FROM teams t
JOIN team_members tm ON t.id = tm.team_id
JOIN users u ON tm.user_id = u.id
WHERE u.auth0_id = $1;
```

### 2. Multiple Teams Support
**Decision**: Yes, users can belong to multiple teams.

**Implementation approach**:
- Store `active_team_id` in user session/context
- UI: Team selector dropdown in sidebar header when user has multiple teams
- API: Pass `team_id` (active team) in all requests
- File access: User can only access files for their **active** team at a time
- Team switching: Changing active team refreshes the workspace view

```typescript
// Frontend context
interface UserContext {
  user_id: string;
  tenant_id: string;
  teams: TeamMembership[];
  active_team_id: string | null;
  roles: string[];       // From Auth0 RBAC
  permissions: string[]; // From Auth0 RBAC
}

// On team switch:
// 1. Update active_team_id in context
// 2. Re-fetch thread list (filtered by team)
// 3. Clear current thread if it belongs to different team
```

### 3. User Roles & Permissions Source
**Decision**: Use Auth0 RBAC (Role-Based Access Control) with first-class Roles and Permissions.

**Auth0 RBAC Setup**:

1. **Create API in Auth0** (Dashboard > Applications > APIs):
   - Name: `xFoundry Filesystem API`
   - Identifier: `https://api.xfoundry.io/filesystem`
   - Enable RBAC: ✓
   - Add Permissions in Access Token: ✓

2. **Define Permissions on the API**:
   ```
   read:thread, write:thread
   read:user, write:user
   read:team, write:team
   read:tenant, write:tenant
   promote:to_user, promote:to_team, promote:to_tenant
   ```

3. **Create Roles** (Dashboard > User Management > Roles):
   - `guest` → `read:thread`, `read:tenant`
   - `student` → all `read:*`, `write:thread`, `write:user`, `promote:to_user`
   - `mentor` → all `read:*`, `write:thread`, `write:user`, `write:team`, `promote:to_user`, `promote:to_team`
   - `curator` → same as mentor
   - `admin` → all permissions
   - `super_admin` → all permissions

4. **Assign Roles to Users** (Dashboard > User Management > Users > [User] > Roles)

**Auth0 Post-Login Action** (to add roles to token):
```javascript
// Actions > Flows > Login > Add Action
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://xfoundry.io';

  if (event.authorization) {
    // Add roles array to tokens
    api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
    api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
  }
};
```

**Note**: Permissions are automatically included in the `scope` claim when "Add Permissions in Access Token" is enabled on the API. The Action above adds roles (which aren't included by default).

**Frontend retrieval**:
```typescript
// xfoundry-chat/src/lib/auth0.ts
interface Auth0TokenClaims {
  sub: string;
  email: string;
  "https://xfoundry.io/roles"?: string[];  // From Action
  scope?: string;  // Permissions as space-separated string
}

export function getUserRoles(token: Auth0TokenClaims): string[] {
  return token["https://xfoundry.io/roles"] || [];
}

export function getUserPermissions(token: Auth0TokenClaims): string[] {
  // Permissions come from scope claim when RBAC is enabled
  return token.scope?.split(" ").filter(s => s.includes(":")) || [];
}
```

**Backend config**:
```python
# Extract from request (frontend passes these from token)
config = get_thread_config(
    thread_id=thread_id,
    user_id=request.user_id,
    tenant_id=request.tenant_id,
    team_id=request.team_id,
    user_roles=request.user_context.roles if request.user_context else [],
    user_permissions=request.user_context.permissions if request.user_context else [],
)
```

---

## Deferred Decisions

These questions are acknowledged but deferred for future consideration:

1. **File versioning**: Track versions for promoted files?
   - *Deferred*: Not needed for MVP. Can add later if requested.

2. **Quotas**: Enforce storage quotas per scope?
   - *Deferred*: Monitor usage first, add if needed.

3. **Expiration**: Should thread-scoped files expire after N days?
   - *Deferred*: Likely useful for `/context/` cleanup. Consider 30-day retention policy later.

---

## References

### Auth0 RBAC
- [Auth0 RBAC Overview](https://auth0.com/docs/manage-users/access-control/rbac)
- [Enable RBAC for APIs](https://auth0.com/docs/get-started/apis/enable-role-based-access-control-for-apis)
- [Add Roles to Access Token (Community)](https://community.auth0.com/t/add-roles-to-access-token/132127)
- [Auth0 Actions - Post Login](https://auth0.com/docs/customize/actions/flows-and-triggers/login-flow)

### Deep Agents & LangGraph
- [Deep Agents Backends Documentation](https://docs.langchain.com/oss/python/deepagents/backends)
- [LangGraph Store Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence)
- [Deep Agents Human-in-the-Loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
