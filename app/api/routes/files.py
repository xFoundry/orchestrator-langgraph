"""File API endpoints for accessing virtual filesystem from UI.

Provides REST endpoints for:
- Listing files in different scopes (thread, user, tenant)
- Reading file content
- Promoting files to higher scopes
- Getting hierarchical file tree for UI display
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.persistence.redis import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/files", tags=["files"])


# =============================================================================
# Models
# =============================================================================


class FileInfo(BaseModel):
    """Information about a single file."""

    key: str = Field(..., description="File key/name")
    path: str = Field(..., description="Full virtual path")
    size: int = Field(0, description="Content size in bytes")
    scope: str = Field(..., description="File scope: thread, user, or tenant")


class FileListResponse(BaseModel):
    """Response for file listing endpoints."""

    files: list[FileInfo] = Field(default_factory=list)
    scope: str = Field(..., description="Scope that was queried")
    namespace: str = Field(..., description="Store namespace used")


class FileContentResponse(BaseModel):
    """Response for file read endpoint."""

    path: str
    content: str
    size: int


class FileTreeNode(BaseModel):
    """A node in the file tree (file or folder)."""

    name: str
    path: str
    type: str = Field(..., description="'file' or 'folder'")
    children: Optional[list["FileTreeNode"]] = None


class FileTreeResponse(BaseModel):
    """Hierarchical file tree for UI display."""

    tree: list[FileTreeNode] = Field(default_factory=list)


class PromoteFileRequest(BaseModel):
    """Request to promote a file to a higher scope."""

    source_path: str = Field(..., description="Path of file to promote")
    target_scope: str = Field(..., description="Target scope: 'user' or 'tenant'")
    new_name: Optional[str] = Field(None, description="Optional new filename")


class PromoteFileResponse(BaseModel):
    """Response for file promotion."""

    success: bool
    source_path: str
    target_path: str
    message: str


# =============================================================================
# Namespace Helpers
# =============================================================================


def get_namespace_for_scope(
    scope: str,
    tenant_id: str,
    user_id: str,
    thread_id: Optional[str] = None,
) -> tuple[str, ...]:
    """
    Build a namespace tuple for the given scope.

    Namespace hierarchy (simplified for staff MVP - no teams):
        tenant/shared/                          -> tenant scope
        tenant/users/{user_id}/saved/           -> user saved scope
        tenant/users/{user_id}/memories/        -> user memories scope
        tenant/users/{user_id}/{thread_id}/     -> thread scope
    """
    if scope == "tenant":
        return (tenant_id, "shared")

    if scope == "user_saved":
        return (tenant_id, "users", user_id, "saved")

    if scope == "user_memories":
        return (tenant_id, "users", user_id, "memories")

    if scope == "thread":
        if not thread_id:
            raise ValueError("thread_id required for thread scope")
        return (tenant_id, "users", user_id, thread_id)

    # Default to thread artifacts
    if scope == "thread_artifacts":
        if not thread_id:
            raise ValueError("thread_id required for thread_artifacts scope")
        return (tenant_id, "users", user_id, thread_id, "artifacts")

    if scope == "thread_context":
        if not thread_id:
            raise ValueError("thread_id required for thread_context scope")
        return (tenant_id, "users", user_id, thread_id, "context")

    raise ValueError(f"Unknown scope: {scope}")


def get_virtual_path_for_scope(scope: str, key: str) -> str:
    """Convert scope + key to virtual path."""
    scope_to_prefix = {
        "tenant": "/shared/",
        "user_saved": "/artifacts/saved/",
        "user_memories": "/memories/",
        "thread": "/",
        "thread_artifacts": "/artifacts/",
        "thread_context": "/context/",
    }
    prefix = scope_to_prefix.get(scope, "/")
    return f"{prefix}{key}"


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/list", response_model=FileListResponse)
async def list_files(
    scope: str = Query(
        "thread_artifacts",
        description="Scope to list: thread_artifacts, thread_context, user_saved, user_memories, tenant"
    ),
    tenant_id: str = Query("default", description="Tenant ID"),
    user_id: str = Query(..., description="User ID"),
    thread_id: Optional[str] = Query(None, description="Thread ID (required for thread scopes)"),
):
    """
    List files in the specified scope.

    Scopes:
    - thread_artifacts: Files in /artifacts/ for current thread
    - thread_context: Files in /context/ for current thread
    - user_saved: Files in /artifacts/saved/ (cross-thread)
    - user_memories: Files in /memories/ (cross-thread)
    - tenant: Files in /shared/ (all staff can access)
    """
    try:
        namespace = get_namespace_for_scope(scope, tenant_id, user_id, thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = await get_store()

    try:
        # Search for all items in this namespace using async method
        items = [item async for item in store.asearch(namespace)]

        files = []
        for item in items:
            content = item.value.get("content", "") if isinstance(item.value, dict) else ""
            files.append(FileInfo(
                key=item.key,
                path=get_virtual_path_for_scope(scope, item.key),
                size=len(content) if content else 0,
                scope=scope,
            ))

        return FileListResponse(
            files=files,
            scope=scope,
            namespace=":".join(namespace),
        )

    except Exception as e:
        logger.error(f"Error listing files in {namespace}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


@router.get("/read", response_model=FileContentResponse)
async def read_file(
    path: str = Query(..., description="Virtual path to read (e.g., /artifacts/report.md)"),
    tenant_id: str = Query("default", description="Tenant ID"),
    user_id: str = Query(..., description="User ID"),
    thread_id: Optional[str] = Query(None, description="Thread ID"),
):
    """
    Read file content by virtual path.

    Path examples:
    - /artifacts/report.md -> thread artifacts
    - /artifacts/saved/template.md -> user saved
    - /memories/facts.md -> user memories
    - /shared/guide.md -> tenant shared
    - /context/search_results.md -> thread context
    """
    # Determine scope from path
    if path.startswith("/shared/"):
        scope = "tenant"
        key = path[len("/shared/"):]
    elif path.startswith("/artifacts/saved/"):
        scope = "user_saved"
        key = path[len("/artifacts/saved/"):]
    elif path.startswith("/artifacts/"):
        scope = "thread_artifacts"
        key = path[len("/artifacts/"):]
    elif path.startswith("/memories/"):
        scope = "user_memories"
        key = path[len("/memories/"):]
    elif path.startswith("/context/"):
        scope = "thread_context"
        key = path[len("/context/"):]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    try:
        namespace = get_namespace_for_scope(scope, tenant_id, user_id, thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = await get_store()

    try:
        # Use async method
        result = await store.aget(namespace, key)

        if not result:
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        content = ""
        if isinstance(result.value, dict):
            content = result.value.get("content", "")
        elif isinstance(result.value, str):
            content = result.value

        return FileContentResponse(
            path=path,
            content=content,
            size=len(content),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@router.get("/tree", response_model=FileTreeResponse)
async def get_file_tree(
    tenant_id: str = Query("default", description="Tenant ID"),
    user_id: str = Query(..., description="User ID"),
    thread_id: Optional[str] = Query(None, description="Thread ID"),
    include_thread: bool = Query(True, description="Include thread-scoped files"),
    include_user: bool = Query(True, description="Include user-scoped files"),
    include_tenant: bool = Query(True, description="Include tenant-scoped files"),
):
    """
    Get hierarchical file tree for UI display.

    Returns a tree structure suitable for the FileTree component.
    """
    logger.debug(f"[FileTree] Request: tenant_id={tenant_id}, user_id={user_id[:20] if user_id else 'None'}..., thread_id={thread_id}")

    store = await get_store()
    tree: list[FileTreeNode] = []

    async def get_files_for_scope(scope: str) -> list[FileInfo]:
        try:
            namespace = get_namespace_for_scope(scope, tenant_id, user_id, thread_id)
            logger.debug(f"[FileTree] Querying namespace for scope '{scope}': {namespace}")
            # Use async method to avoid blocking event loop
            items = [item async for item in store.asearch(namespace)]
            logger.debug(f"[FileTree] Found {len(items)} items in scope '{scope}'")
            return [
                FileInfo(
                    key=item.key,
                    path=get_virtual_path_for_scope(scope, item.key),
                    size=len(item.value.get("content", "")) if isinstance(item.value, dict) else 0,
                    scope=scope,
                )
                for item in items
            ]
        except Exception as e:
            logger.warning(f"Error getting files for scope {scope}: {e}")
            return []

    # Thread-scoped files
    if include_thread and thread_id:
        # Artifacts folder
        artifacts = await get_files_for_scope("thread_artifacts")
        if artifacts:
            tree.append(FileTreeNode(
                name="artifacts",
                path="/artifacts",
                type="folder",
                children=[
                    FileTreeNode(name=f.key, path=f.path, type="file")
                    for f in artifacts
                ],
            ))

        # Context folder
        context = await get_files_for_scope("thread_context")
        if context:
            tree.append(FileTreeNode(
                name="context",
                path="/context",
                type="folder",
                children=[
                    FileTreeNode(name=f.key, path=f.path, type="file")
                    for f in context
                ],
            ))

    # User-scoped files
    if include_user:
        # Saved artifacts
        saved = await get_files_for_scope("user_saved")
        if saved:
            # Find or create artifacts folder, then add saved subfolder
            artifacts_folder = next((n for n in tree if n.path == "/artifacts"), None)
            saved_node = FileTreeNode(
                name="saved",
                path="/artifacts/saved",
                type="folder",
                children=[
                    FileTreeNode(name=f.key, path=f.path, type="file")
                    for f in saved
                ],
            )
            if artifacts_folder and artifacts_folder.children is not None:
                artifacts_folder.children.append(saved_node)
            else:
                tree.append(FileTreeNode(
                    name="artifacts",
                    path="/artifacts",
                    type="folder",
                    children=[saved_node],
                ))

        # Memories
        memories = await get_files_for_scope("user_memories")
        if memories:
            tree.append(FileTreeNode(
                name="memories",
                path="/memories",
                type="folder",
                children=[
                    FileTreeNode(name=f.key, path=f.path, type="file")
                    for f in memories
                ],
            ))

    # Tenant-scoped files
    if include_tenant:
        shared = await get_files_for_scope("tenant")
        if shared:
            tree.append(FileTreeNode(
                name="shared",
                path="/shared",
                type="folder",
                children=[
                    FileTreeNode(name=f.key, path=f.path, type="file")
                    for f in shared
                ],
            ))

    return FileTreeResponse(tree=tree)


@router.post("/promote", response_model=PromoteFileResponse)
async def promote_file(
    request: PromoteFileRequest,
    tenant_id: str = Query("default", description="Tenant ID"),
    user_id: str = Query(..., description="User ID"),
    thread_id: Optional[str] = Query(None, description="Thread ID"),
):
    """
    Promote a file to a higher scope.

    - thread -> user: Save to /artifacts/saved/
    - user -> tenant: Share to /shared/

    Staff users can promote to any scope (no permission check in MVP).
    """
    source_path = request.source_path
    target_scope = request.target_scope

    # Determine source scope from path
    if source_path.startswith("/shared/"):
        raise HTTPException(status_code=400, detail="Cannot promote from tenant scope")
    elif source_path.startswith("/artifacts/saved/"):
        source_scope = "user_saved"
        source_key = source_path[len("/artifacts/saved/"):]
    elif source_path.startswith("/artifacts/"):
        source_scope = "thread_artifacts"
        source_key = source_path[len("/artifacts/"):]
    elif source_path.startswith("/context/"):
        source_scope = "thread_context"
        source_key = source_path[len("/context/"):]
    else:
        raise HTTPException(status_code=400, detail=f"Invalid source path: {source_path}")

    # Validate target scope
    if target_scope not in ("user", "tenant"):
        raise HTTPException(status_code=400, detail="target_scope must be 'user' or 'tenant'")

    # Map target scope to internal scope name
    target_scope_internal = "user_saved" if target_scope == "user" else "tenant"

    try:
        source_namespace = get_namespace_for_scope(source_scope, tenant_id, user_id, thread_id)
        target_namespace = get_namespace_for_scope(target_scope_internal, tenant_id, user_id, thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    store = await get_store()

    # Read source file
    try:
        # Use async method
        result = await store.aget(source_namespace, source_key)
        if not result:
            raise HTTPException(status_code=404, detail=f"Source file not found: {source_path}")

        source_value = result.value
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading source file {source_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read source file: {e}")

    # Determine target key
    target_key = request.new_name or source_key
    target_path = get_virtual_path_for_scope(target_scope_internal, target_key)

    # Write to target
    try:
        # Update the path in the value
        if isinstance(source_value, dict):
            source_value["path"] = target_path
            source_value["promoted_from"] = source_path

        # Use async method
        await store.aput(target_namespace, target_key, source_value)

        return PromoteFileResponse(
            success=True,
            source_path=source_path,
            target_path=target_path,
            message=f"File promoted to {target_scope} scope",
        )
    except Exception as e:
        logger.error(f"Error promoting file to {target_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to promote file: {e}")
