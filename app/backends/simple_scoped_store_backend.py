"""
SimpleScopedStoreBackend - Hierarchical namespace storage for Deep Agents.

Provides persistent file storage with hierarchical namespacing:
    tenant/shared/                          -> /shared/*
    tenant/users/{user_id}/saved/           -> /artifacts/saved/*
    tenant/users/{user_id}/memories/        -> /memories/*
    tenant/users/{user_id}/{thread_id}/artifacts/ -> /artifacts/*
    tenant/users/{user_id}/{thread_id}/context/   -> /context/*

This backend implements the Deep Agents BackendProtocol to provide
persistent storage for different file scopes while maintaining isolation
between tenants, users, and threads.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.config import get_config
from langgraph.store.base import BaseStore

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileInfo,
    GrepMatch,
    WriteResult,
    FileUploadResponse,
    FileDownloadResponse,
)

logger = logging.getLogger(__name__)


class SimpleScopedStoreBackend(BackendProtocol):
    """
    Backend that persists files to LangGraph Store with hierarchical namespacing.

    Namespacing is based on:
    - tenant_id: Multi-tenancy isolation
    - user_id: User-level isolation
    - thread_id: Thread/conversation isolation

    Path prefixes map to different namespace scopes:
    - /shared/* -> tenant-wide (all staff can access)
    - /artifacts/saved/* -> user-wide (persists across threads)
    - /memories/* -> user-wide (persists across threads)
    - /artifacts/* -> thread-scoped
    - /context/* -> thread-scoped
    """

    # Path prefix to scope mapping (order matters - longest match first)
    PATH_SCOPES = [
        ("/shared/", "tenant"),
        ("/artifacts/saved/", "user_saved"),
        ("/memories/", "user_memories"),
        ("/artifacts/", "thread_artifacts"),
        ("/context/", "thread_context"),
    ]

    def __init__(self, runtime: Any = None):
        """
        Initialize the backend.

        Args:
            runtime: Deep Agent runtime (contains store reference)
        """
        self._runtime = runtime
        self._store: Optional[BaseStore] = None

    @property
    def store(self) -> BaseStore:
        """Get the LangGraph store from runtime or global."""
        if self._store is not None:
            return self._store

        # Try to get from runtime
        if self._runtime is not None:
            if hasattr(self._runtime, "store"):
                self._store = self._runtime.store
                return self._store

        # Fallback: try to get from app state (async singleton)
        # This requires the store to be initialized at app startup
        raise RuntimeError("No store available. Ensure store is initialized.")

    def _get_config(self) -> dict:
        """Get the current LangGraph config."""
        try:
            config = get_config()
            return config.get("configurable", {})
        except Exception:
            return {}

    def _get_scope_for_path(self, path: str) -> str:
        """Determine the scope for a given path."""
        for prefix, scope in self.PATH_SCOPES:
            if path.startswith(prefix):
                return scope
        # Default to thread context for unknown paths
        return "thread_context"

    def _get_namespace(self, path: str) -> tuple[str, ...]:
        """
        Build namespace tuple for the given path.

        Uses config values for tenant_id, user_id, thread_id.
        """
        cfg = self._get_config()
        tenant_id = cfg.get("tenant_id", "default")
        user_id = cfg.get("user_id", "anonymous")
        thread_id = cfg.get("thread_id")

        scope = self._get_scope_for_path(path)

        if scope == "tenant":
            return (tenant_id, "shared")

        if scope == "user_saved":
            return (tenant_id, "users", user_id, "saved")

        if scope == "user_memories":
            return (tenant_id, "users", user_id, "memories")

        # Thread-scoped
        if not thread_id:
            # Fallback to user scope if no thread_id
            logger.warning(f"No thread_id in config, using user scope for {path}")
            return (tenant_id, "users", user_id, "fallback")

        if scope == "thread_artifacts":
            return (tenant_id, "users", user_id, thread_id, "artifacts")

        if scope == "thread_context":
            return (tenant_id, "users", user_id, thread_id, "context")

        # Default
        return (tenant_id, "users", user_id, thread_id)

    def _get_key_from_path(self, path: str) -> str:
        """Extract the key (filename) from a path."""
        for prefix, _ in self.PATH_SCOPES:
            if path.startswith(prefix):
                return path[len(prefix):]
        # If no prefix matches, use the path as-is (strip leading /)
        return path.lstrip("/")

    def _get_path_from_key(self, key: str, scope: str) -> str:
        """Reconstruct the full path from a key and scope."""
        scope_to_prefix = {
            "tenant": "/shared/",
            "user_saved": "/artifacts/saved/",
            "user_memories": "/memories/",
            "thread_artifacts": "/artifacts/",
            "thread_context": "/context/",
        }
        prefix = scope_to_prefix.get(scope, "/")
        return f"{prefix}{key}"

    def _format_lines(self, content: str, offset: int = 0, limit: int = 2000) -> str:
        """Format content with line numbers (cat -n format)."""
        lines = content.split("\n")
        lines = lines[offset:offset + limit]

        formatted = []
        for i, line in enumerate(lines, start=offset + 1):
            # Truncate lines longer than 2000 chars
            if len(line) > 2000:
                line = line[:2000] + "..."
            formatted.append(f"{i:6}\t{line}")

        return "\n".join(formatted)

    # =========================================================================
    # BackendProtocol Implementation - Sync Methods
    # =========================================================================

    def ls_info(self, path: str) -> list[FileInfo]:
        """List all files in a directory with metadata."""
        # Normalize path
        if not path.endswith("/"):
            path = path + "/"
        if path == "/":
            # Root listing - return top-level directories
            return [
                {"path": "/artifacts/", "is_dir": True, "size": 0},
                {"path": "/memories/", "is_dir": True, "size": 0},
                {"path": "/context/", "is_dir": True, "size": 0},
                {"path": "/shared/", "is_dir": True, "size": 0},
            ]

        namespace = self._get_namespace(path)
        scope = self._get_scope_for_path(path)

        try:
            # Run async operation - use event loop if running, else run_until_complete
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_ls(namespace))
                    items = future.result()
            else:
                items = asyncio.run(self._async_ls(namespace))

            files: list[FileInfo] = []
            for item in items:
                file_path = self._get_path_from_key(item.key, scope)
                content = item.value.get("content", "") if isinstance(item.value, dict) else ""
                modified_at = item.value.get("modified_at", "") if isinstance(item.value, dict) else ""
                files.append({
                    "path": file_path,
                    "is_dir": False,
                    "size": len(content),
                    "modified_at": modified_at,
                })
            return files

        except Exception as e:
            logger.error(f"Error listing {path}: {e}")
            return []

    async def _async_ls(self, namespace: tuple) -> list:
        """Async list operation."""
        # Use async method to avoid blocking event loop
        return list(await self.store.asearch(namespace))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with line numbers."""
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_read(namespace, key))
                    result = future.result()
            else:
                result = asyncio.run(self._async_read(namespace, key))

            if result is None:
                return f"Error: File not found: {file_path}"

            return self._format_lines(result, offset, limit)

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return f"Error reading file: {e}"

    async def _async_read(self, namespace: tuple, key: str) -> Optional[str]:
        """Async read operation."""
        # Use async method to avoid blocking event loop
        result = await self.store.aget(namespace, key)
        if result and result.value:
            if isinstance(result.value, dict):
                return result.value.get("content", "")
            return str(result.value)
        return None

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to a file (creates new file)."""
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)
        scope = self._get_scope_for_path(file_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._async_write(namespace, key, file_path, content, scope)
                    )
                    future.result()
            else:
                asyncio.run(self._async_write(namespace, key, file_path, content, scope))

            logger.debug(f"Wrote {len(content)} bytes to {file_path}")
            return WriteResult(path=file_path, files_update=None)

        except Exception as e:
            logger.error(f"Error writing {file_path}: {e}")
            return WriteResult(error=str(e))

    async def _async_write(
        self,
        namespace: tuple,
        key: str,
        path: str,
        content: str,
        scope: str
    ) -> None:
        """Async write operation."""
        # Use async method to avoid blocking event loop
        now = datetime.now(timezone.utc).isoformat()
        logger.info(f"[SimpleScopedStoreBackend] Writing file: path={path}, key={key}, scope={scope}")
        logger.info(f"[SimpleScopedStoreBackend] Namespace: {namespace}")
        cfg = self._get_config()
        logger.info(f"[SimpleScopedStoreBackend] Config: tenant_id={cfg.get('tenant_id')}, user_id={cfg.get('user_id', 'None')[:20] if cfg.get('user_id') else 'None'}..., thread_id={cfg.get('thread_id')}")
        await self.store.aput(
            namespace,
            key,
            {
                "content": content,
                "path": path,
                "scope": scope,
                "created_at": now,
                "modified_at": now,
            }
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Perform exact string replacements in an existing file."""
        # Read current content (raw, not formatted)
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._async_read(namespace, key))
                    content = future.result()
            else:
                content = asyncio.run(self._async_read(namespace, key))
        except Exception as e:
            return EditResult(error=f"Error reading file: {e}")

        if content is None:
            return EditResult(error=f"File not found: {file_path}")

        # Count and perform replacements
        if replace_all:
            count = content.count(old_string)
            new_content = content.replace(old_string, new_string)
        else:
            count = 1 if old_string in content else 0
            new_content = content.replace(old_string, new_string, 1)

        if count == 0:
            return EditResult(error=f"Text not found in {file_path}")

        # Write back
        result = self.write(file_path, new_content)
        if result.error:
            return EditResult(error=result.error)

        return EditResult(path=file_path, occurrences=count, files_update=None)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern in files."""
        # Get files to search
        search_path = path or "/"
        if glob:
            files = self.glob_info(glob, search_path)
        else:
            files = self.ls_info(search_path)
            # Also check subdirectories
            for prefix, _ in self.PATH_SCOPES:
                if prefix.startswith(search_path) or search_path == "/":
                    files.extend(self.ls_info(prefix))

        # Remove directories and duplicates
        files = [f for f in files if not f.get("is_dir", False)]
        seen = set()
        unique_files = []
        for f in files:
            if f["path"] not in seen:
                seen.add(f["path"])
                unique_files.append(f)

        # Search in each file (literal match, not regex)
        matches: list[GrepMatch] = []

        for f in unique_files:
            # Read raw content
            namespace = self._get_namespace(f["path"])
            key = self._get_key_from_path(f["path"])

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._async_read(namespace, key))
                        content = future.result()
                else:
                    content = asyncio.run(self._async_read(namespace, key))
            except Exception:
                continue

            if content is None:
                continue

            for i, line in enumerate(content.split("\n"), 1):
                if pattern in line:  # Literal match
                    matches.append({
                        "path": f["path"],
                        "line": i,
                        "text": line.strip(),
                    })

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern."""
        # Get all files recursively
        all_files: list[FileInfo] = []
        for prefix, scope in self.PATH_SCOPES:
            files = self.ls_info(prefix)
            all_files.extend(files)

        # Filter by pattern
        matches: list[FileInfo] = []
        for f in all_files:
            file_path = f["path"]
            filename = file_path.split("/")[-1]
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(filename, pattern):
                matches.append(f)

        return matches

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files."""
        responses = []
        for path, content in files:
            try:
                # Decode bytes to string for text files
                text_content = content.decode("utf-8")
                result = self.write(path, text_content)
                responses.append(FileUploadResponse(path=path, error=result.error))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files."""
        responses = []
        for path in paths:
            namespace = self._get_namespace(path)
            key = self._get_key_from_path(path)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, self._async_read(namespace, key))
                        content = future.result()
                else:
                    content = asyncio.run(self._async_read(namespace, key))

                if content is None:
                    responses.append(FileDownloadResponse(path=path, error="file_not_found"))
                else:
                    responses.append(FileDownloadResponse(path=path, content=content.encode("utf-8")))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))

        return responses

    # =========================================================================
    # BackendProtocol Implementation - Async Methods
    # These override the default implementations to use native async
    # =========================================================================

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async version of ls_info."""
        # Normalize path
        if not path.endswith("/"):
            path = path + "/"
        if path == "/":
            return [
                {"path": "/artifacts/", "is_dir": True, "size": 0},
                {"path": "/memories/", "is_dir": True, "size": 0},
                {"path": "/context/", "is_dir": True, "size": 0},
                {"path": "/shared/", "is_dir": True, "size": 0},
            ]

        namespace = self._get_namespace(path)
        scope = self._get_scope_for_path(path)

        try:
            items = await self._async_ls(namespace)
            files: list[FileInfo] = []
            for item in items:
                file_path = self._get_path_from_key(item.key, scope)
                content = item.value.get("content", "") if isinstance(item.value, dict) else ""
                modified_at = item.value.get("modified_at", "") if isinstance(item.value, dict) else ""
                files.append({
                    "path": file_path,
                    "is_dir": False,
                    "size": len(content),
                    "modified_at": modified_at,
                })
            return files
        except Exception as e:
            logger.error(f"Error listing {path}: {e}")
            return []

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async version of read."""
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)

        try:
            result = await self._async_read(namespace, key)
            if result is None:
                return f"Error: File not found: {file_path}"
            return self._format_lines(result, offset, limit)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return f"Error reading file: {e}"

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async version of write."""
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)
        scope = self._get_scope_for_path(file_path)

        try:
            await self._async_write(namespace, key, file_path, content, scope)
            logger.debug(f"Wrote {len(content)} bytes to {file_path}")
            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            logger.error(f"Error writing {file_path}: {e}")
            return WriteResult(error=str(e))

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of edit."""
        namespace = self._get_namespace(file_path)
        key = self._get_key_from_path(file_path)

        try:
            content = await self._async_read(namespace, key)
        except Exception as e:
            return EditResult(error=f"Error reading file: {e}")

        if content is None:
            return EditResult(error=f"File not found: {file_path}")

        if replace_all:
            count = content.count(old_string)
            new_content = content.replace(old_string, new_string)
        else:
            count = 1 if old_string in content else 0
            new_content = content.replace(old_string, new_string, 1)

        if count == 0:
            return EditResult(error=f"Text not found in {file_path}")

        result = await self.awrite(file_path, new_content)
        if result.error:
            return EditResult(error=result.error)

        return EditResult(path=file_path, occurrences=count, files_update=None)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async version of grep_raw."""
        search_path = path or "/"
        if glob:
            files = await self.aglob_info(glob, search_path)
        else:
            files = await self.als_info(search_path)
            for prefix, _ in self.PATH_SCOPES:
                if prefix.startswith(search_path) or search_path == "/":
                    files.extend(await self.als_info(prefix))

        files = [f for f in files if not f.get("is_dir", False)]
        seen = set()
        unique_files = []
        for f in files:
            if f["path"] not in seen:
                seen.add(f["path"])
                unique_files.append(f)

        matches: list[GrepMatch] = []

        for f in unique_files:
            namespace = self._get_namespace(f["path"])
            key = self._get_key_from_path(f["path"])

            try:
                content = await self._async_read(namespace, key)
            except Exception:
                continue

            if content is None:
                continue

            for i, line in enumerate(content.split("\n"), 1):
                if pattern in line:
                    matches.append({
                        "path": f["path"],
                        "line": i,
                        "text": line.strip(),
                    })

        return matches

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async version of glob_info."""
        all_files: list[FileInfo] = []
        for prefix, scope in self.PATH_SCOPES:
            files = await self.als_info(prefix)
            all_files.extend(files)

        matches: list[FileInfo] = []
        for f in all_files:
            file_path = f["path"]
            filename = file_path.split("/")[-1]
            if fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(filename, pattern):
                matches.append(f)

        return matches

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        responses = []
        for path, content in files:
            try:
                text_content = content.decode("utf-8")
                result = await self.awrite(path, text_content)
                responses.append(FileUploadResponse(path=path, error=result.error))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        responses = []
        for path in paths:
            namespace = self._get_namespace(path)
            key = self._get_key_from_path(path)

            try:
                content = await self._async_read(namespace, key)
                if content is None:
                    responses.append(FileDownloadResponse(path=path, error="file_not_found"))
                else:
                    responses.append(FileDownloadResponse(path=path, content=content.encode("utf-8")))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))

        return responses


def create_simple_scoped_backend(store: BaseStore, runtime: Any = None):
    """
    Factory function to create a SimpleScopedStoreBackend.

    Args:
        store: LangGraph BaseStore instance
        runtime: Optional Deep Agent runtime

    Returns:
        SimpleScopedStoreBackend instance
    """
    backend = SimpleScopedStoreBackend(runtime)
    backend._store = store
    return backend
