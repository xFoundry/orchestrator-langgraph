"""
Mentor Hub API Tools - Direct access to application data.

Provides LangGraph tools for accessing live data from the Mentor Hub
application. These tools supplement the Cognee RAG data with real-time
information about sessions, teams, mentors, and tasks.

All tools call the Mentor Hub Next.js API endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.sanitize import sanitize_for_json

logger = logging.getLogger(__name__)


def _get_mentor_hub_url() -> str:
    """Get the Mentor Hub API URL from settings or default."""
    settings = get_settings()
    # Default to port 3001 (dev server), config can override if needed
    return getattr(settings, "mentor_hub_api_url", "http://localhost:3001/api")


async def _mentor_hub_graphql(
    query: str,
    variables: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Make a GraphQL request to the Mentor Hub API proxy."""
    base_url = _get_mentor_hub_url()
    url = f"{base_url}/graphql"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={"query": query, "variables": variables or {}},
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return {"error": data["errors"][0].get("message", "GraphQL error")}

            return data.get("data", {})

    except httpx.HTTPError as e:
        logger.error(f"Mentor Hub GraphQL request failed: {e}")
        return {"error": str(e)}


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class SessionsInput(BaseModel):
    """Input for getting sessions."""
    team_id: Optional[str] = Field(None, description="Filter by team name")
    mentor_id: Optional[str] = Field(None, description="Filter by mentor name")
    limit: int = Field(default=10, description="Maximum number of sessions")
    upcoming_only: bool = Field(default=True, description="Only return future sessions (set to False for past sessions)")
    include_transcript: bool = Field(default=False, description="Include full transcript in response (can be large)")


class TeamInput(BaseModel):
    """Input for getting team details."""
    team_id: str = Field(..., description="The team's unique ID")


class MentorSearchInput(BaseModel):
    """Input for searching mentors."""
    expertise: Optional[str] = Field(None, description="Expertise area to search for")
    cohort_id: Optional[str] = Field(None, description="Filter by cohort")
    limit: int = Field(default=20, description="Maximum results")


class TasksInput(BaseModel):
    """Input for getting tasks."""
    team_id: Optional[str] = Field(None, description="Filter by team name")
    assignee_id: Optional[str] = Field(None, description="Filter by assignee name")
    status: Optional[str] = Field(None, description="Filter by status (Not Started, In Progress, Completed, Cancelled)")
    limit: int = Field(default=20, description="Maximum number of tasks")


# =============================================================================
# TOOLS
# =============================================================================

@tool(args_schema=SessionsInput)
async def get_mentor_hub_sessions(
    team_id: Optional[str] = None,
    mentor_id: Optional[str] = None,
    limit: int = 10,
    upcoming_only: bool = True,
    include_transcript: bool = False,
) -> dict[str, Any]:
    """
    Get mentorship sessions from Mentor Hub.

    Use this to answer questions about:
    - Upcoming sessions for a team or mentor (upcoming_only=True)
    - Past session history and what was discussed (upcoming_only=False)
    - Session details, agendas, summaries, and key topics

    This provides LIVE data from the application, not historical RAG data.

    Args:
        team_id: Filter by team name (partial match)
        mentor_id: Filter by mentor name (partial match)
        limit: Maximum sessions to return
        upcoming_only: True for future sessions, False for past sessions
        include_transcript: Include full transcript in response (can be large, default False)

    Returns:
        List of sessions with details including date, mentors, team, agenda, summary, key topics
    """
    logger.info(f"Fetching Mentor Hub sessions: team={team_id}, mentor={mentor_id}, upcoming_only={upcoming_only}")

    # Conditionally include fullTranscript (can be very large)
    transcript_field = "fullTranscript" if include_transcript else ""

    # Order by date - descending for past sessions, ascending for upcoming
    order_direction = "asc" if upcoming_only else "desc"

    # BaseQL uses _order_by syntax - no status filter, we filter by date client-side
    query = f"""
    query GetSessions {{
        sessions(_order_by: {{scheduledStart: "{order_direction}"}}) {{
            id
            sessionId
            sessionType
            scheduledStart
            duration
            status
            agenda
            summary
            keyTopics
            meetingUrl
            recordingUrl
            granolaNotesUrl
            {transcript_field}
            team {{
                id
                teamName
            }}
            mentor {{
                id
                fullName
            }}
        }}
    }}
    """

    result = await _mentor_hub_graphql(query)

    if "error" in result:
        return result

    sessions = result.get("sessions", [])

    # Client-side filtering by date
    now = datetime.now(timezone.utc)
    filtered_sessions = []
    for s in sessions:
        scheduled_start = s.get("scheduledStart")
        if scheduled_start:
            try:
                # Parse ISO format datetime
                session_dt = datetime.fromisoformat(scheduled_start.replace("Z", "+00:00"))
                if upcoming_only:
                    # Future sessions only
                    if session_dt >= now:
                        filtered_sessions.append(s)
                else:
                    # Past sessions only
                    if session_dt < now:
                        filtered_sessions.append(s)
            except (ValueError, TypeError):
                # If we can't parse the date, include the session
                filtered_sessions.append(s)
        else:
            filtered_sessions.append(s)
    sessions = filtered_sessions

    # Client-side filtering for team/mentor (BaseQL doesn't support nested filters well)
    if team_id:
        team_id_lower = team_id.lower()
        sessions = [
            s for s in sessions
            if s.get("team") and any(
                team_id_lower in t.get("teamName", "").lower()
                for t in (s.get("team") if isinstance(s.get("team"), list) else [s.get("team")])
            )
        ]
    if mentor_id:
        mentor_id_lower = mentor_id.lower()
        sessions = [
            s for s in sessions
            if s.get("mentor") and any(
                mentor_id_lower in m.get("fullName", "").lower()
                for m in (s.get("mentor") if isinstance(s.get("mentor"), list) else [s.get("mentor")])
            )
        ]

    # Apply limit after client-side filtering
    sessions = sessions[:limit]

    # Format for readability
    formatted = []
    for session in sessions:
        team_data = session.get("team")
        mentor_data = session.get("mentor")
        session_info = {
            "id": session.get("id"),
            "session_id": session.get("sessionId"),
            "type": session.get("sessionType"),
            "scheduled_start": session.get("scheduledStart"),
            "duration": session.get("duration"),
            "status": session.get("status"),
            "team": (team_data[0].get("teamName") if isinstance(team_data, list) and team_data else
                     team_data.get("teamName") if isinstance(team_data, dict) else None),
            "mentor": (mentor_data[0].get("fullName") if isinstance(mentor_data, list) and mentor_data else
                       mentor_data.get("fullName") if isinstance(mentor_data, dict) else None),
            "agenda": session.get("agenda"),
            "summary": session.get("summary"),
            "key_topics": session.get("keyTopics"),
            "meeting_url": session.get("meetingUrl"),
            "recording_url": session.get("recordingUrl"),
            "granola_notes_url": session.get("granolaNotesUrl"),
        }
        # Only include transcript if requested and available
        if include_transcript and session.get("fullTranscript"):
            session_info["full_transcript"] = session.get("fullTranscript")
        formatted.append(session_info)

    return sanitize_for_json({
        "sessions": formatted,
        "count": len(formatted),
        "source": "mentor_hub_live",
    })


@tool(args_schema=TeamInput)
async def get_mentor_hub_team(team_id: str) -> dict[str, Any]:
    """
    Get detailed information about a team from Mentor Hub.

    Use this to get:
    - Team members and their contact info
    - Team's project description
    - Current cohort
    - Team status

    This provides LIVE data from the application.

    Args:
        team_id: The team's unique identifier (name or ID)

    Returns:
        Team details including members, project info, and status
    """
    logger.info(f"Fetching Mentor Hub team: {team_id}")

    # BaseQL requires _filter syntax and GraphQL variables for nested relations to work
    query = """
    query GetTeamDetail($teamName: String!) {
        teams(_filter: {teamName: {_eq: $teamName}}) {
            id
            teamId
            teamName
            description
            teamStatus
            activeCount
            members(_filter: {status: {_eq: "Active"}}) {
                id
                memberId
                status
                type
                memberName
                emailFromContact
                contact {
                    id
                    fullName
                    email
                    bio
                    linkedIn
                }
            }
            cohorts {
                id
                shortName
                status
            }
            mentorshipSessions(_order_by: {scheduledStart: "desc"}) {
                id
                sessionType
                scheduledStart
                status
                mentor {
                    fullName
                }
            }
        }
    }
    """

    # First try exact match
    result = await _mentor_hub_graphql(query, variables={"teamName": team_id})

    if "error" in result:
        return result

    teams = result.get("teams", [])

    # If exact match fails, try case-insensitive search
    if not teams:
        logger.info(f"Exact match failed for '{team_id}', trying case-insensitive search")
        # Fetch all teams and filter client-side
        all_teams_query = """
        query GetAllTeams {
            teams {
                id
                teamId
                teamName
                description
                teamStatus
                activeCount
                members(_filter: {status: {_eq: "Active"}}) {
                    id
                    memberId
                    status
                    type
                    memberName
                    emailFromContact
                    contact {
                        id
                        fullName
                        email
                        bio
                        linkedIn
                    }
                }
                cohorts {
                    id
                    shortName
                    status
                }
                mentorshipSessions(_order_by: {scheduledStart: "desc"}) {
                    id
                    sessionType
                    scheduledStart
                    status
                    mentor {
                        fullName
                    }
                }
            }
        }
        """
        all_result = await _mentor_hub_graphql(all_teams_query)
        if "error" not in all_result:
            all_teams = all_result.get("teams", [])
            team_id_lower = team_id.lower()
            for t in all_teams:
                if team_id_lower in t.get("teamName", "").lower():
                    teams = [t]
                    logger.info(f"Found team via case-insensitive match: {t.get('teamName')}")
                    break

    if not teams:
        return {"error": f"Team '{team_id}' not found. Try using find_entity first to get the exact team name."}

    team = teams[0]

    def extract_field(data, field: str):
        """Extract field from data which may be dict or list."""
        if isinstance(data, list) and data:
            return data[0].get(field)
        elif isinstance(data, dict):
            return data.get(field)
        return None

    def extract_all(data, field: str):
        """Extract field from all items in data."""
        if isinstance(data, list):
            return [item.get(field) for item in data if item.get(field)]
        elif isinstance(data, dict):
            return [data.get(field)] if data.get(field) else []
        return []

    # Members are already filtered to Active by the GraphQL query
    active_members = team.get("members", [])

    # Debug: log raw member data
    if active_members:
        logger.info(f"Raw member sample: {active_members[0]}")

    logger.info(f"Team {team_id}: {len(active_members)} active members")

    def format_member(m):
        # memberName and emailFromContact are arrays of strings from BaseQL
        # e.g., memberName: ["Srinidhi Gubba"], emailFromContact: ["sgubba@terpmail.umd.edu"]
        member_name = m.get("memberName", [])
        member_email = m.get("emailFromContact", [])
        contact = m.get("contact", [])

        # Extract name - from memberName array or fallback to contact
        if isinstance(member_name, list) and member_name:
            name = member_name[0]
        elif isinstance(contact, list) and contact:
            name = contact[0].get("fullName") if isinstance(contact[0], dict) else None
        elif isinstance(contact, dict):
            name = contact.get("fullName")
        else:
            name = None

        # Extract email - from emailFromContact array or fallback to contact
        if isinstance(member_email, list) and member_email:
            email = member_email[0]
        elif isinstance(contact, list) and contact:
            email = contact[0].get("email") if isinstance(contact[0], dict) else None
        elif isinstance(contact, dict):
            email = contact.get("email")
        else:
            email = None

        # Extract bio from contact if available
        bio = None
        if isinstance(contact, list) and contact and isinstance(contact[0], dict):
            bio = contact[0].get("bio")
        elif isinstance(contact, dict):
            bio = contact.get("bio")

        result = {
            "name": name,
            "email": email,
            "type": m.get("type"),
            "status": m.get("status"),
            "bio": bio,
        }
        logger.debug(f"Formatted member: {name} ({m.get('type')})")
        return result

    # Process sessions - get recent ones
    sessions = team.get("mentorshipSessions", [])
    recent_sessions = []
    for s in sessions[:5]:  # Limit to 5 recent sessions
        mentor_data = s.get("mentor")
        recent_sessions.append({
            "type": s.get("sessionType"),
            "date": s.get("scheduledStart"),
            "status": s.get("status"),
            "mentor": extract_field(mentor_data, "fullName"),
        })

    formatted_active = [format_member(m) for m in active_members]

    logger.info(f"Returning team data with {len(formatted_active)} active members: {[m.get('name') for m in formatted_active]}")

    return sanitize_for_json({
        "id": team.get("id"),
        "team_id": team.get("teamId"),
        "name": team.get("teamName"),
        "description": team.get("description"),
        "status": team.get("teamStatus"),
        "active_member_count": team.get("activeCount"),
        "active_members": formatted_active,
        "cohorts": extract_all(team.get("cohorts"), "shortName"),
        "recent_sessions": recent_sessions,
        "source": "mentor_hub_live",
    })


@tool(args_schema=MentorSearchInput)
async def search_mentor_hub_mentors(
    expertise: Optional[str] = None,
    cohort_id: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for mentors in Mentor Hub.

    Use this to:
    - Find mentors with specific expertise
    - Get mentor availability
    - Find mentors in a specific cohort
    - Get mentor profiles for matching

    This provides LIVE data from the application.

    Args:
        expertise: Expertise area to search for
        cohort_id: Filter by cohort
        limit: Maximum results

    Returns:
        List of mentors with profiles and expertise
    """
    logger.info(f"Searching Mentor Hub mentors: expertise={expertise}")

    # BaseQL query - fetch contacts with participation data
    query = """
    query SearchMentors {
        contacts {
            id
            fullName
            email
            bio
            linkedIn
            participation {
                capacity
                cohorts {
                    shortName
                }
            }
        }
    }
    """

    result = await _mentor_hub_graphql(query)

    if "error" in result:
        return result

    contacts = result.get("contacts", [])

    # Client-side filtering for mentors
    mentors = []
    for contact in contacts:
        # Check if contact is a mentor
        participations = contact.get("participation", [])
        is_mentor = any(
            p.get("capacity") == "Mentor"
            for p in (participations if isinstance(participations, list) else [participations])
        )
        if not is_mentor:
            continue

        # Filter by expertise (search in bio and name)
        if expertise:
            expertise_lower = expertise.lower()
            bio = (contact.get("bio") or "").lower()
            name = (contact.get("fullName") or "").lower()
            if expertise_lower not in bio and expertise_lower not in name:
                continue

        # Filter by cohort
        if cohort_id:
            cohort_id_lower = cohort_id.lower()
            has_cohort = False
            for p in (participations if isinstance(participations, list) else [participations]):
                cohorts = p.get("cohorts", [])
                if isinstance(cohorts, dict):
                    cohorts = [cohorts]
                for c in cohorts:
                    if cohort_id_lower in (c.get("shortName") or "").lower():
                        has_cohort = True
                        break
            if not has_cohort:
                continue

        mentors.append(contact)

    # Apply limit
    mentors = mentors[:limit]

    def extract_cohorts(participations):
        """Extract cohort names from participation data."""
        cohort_names = []
        if not participations:
            return cohort_names
        if isinstance(participations, dict):
            participations = [participations]
        for p in participations:
            cohorts = p.get("cohorts", [])
            if isinstance(cohorts, dict):
                cohorts = [cohorts]
            for c in cohorts:
                if c.get("shortName"):
                    cohort_names.append(c.get("shortName"))
        return cohort_names

    formatted = []
    for contact in mentors:
        formatted.append({
            "id": contact.get("id"),
            "name": contact.get("fullName"),
            "email": contact.get("email"),
            "bio": contact.get("bio"),
            "linkedin": contact.get("linkedIn"),
            "cohorts": extract_cohorts(contact.get("participation")),
        })

    return sanitize_for_json({
        "mentors": formatted,
        "count": len(formatted),
        "source": "mentor_hub_live",
    })


@tool(args_schema=TasksInput)
async def get_mentor_hub_tasks(
    team_id: Optional[str] = None,
    assignee_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Get action items/tasks from Mentor Hub.

    Use this to:
    - Get a team's current tasks and blockers
    - Find overdue or blocked tasks
    - Check task status for follow-ups
    - See what a team is working on

    This provides LIVE data from the application.

    Args:
        team_id: Filter by team (name or ID)
        assignee_id: Filter by assignee
        status: Filter by status (Not Started, In Progress, Completed, Blocked)
        limit: Maximum tasks to return

    Returns:
        List of tasks with assignees, due dates, and status
    """
    logger.info(f"Fetching Mentor Hub tasks: team={team_id}")

    # Build BaseQL filter
    filter_clause = ""
    if status:
        filter_clause = f'_filter: {{status: {{_eq: "{status}"}}}}'

    # BaseQL schema: tasks table
    query = f"""
    query GetTasks {{
        tasks({filter_clause}, _order_by: {{due: "asc"}}) {{
            id
            taskId
            name
            description
            status
            priority
            levelOfEffort
            due
            source
            assignedTo {{
                id
                fullName
            }}
            team {{
                id
                teamName
            }}
            session {{
                id
                sessionType
                scheduledStart
            }}
        }}
    }}
    """

    result = await _mentor_hub_graphql(query)

    if "error" in result:
        return result

    tasks = result.get("tasks", [])

    # Client-side filtering for team and assignee
    if team_id:
        team_id_lower = team_id.lower()
        filtered_tasks = []
        for task in tasks:
            team_data = task.get("team")
            if team_data:
                if isinstance(team_data, list):
                    team_names = [t.get("teamName", "") for t in team_data]
                else:
                    team_names = [team_data.get("teamName", "")]
                if any(team_id_lower in name.lower() for name in team_names):
                    filtered_tasks.append(task)
        tasks = filtered_tasks

    if assignee_id:
        assignee_lower = assignee_id.lower()
        filtered_tasks = []
        for task in tasks:
            assignee_data = task.get("assignedTo")
            if assignee_data:
                if isinstance(assignee_data, list):
                    names = [a.get("fullName", "") for a in assignee_data]
                else:
                    names = [assignee_data.get("fullName", "")]
                if any(assignee_lower in name.lower() for name in names):
                    filtered_tasks.append(task)
        tasks = filtered_tasks

    # Apply limit
    tasks = tasks[:limit]

    def extract_name(data, field: str):
        """Extract field from data which may be dict or list."""
        if isinstance(data, list) and data:
            return data[0].get(field)
        elif isinstance(data, dict):
            return data.get(field)
        return None

    formatted = []
    for task in tasks:
        formatted.append({
            "id": task.get("id"),
            "task_id": task.get("taskId"),
            "title": task.get("name"),  # Field is 'name' in BaseQL schema
            "description": task.get("description"),
            "status": task.get("status"),
            "priority": task.get("priority"),
            "level_of_effort": task.get("levelOfEffort"),
            "due_date": task.get("due"),  # Field is 'due' in BaseQL schema
            "source": task.get("source"),
            "assignee": extract_name(task.get("assignedTo"), "fullName"),
            "team": extract_name(task.get("team"), "teamName"),
            "session_type": extract_name(task.get("session"), "sessionType") if task.get("session") else None,
        })

    return sanitize_for_json({
        "tasks": formatted,
        "count": len(formatted),
        "source": "mentor_hub_live",
    })


@tool
async def get_mentor_hub_user_context(email: str) -> dict[str, Any]:
    """
    Get context about a user from Mentor Hub.

    Use this when you need to understand:
    - What teams a user is part of
    - Their role (student, mentor, staff)
    - Their participation in cohorts
    - Their upcoming sessions

    Args:
        email: The user's email address

    Returns:
        User context including teams, role, and participation
    """
    logger.info(f"Fetching Mentor Hub user context: {email}")

    # BaseQL uses _filter syntax
    query = f"""
    query GetUserContext {{
        contacts(_filter: {{email: {{_eq: "{email}"}}}}) {{
            id
            fullName
            email
            type
            members {{
                team {{
                    teamName
                }}
                type
                status
            }}
            participation {{
                capacity
                status
                cohorts {{
                    shortName
                }}
            }}
        }}
    }}
    """

    result = await _mentor_hub_graphql(query)

    if "error" in result:
        return result

    contacts = result.get("contacts", [])
    if not contacts:
        return {"error": f"User with email '{email}' not found"}

    contact = contacts[0]

    def extract_team_names(members):
        """Extract team names from members data."""
        team_names = []
        if not members:
            return team_names
        if isinstance(members, dict):
            members = [members]
        for m in members:
            team_data = m.get("team")
            if team_data:
                if isinstance(team_data, list) and team_data:
                    team_names.append(team_data[0].get("teamName"))
                elif isinstance(team_data, dict):
                    team_names.append(team_data.get("teamName"))
        return [t for t in team_names if t]

    def extract_participations(participations):
        """Extract participation info."""
        result = []
        if not participations:
            return result
        if isinstance(participations, dict):
            participations = [participations]
        for p in participations:
            cohort_name = None
            cohorts = p.get("cohorts")
            if cohorts:
                if isinstance(cohorts, list) and cohorts:
                    cohort_name = cohorts[0].get("shortName")
                elif isinstance(cohorts, dict):
                    cohort_name = cohorts.get("shortName")
            result.append({
                "cohort": cohort_name,
                "capacity": p.get("capacity"),
                "status": p.get("status"),
            })
        return result

    return sanitize_for_json({
        "id": contact.get("id"),
        "name": contact.get("fullName"),
        "email": contact.get("email"),
        "type": contact.get("type"),
        "teams": extract_team_names(contact.get("members")),
        "participations": extract_participations(contact.get("participation")),
        "source": "mentor_hub_live",
    })


# =============================================================================
# TOOL COLLECTION
# =============================================================================

def get_mentor_hub_tools() -> list:
    """Get all Mentor Hub tools for registration with the orchestrator."""
    return [
        get_mentor_hub_sessions,
        get_mentor_hub_team,
        search_mentor_hub_mentors,
        get_mentor_hub_tasks,
        get_mentor_hub_user_context,
    ]
