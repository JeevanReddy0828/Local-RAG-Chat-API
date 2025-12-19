"""
Session memory management module.

Handles:
- Conversation history per session
- Active file tracking per session
- Memory clearing

Note: This is an in-memory implementation. For production,
consider using Redis or a database for persistence.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class SessionState:
    """Complete state for a session."""
    history: List[ConversationTurn] = field(default_factory=list)
    active_file: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STORAGE
# ═══════════════════════════════════════════════════════════════════════════════


# In-memory session storage
# For production: Replace with Redis or database
_sessions: Dict[str, SessionState] = {}


def _get_or_create_session(session_id: str) -> SessionState:
    """Get existing session or create new one."""
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
        logger.debug(f"Created new session: {session_id}")
    return _sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════


def append_turn(session_id: str, role: str, content: str) -> None:
    """
    Add a conversation turn to session history.
    
    Args:
        session_id: Unique session identifier
        role: "user" or "assistant"
        content: Message content
    """
    session = _get_or_create_session(session_id)
    session.history.append(ConversationTurn(role=role, content=content))
    logger.debug(f"Session {session_id}: Added {role} turn ({len(content)} chars)")


def format_history(session_id: str, max_turns: int = 8) -> str:
    """
    Format conversation history as a string for prompt construction.
    
    Args:
        session_id: Unique session identifier
        max_turns: Maximum number of turns to include (each turn = user + assistant)
        
    Returns:
        Formatted conversation history string
    """
    session = _get_or_create_session(session_id)
    
    # Get last N*2 messages (N turns = N user + N assistant messages)
    recent = session.history[-(max_turns * 2):]
    
    if not recent:
        return ""
    
    lines = []
    for turn in recent:
        role_label = turn.role.capitalize()
        lines.append(f"{role_label}: {turn.content}")
    
    return "\n".join(lines)


def get_history_length(session_id: str) -> int:
    """
    Get the number of turns in session history.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Number of conversation turns
    """
    session = _get_or_create_session(session_id)
    return len(session.history)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVE FILE TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


def set_active_file(session_id: str, filename: str) -> None:
    """
    Set the active document for a session.
    
    The active file is used for document-scoped retrieval.
    
    Args:
        session_id: Unique session identifier
        filename: Name of the uploaded file
    """
    session = _get_or_create_session(session_id)
    session.active_file = filename
    logger.info(f"Session {session_id}: Active file set to '{filename}'")


def get_active_file(session_id: str) -> Optional[str]:
    """
    Get the active document for a session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Filename of active document, or None if not set
    """
    session = _get_or_create_session(session_id)
    return session.active_file


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def clear_session(session_id: str) -> bool:
    """
    Clear all data for a session (history + active file).
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        True if session existed and was cleared, False otherwise
    """
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session {session_id}: Cleared")
        return True
    return False


def clear_history(session_id: str) -> None:
    """
    Clear only conversation history (keep active file).
    
    Args:
        session_id: Unique session identifier
    """
    session = _get_or_create_session(session_id)
    session.history.clear()
    logger.debug(f"Session {session_id}: History cleared")


def get_all_session_ids() -> List[str]:
    """
    Get list of all active session IDs.
    
    Returns:
        List of session identifiers
    """
    return list(_sessions.keys())


def get_session_count() -> int:
    """
    Get total number of active sessions.
    
    Returns:
        Number of sessions in memory
    """
    return len(_sessions)
