from typing import Dict, List, Optional

# session_id -> list of {"role": "user"/"assistant", "content": "..."}
_MEM: Dict[str, List[dict]] = {}

# session_id -> active filename (most recently uploaded file for that session)
_ACTIVE_FILE: Dict[str, Optional[str]] = {}


def append_turn(session_id: str, role: str, content: str) -> None:
    _MEM.setdefault(session_id, []).append({"role": role, "content": content})


def set_active_file(session_id: str, filename: str) -> None:
    _ACTIVE_FILE[session_id] = filename


def get_active_file(session_id: str) -> Optional[str]:
    return _ACTIVE_FILE.get(session_id)


def clear_session(session_id: str) -> None:
    _MEM.pop(session_id, None)
    _ACTIVE_FILE.pop(session_id, None)


def format_history(session_id: str, max_turns: int = 8) -> str:
    turns = _MEM.get(session_id, [])[-max_turns * 2 :]
    out = []
    for t in turns:
        out.append(f"{t['role'].capitalize()}: {t['content']}")
    return "\n".join(out).strip()
