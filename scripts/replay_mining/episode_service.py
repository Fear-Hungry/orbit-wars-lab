"""Thin client for the Kaggle EpisodeService (real leaderboard replays).

Two endpoints, discovered empirically (2026-06-14) for the `orbit-wars` competition:

  * Episode metadata (outcome / opponents / 2p-vs-4p):
        POST https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes
        body {"submissionId": <int>}  (camelCase, lowercase first letter)
        auth = HTTP basic (kaggle.json username/key)
    Returns {"episodes": [{id, createTime, endTime, state, type,
                           agents: [{submissionId, reward, index, initialScore,
                                     updatedScore, teamId}, ...]}, ...]}
    `reward` is +1 win / -1 loss / 0 from THAT agent's perspective; len(agents)
    is the player count (2 or 4); the agent whose submissionId == ours carries
    our `index` (player slot, 0 when omitted).

  * Full replay (per-step state + actions), served from the CDN, no auth:
        GET https://www.kaggleusercontent.com/episodes/<episodeId>.json
    The classic `/requests/EpisodeService/GetEpisodeReplay` and
    `/api/i/.../GetEpisodeReplay` routes 400/404 for this comp — the CDN path
    is the one that works.

This module is intentionally dependency-light (requests + stdlib) so it can run
under .venv/bin/python without importing kaggle_environments (which pulls in the
heavy OpenSpiel env registry).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

_KAGGLE_JSON = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
_LIST_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
_REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{episode_id}.json"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def _auth() -> tuple[str, str]:
    cred = json.loads(_KAGGLE_JSON.read_text())
    return cred["username"], cred["key"]


def list_episodes(submission_id: int, *, retries: int = 4) -> list[dict[str, Any]]:
    """All episodes a submission participated in (metadata only, one cheap call)."""
    auth = _auth()
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(
                _LIST_URL,
                json={"submissionId": int(submission_id)},
                headers=_HEADERS,
                auth=auth,
                timeout=90,
            )
            if r.status_code == 200:
                return r.json().get("episodes", [])
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as exc:  # pragma: no cover - network
            last = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ListEpisodes({submission_id}) failed: {last}")


def download_replay(episode_id: int, dest: Path | None = None, *, retries: int = 4) -> dict[str, Any]:
    """Full replay JSON for one episode; optionally cached to `dest`."""
    if dest is not None and dest.exists() and dest.stat().st_size > 0:
        return json.loads(dest.read_text())
    url = _REPLAY_URL.format(episode_id=int(episode_id))
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
            if r.status_code == 200 and r.content:
                data = r.json()
                if dest is not None:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(data))
                return data
            last = f"HTTP {r.status_code} ({len(r.content)} bytes)"
        except (requests.RequestException, json.JSONDecodeError) as exc:  # pragma: no cover
            last = str(exc)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download_replay({episode_id}) failed: {last}")


def episode_outcome(episode: dict[str, Any], submission_id: int) -> dict[str, Any] | None:
    """Reduce a ListEpisodes record to our perspective.

    Returns {episode_id, n_players, our_index, our_reward, our_updated_score,
             opponents:[{submission_id, reward, score, team_id}], created} or
    None if our submission is not in this episode.
    """
    agents = episode.get("agents", [])
    mine = [a for a in agents if a.get("submissionId") == submission_id]
    if not mine:
        return None
    me = mine[0]
    opponents = [
        {
            "submission_id": a.get("submissionId"),
            "reward": a.get("reward"),
            "score": a.get("updatedScore"),
            "team_id": a.get("teamId"),
        }
        for a in agents
        if a.get("submissionId") != submission_id
    ]
    return {
        "episode_id": episode.get("id"),
        "n_players": len(agents),
        "our_index": me.get("index", 0),
        "our_reward": me.get("reward"),
        "our_updated_score": me.get("updatedScore"),
        "opponents": opponents,
        "created": episode.get("createTime"),
        "state": episode.get("state"),
    }
