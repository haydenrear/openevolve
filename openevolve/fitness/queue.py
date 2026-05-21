"""Controller-owned pending scoring queue."""

from __future__ import annotations


class PendingScoringQueue:
    def __init__(self):
        self._program_ids: list[str] = []
        self._seen: set[str] = set()

    def add(self, program_id: str) -> None:
        if program_id not in self._seen:
            self._program_ids.append(program_id)
            self._seen.add(program_id)

    def remove_many(self, program_ids: set[str]) -> None:
        self._program_ids = [
            program_id for program_id in self._program_ids if program_id not in program_ids
        ]
        self._seen -= program_ids

    def as_list(self) -> list[str]:
        return list(self._program_ids)

    def __len__(self) -> int:
        return len(self._program_ids)
