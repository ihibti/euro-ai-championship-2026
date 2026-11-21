"""The three championship tasks and the leaderboard server they share."""

from __future__ import annotations

from dataclasses import dataclass

SERVER_URL = "http://34.147.229.55"


@dataclass(frozen=True)
class TaskInfo:
    name: str
    hf_dataset: str
    server_task_id: str | None
    suffixes: tuple[str, ...]
    max_bytes: int | None


CHALLENGE_TASKS: dict[str, TaskInfo] = {
    info.name: info
    for info in (
        TaskInfo(
            name="watermark_localization",
            hf_dataset="SprintML/Hackathon_Watermark_Localisation_Task",
            server_task_id=None,
            suffixes=(".pt",),
            max_bytes=20_000_000,
        ),
        TaskInfo(
            name="model_equality",
            hf_dataset="SprintML/model_equality",
            server_task_id="33-model-equality",
            suffixes=(".csv",),
            max_bytes=None,
        ),
        TaskInfo(
            name="unlearning",
            hf_dataset="SprintML/unlearning",
            server_task_id="31-unlearning",
            suffixes=(".pt", ".pth"),
            max_bytes=450_000_000,
        ),
    )
}


def get_task(name: str) -> TaskInfo:
    try:
        return CHALLENGE_TASKS[name]
    except KeyError:
        valid = ", ".join(sorted(CHALLENGE_TASKS))
        raise ValueError(f"unknown task {name!r}; valid tasks: {valid}") from None
