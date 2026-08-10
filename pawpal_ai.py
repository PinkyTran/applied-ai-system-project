"""RAG layer: turn retrieved care guidance into validated PawPal+ tasks.

Pipeline for one pet:

    build_query(pet, owner)      # what do we need to know?
    KnowledgeBase.search(...)    # retrieve the relevant care chunks
    Claude + structured output   # propose tasks grounded in those chunks
    _validate(...)               # reject anything unsafe or ungrounded
    -> Task objects              # fed straight into the existing Scheduler

The retrieved text is the only source of care guidance the model is given, and
every proposed task must cite the chunk it came from. A task citing a chunk we
did not retrieve is dropped, so the retrieval genuinely drives the output
instead of decorating an answer the model already had.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pawpal_system import Owner, Pet, Priority, Task
from retriever import Chunk, KnowledgeBase, build_query

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("PAWPAL_MODEL", "claude-opus-5")
LOG_FILE = Path(__file__).parent / "pawpal.log"

# --- guardrail limits ------------------------------------------------------
MAX_TASKS = 8  # a day's plan longer than this is unusable, not helpful
MIN_DURATION = 1
MAX_DURATION = 240  # 4 hours; anything longer is a model error, not a chore
MAX_TITLE_LEN = 60
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

PRIORITY_ENUM = {"high": Priority.HIGH, "medium": Priority.MEDIUM, "low": Priority.LOW}

SYSTEM_PROMPT = """\
You are the care-planning assistant inside PawPal+, an app that builds a pet \
owner's daily care schedule.

You will be given a pet profile, the owner's daily time budget, and a set of \
numbered care guidance sources retrieved from PawPal+'s knowledge base.

Rules:
- Base every task on the retrieved sources. Do not use pet-care knowledge that \
is not in the sources, even if you are confident it is correct.
- Every task must set source_id to the id of the source that justifies it. \
Use the id exactly as written in the source header.
- Take durations and frequencies from the sources. If a source gives a range, \
choose a value in that range that suits this specific pet.
- Respect limits stated in the sources. If a source caps a walk at 20 minutes \
for this kind of pet, never propose a longer one.
- Order matters: propose the highest-priority tasks first (medication, then \
feeding, then toileting, then enrichment, then grooming).
- Propose only tasks for a normal day. Do not propose vet appointments, \
one-off purchases, or anything the owner cannot do at home today.
- Keep the total roughly within the owner's time budget. If everything cannot \
fit, propose the essentials and say what you left out in coverage_note.
- If the sources do not cover this species, propose no tasks and explain that \
in coverage_note. Do not guess.

Titles are short and concrete: "Morning walk", "Evening meal", "Scoop litter \
box". Set time only when the sources imply a specific time of day, otherwise \
leave it as an empty string."""


class PawPalAIError(RuntimeError):
    """Raised when a suggestion cannot be produced. Carries a user-safe message."""


# --- structured output schema ---------------------------------------------
# Deliberately loose (plain int, no bounds): bounds are enforced in _validate()
# so a bad value becomes a logged, user-visible rejection rather than an
# exception, and the guardrails stay testable without calling the API.


class ProposedTask(BaseModel):
    """One care task proposed by the model, before validation."""

    title: str = Field(description="Short, concrete task name, e.g. 'Morning walk'.")
    duration_minutes: int = Field(description="How many minutes the task takes.")
    priority: Literal["high", "medium", "low"]
    frequency: Literal["once", "daily", "weekly"]
    time: str = Field(description="Preferred time as 'HH:MM', or '' for no preference.")
    category: str = Field(description="One of: medication, feeding, exercise, hygiene, enrichment.")
    source_id: str = Field(description="The id of the source this task is based on.")
    rationale: str = Field(description="One sentence citing what the source says.")


class CarePlanSuggestion(BaseModel):
    """The model's full response for one pet."""

    tasks: list[ProposedTask]
    coverage_note: str = Field(
        description="What the sources did not cover, or what was left out for time."
    )


# --- results ---------------------------------------------------------------


@dataclass
class SuggestedTask:
    """A validated Task plus the source that justifies it."""

    task: Task
    source_id: str
    source_title: str
    rationale: str


@dataclass
class SuggestionResult:
    """Everything one suggestion run produced, including what was thrown away."""

    suggestions: list[SuggestedTask] = field(default_factory=list)
    sources: list[Chunk] = field(default_factory=list)
    coverage_note: str = ""
    rejected: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging to pawpal.log plus the console, once per process.

    Called by the Streamlit app and the eval script. Safe to call repeatedly:
    handlers are only attached the first time.

    Args:
        level: the root log level to apply.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    # The SDK logs every HTTP request at DEBUG; keep that out of our log file.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def format_sources(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as the source block shown to the model.

    Each source is labelled with the exact id the model must cite back, which
    is what makes the grounding check in _validate() possible.

    Args:
        chunks: the retrieved knowledge chunks.

    Returns:
        A single string of delimited sources.
    """
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"<source id=\"{chunk.id}\" title=\"{chunk.title}\">\n{chunk.text}\n</source>"
        )
    return "\n\n".join(blocks)


def build_user_prompt(pet: Pet, owner: Owner, chunks: list[Chunk]) -> str:
    """Assemble the user turn: the pet profile, the budget, and the sources."""
    traits = [f"Name: {pet.name}", f"Species: {pet.species or 'unknown'}"]
    if pet.breed:
        traits.append(f"Breed: {pet.breed}")
    if pet.age is not None:
        traits.append(f"Age: {pet.age} years")
    if pet.tasks:
        existing = ", ".join(t.title for t in pet.tasks)
        traits.append(f"Tasks the owner has already added: {existing}")

    other_pets = [p.name for p in owner.pets if p is not pet]
    budget_line = (
        f"The owner has {owner.available_minutes} minutes available today"
        + (f", shared across {len(owner.pets)} pets ({', '.join(other_pets)} as well)."
           if other_pets else " for this pet.")
    )

    return (
        "Pet profile:\n"
        + "\n".join(f"- {t}" for t in traits)
        + f"\n\n{budget_line}\n\n"
        "Retrieved care guidance:\n\n"
        + format_sources(chunks)
        + "\n\nPropose the daily care tasks for this pet, grounded in the sources above. "
        "Do not repeat a task the owner has already added."
    )


def _validate(
    proposals: list[ProposedTask],
    allowed_ids: dict[str, Chunk],
    existing_titles: set[str],
) -> tuple[list[SuggestedTask], list[str]]:
    """Filter model proposals down to safe, grounded, non-duplicate tasks.

    This is the guardrail layer. Nothing reaches the domain model without
    passing every check here, and each rejection is recorded with a reason so
    it can be shown in the UI and asserted on in the eval suite.

    Checks applied, in order:
      1. Title is present and not absurdly long.
      2. Duration is an integer within MIN_DURATION..MAX_DURATION.
      3. Priority and frequency are known values.
      4. time is "HH:MM" or empty (a bad time is cleared, not rejected).
      5. source_id matches a chunk we actually retrieved  <- the grounding check.
      6. The task is not a duplicate of an existing or already-accepted task.
      7. No more than MAX_TASKS survive.

    Args:
        proposals: raw proposals from the model.
        allowed_ids: chunk id -> Chunk, the sources that were actually retrieved.
        existing_titles: lowercased titles the pet already has.

    Returns:
        (accepted suggestions, human-readable rejection reasons).
    """
    accepted: list[SuggestedTask] = []
    rejected: list[str] = []
    seen = set(existing_titles)

    for proposal in proposals:
        label = proposal.title.strip() or "(untitled)"

        if not proposal.title.strip():
            rejected.append("A task with no title was dropped.")
            continue
        if len(proposal.title) > MAX_TITLE_LEN:
            rejected.append(f"'{label[:40]}…' was dropped: title too long.")
            continue

        duration = proposal.duration_minutes
        if not isinstance(duration, int) or not MIN_DURATION <= duration <= MAX_DURATION:
            rejected.append(
                f"'{label}' was dropped: {duration} min is outside the allowed "
                f"{MIN_DURATION}–{MAX_DURATION} min range."
            )
            continue

        if proposal.priority not in PRIORITY_ENUM:
            rejected.append(f"'{label}' was dropped: unknown priority '{proposal.priority}'.")
            continue
        if proposal.frequency not in {"once", "daily", "weekly"}:
            rejected.append(f"'{label}' was dropped: unknown frequency '{proposal.frequency}'.")
            continue

        # The grounding check: a citation we never retrieved means the model
        # went outside its sources, so the task is not trustworthy.
        source = allowed_ids.get(proposal.source_id.strip())
        if source is None:
            rejected.append(
                f"'{label}' was dropped: cites unknown source "
                f"'{proposal.source_id}' that was not retrieved."
            )
            logger.warning("Ungrounded task %r cited source %r", label, proposal.source_id)
            continue

        key = proposal.title.strip().lower()
        if key in seen:
            rejected.append(f"'{label}' was dropped: duplicate of an existing task.")
            continue

        if len(accepted) >= MAX_TASKS:
            rejected.append(f"'{label}' was dropped: more than {MAX_TASKS} tasks proposed.")
            continue

        # A malformed time is not worth discarding a good task over — clear it
        # and let the scheduler place the task by priority instead.
        time_value = proposal.time.strip()
        if time_value and not TIME_RE.match(time_value):
            logger.info("Clearing malformed time %r on task %r", time_value, label)
            time_value = ""

        seen.add(key)
        accepted.append(
            SuggestedTask(
                task=Task(
                    title=proposal.title.strip(),
                    duration_minutes=duration,
                    priority=PRIORITY_ENUM[proposal.priority],
                    category=proposal.category.strip(),
                    time=time_value,
                    frequency=proposal.frequency,
                ),
                source_id=source.id,
                source_title=source.title,
                rationale=proposal.rationale.strip(),
            )
        )

    return accepted, rejected


def suggest_tasks(
    pet: Pet,
    owner: Owner,
    kb: KnowledgeBase | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    top_k: int = 4,
) -> SuggestionResult:
    """Retrieve care guidance for one pet and propose validated tasks from it.

    Args:
        pet: the pet to plan for.
        owner: the pet's owner, for the time budget.
        kb: knowledge base to retrieve from (loaded from disk if omitted).
        client: Anthropic client (created from the environment if omitted).
        model: model id to call.
        top_k: how many knowledge chunks to retrieve.

    Returns:
        A SuggestionResult holding accepted tasks, the sources used, the
        model's coverage note, rejections, and token usage.

    Raises:
        PawPalAIError: on any API or configuration failure, with a message
            that is safe to show the user.
    """
    kb = kb or KnowledgeBase.load()

    query = build_query(pet, owner)
    chunks = kb.search(query, top_k=top_k)
    logger.info(
        "Suggesting tasks for %s (%s, age %s) using sources %s",
        pet.name, pet.species, pet.age, [c.id for c in chunks],
    )

    if not chunks:
        return SuggestionResult(
            coverage_note=(
                "No care guidance matched this pet, so no tasks were suggested. "
                "Add a species and breed, or extend the knowledge base."
            )
        )

    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise PawPalAIError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        client = anthropic.Anthropic()

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(pet, owner, chunks)}],
            output_format=CarePlanSuggestion,
        )
    except anthropic.AuthenticationError as exc:
        raise PawPalAIError("Your API key was rejected. Check ANTHROPIC_API_KEY in .env.") from exc
    except anthropic.RateLimitError as exc:
        raise PawPalAIError("Rate limited by the API. Wait a moment and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise PawPalAIError("Could not reach the API. Check your internet connection.") from exc
    except anthropic.NotFoundError as exc:
        raise PawPalAIError(f"Model '{model}' was not found. Check PAWPAL_MODEL in .env.") from exc
    except anthropic.APIStatusError as exc:
        logger.exception("API error while suggesting tasks for %s", pet.name)
        raise PawPalAIError(f"The API returned an error ({exc.status_code}). Try again.") from exc

    # A safety refusal is a normal 200 response with empty content, so check it
    # before touching parsed_output.
    if response.stop_reason == "refusal":
        logger.warning("Model refused the request for pet %s", pet.name)
        raise PawPalAIError("The model declined this request. Try rephrasing the pet details.")

    parsed = response.parsed_output
    if parsed is None:
        logger.error("No parsed output; stop_reason=%s", response.stop_reason)
        raise PawPalAIError("The model returned an unreadable response. Try again.")

    accepted, rejected = _validate(
        proposals=parsed.tasks,
        allowed_ids={c.id: c for c in chunks},
        existing_titles={t.title.strip().lower() for t in pet.tasks},
    )

    logger.info(
        "Pet %s: %d proposed -> %d accepted, %d rejected (tokens in=%d out=%d)",
        pet.name, len(parsed.tasks), len(accepted), len(rejected),
        response.usage.input_tokens, response.usage.output_tokens,
    )
    for reason in rejected:
        logger.warning("Rejected: %s", reason)

    return SuggestionResult(
        suggestions=accepted,
        sources=chunks,
        coverage_note=parsed.coverage_note.strip(),
        rejected=rejected,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
