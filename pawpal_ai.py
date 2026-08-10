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

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from pawpal_system import Owner, Pet, Priority, Scheduler, Task
from retriever import Chunk, KnowledgeBase, build_query

load_dotenv()

logger = logging.getLogger(__name__)

LOG_FILE = Path(__file__).parent / "pawpal.log"

# --- model backends --------------------------------------------------------
# The RAG pipeline is provider-agnostic: retrieval, prompt assembly and every
# guardrail below are identical whichever backend generates the tasks. Only the
# one network call differs, so the app runs either against the Anthropic API or
# against a model running locally through Ollama at no cost.
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = 300  # local generation is slower than an API call

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
- Some sources cover several species at once. Only use the parts that apply to \
this pet's species, and ignore the rest. Dogs are walked and need potty \
breaks; cats are not walked and use a litter box instead.
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

Scheduling the day:
- Give each task a specific time in "HH:MM" 24-hour form, inside the owner's \
available window. Spread the routine across the day instead of stacking \
everything at the start.
- Put tasks at the time of day they belong to. Morning meals early, evening \
meals late, and space two daily meals roughly 12 hours apart.
- Medication goes immediately after the meal it accompanies.
- Toileting (walks for dogs, litter scooping for cats) belongs early in the \
day and again near the end of it.
- Grooming, brushing and play are flexible: leave time as an empty string for \
these so the scheduler can slot them into whatever gaps remain.
- Never schedule anything that would finish after the owner's window ends.

Titles are short and concrete: "Morning walk", "Evening meal", "Scoop litter \
box"."""


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
    time: str = Field(
        description=(
            "The time of day to do this, as 24-hour 'HH:MM' inside the owner's "
            "window, e.g. '07:30' for a morning meal or '18:00' for an evening "
            "walk. Required for meals, medication, walks and litter scooping. "
            "Use an empty string ONLY for genuinely flexible tasks such as "
            "brushing, nail trims or play."
        )
    )
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
    provider: str = ""  # which backend produced this, for the UI and logs
    model: str = ""


# Words that place a task in the day, mapped to a fraction of the owner's
# availability window. Used only as a fallback: small local models frequently
# leave the time field blank, which would otherwise pack an "Evening walk" into
# the 7am slot alongside breakfast.
TIME_OF_DAY_HINTS: tuple[tuple[str, float], ...] = (
    ("bedtime", 0.95),
    ("night", 0.92),
    ("evening", 0.85),
    ("supper", 0.85),
    ("dinner", 0.85),
    ("afternoon", 0.60),
    ("midday", 0.45),
    ("lunch", 0.45),
    ("noon", 0.45),
    ("breakfast", 0.10),
    ("morning", 0.10),
)


def infer_time_of_day(title: str, category: str, owner: Owner) -> str:
    """Derive an 'HH:MM' from time-of-day words in a task's own name.

    A task called "Evening walk" states when it belongs; if the model did not
    fill in the time field, that word is still a reliable signal. Positions the
    task at a fraction of the owner's availability window and rounds to the
    nearest 5 minutes.

    Args:
        title: the task title.
        category: the task category, also searched for hints.
        owner: supplies the availability window.

    Returns:
        An "HH:MM" string, or "" when there is no hint or no window (in which
        case the scheduler simply slots the task into any free gap).
    """
    if not (owner.day_start and owner.day_end):
        return ""
    start = Scheduler.parse_time(owner.day_start)
    end = Scheduler.parse_time(owner.day_end)
    if end <= start:
        return ""

    text = f"{title} {category}".lower()
    for keyword, fraction in TIME_OF_DAY_HINTS:
        if keyword in text:
            minutes = start + int((end - start) * fraction)
            minutes = min(minutes - minutes % 5, end)
            return f"{minutes // 60:02d}:{minutes % 60:02d}"
    return ""


def resolve_provider() -> str:
    """Decide which model backend to use.

    Order of precedence:
      1. PAWPAL_PROVIDER in the environment, if it names a known backend.
      2. Anthropic, if an API key is present.
      3. Ollama — the free local fallback, so the app works with no key at all.

    Returns:
        Either "anthropic" or "ollama".
    """
    explicit = os.getenv("PAWPAL_PROVIDER", "").strip().lower()
    if explicit in (PROVIDER_ANTHROPIC, PROVIDER_OLLAMA):
        return explicit
    if explicit:
        logger.warning("Unknown PAWPAL_PROVIDER %r — falling back to auto-detect", explicit)
    return PROVIDER_ANTHROPIC if has_usable_api_key() else PROVIDER_OLLAMA


def has_usable_api_key() -> bool:
    """Return True only for a key that could plausibly be real.

    The unedited .env.example ships the placeholder "sk-ant-...", which is
    non-empty and would otherwise route to the paid backend and fail with a
    confusing 401. A real key is far longer, so a length check separates the
    two and lets the free local backend take over instead.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip("\"'")
    return key.startswith("sk-ant-") and len(key) > 30


def resolve_model(provider: str) -> str:
    """Return the model id for a backend, honouring PAWPAL_MODEL if set."""
    override = os.getenv("PAWPAL_MODEL", "").strip()
    if override:
        return override
    return DEFAULT_OLLAMA_MODEL if provider == PROVIDER_OLLAMA else DEFAULT_ANTHROPIC_MODEL


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
        f"The owner has {owner.available_minutes} minutes of hands-on time today"
        + (f", shared across {len(owner.pets)} pets ({', '.join(other_pets)} as well)."
           if other_pets else " for this pet.")
    )
    window_line = (
        f"They are available between {owner.day_start} and {owner.day_end}. "
        f"Every task must start at or after {owner.day_start} and finish by {owner.day_end}."
        if owner.day_start and owner.day_end
        else "They have not given a specific availability window."
    )

    return (
        "Pet profile:\n"
        + "\n".join(f"- {t}" for t in traits)
        + f"\n\n{budget_line}\n{window_line}\n\n"
        "Retrieved care guidance:\n\n"
        + format_sources(chunks)
        + "\n\nPropose the daily care tasks for this pet, grounded in the sources above. "
        "Do not repeat a task the owner has already added."
    )


def _validate(
    proposals: list[ProposedTask],
    allowed_ids: dict[str, Chunk],
    existing_titles: set[str],
    owner: Owner | None = None,
) -> tuple[list[SuggestedTask], list[str]]:
    """Filter model proposals down to safe, grounded, non-duplicate tasks.

    This is the guardrail layer. Nothing reaches the domain model without
    passing every check here, and each rejection is recorded with a reason so
    it can be shown in the UI and asserted on in the eval suite.

    Checks applied, in order:
      1. Title is present and not absurdly long.
      2. Duration is an integer within MIN_DURATION..MAX_DURATION.
      3. Priority and frequency are known values.
      4. time is "HH:MM" or empty (a bad time is cleared, not rejected); if
         empty, a time-of-day word in the title is used to infer one so a
         model that skips the field doesn't bunch "evening" tasks into the
         first free morning slot.
      5. source_id matches a chunk we actually retrieved  <- the grounding check.
      6. The task is not a duplicate of an existing or already-accepted task.
      7. No more than MAX_TASKS survive.

    Args:
        proposals: raw proposals from the model.
        allowed_ids: chunk id -> Chunk, the sources that were actually retrieved.
        existing_titles: lowercased titles the pet already has.
        owner: supplies the availability window for the time-inference
            fallback; omit to skip inference (times are left blank).

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

        # Small local models frequently leave time blank even when told not
        # to. Fall back to the word in the task's own name/category — "Evening
        # walk" says when it belongs even if the structured field doesn't.
        if not time_value and owner is not None:
            inferred = infer_time_of_day(proposal.title, proposal.category, owner)
            if inferred:
                logger.info("Inferred time %s for %r from its name", inferred, label)
                time_value = inferred

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


def _generate_anthropic(
    system: str, user: str, model: str, client: anthropic.Anthropic | None
) -> tuple[CarePlanSuggestion, int, int]:
    """Generate a care plan with the Anthropic API.

    Returns:
        (parsed suggestion, input tokens, output tokens).

    Raises:
        PawPalAIError: with a user-safe message on any API failure.
    """
    if client is None:
        if not has_usable_api_key():
            raise PawPalAIError(
                "No usable ANTHROPIC_API_KEY found. Either add a real key to .env, or set "
                "PAWPAL_PROVIDER=ollama to run a free local model instead."
            )
        client = anthropic.Anthropic()

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": user}],
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
        logger.exception("Anthropic API error")
        raise PawPalAIError(f"The API returned an error ({exc.status_code}). Try again.") from exc

    # A safety refusal is a normal 200 response with empty content, so check it
    # before touching parsed_output.
    if response.stop_reason == "refusal":
        raise PawPalAIError("The model declined this request. Try rephrasing the pet details.")

    if response.parsed_output is None:
        logger.error("No parsed output; stop_reason=%s", response.stop_reason)
        raise PawPalAIError("The model returned an unreadable response. Try again.")

    return response.parsed_output, response.usage.input_tokens, response.usage.output_tokens


def _generate_ollama(system: str, user: str, model: str) -> tuple[CarePlanSuggestion, int, int]:
    """Generate a care plan with a local model served by Ollama.

    Uses Ollama's JSON-schema structured output so the reply is constrained to
    the same CarePlanSuggestion shape the Anthropic path produces, which is what
    lets the guardrails downstream stay completely provider-agnostic. Talks HTTP
    with the standard library so no extra package is needed.

    Returns:
        (parsed suggestion, prompt tokens, generated tokens).

    Raises:
        PawPalAIError: with a user-safe message if Ollama is unreachable, the
            model is missing, or the reply is not valid against the schema.
    """
    payload = {
        "model": model,
        "stream": False,
        "format": CarePlanSuggestion.model_json_schema(),
        "options": {"temperature": 0},  # deterministic-ish, for reproducibility
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        if exc.code == 404:
            raise PawPalAIError(
                f"Ollama does not have the model '{model}'. Run:  ollama pull {model}"
            ) from exc
        raise PawPalAIError(f"Ollama returned an error ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise PawPalAIError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Start it with:  ollama serve"
        ) from exc
    except TimeoutError as exc:
        raise PawPalAIError(
            f"The local model took longer than {OLLAMA_TIMEOUT}s. Try a smaller model."
        ) from exc

    content = body.get("message", {}).get("content", "")
    try:
        parsed = CarePlanSuggestion.model_validate_json(content)
    except ValidationError as exc:
        # Small local models occasionally drift from the schema. Treat that as a
        # normal failure with a clear message rather than a crash.
        logger.error("Ollama reply failed schema validation: %s", exc)
        raise PawPalAIError(
            "The local model returned a response that didn't match the expected "
            "format. Try again, or use a larger model."
        ) from exc

    # Ollama reports token counts under different names than the Anthropic SDK.
    return parsed, body.get("prompt_eval_count", 0), body.get("eval_count", 0)


def suggest_tasks(
    pet: Pet,
    owner: Owner,
    kb: KnowledgeBase | None = None,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
    top_k: int = 4,
    provider: str | None = None,
) -> SuggestionResult:
    """Retrieve care guidance for one pet and propose validated tasks from it.

    Works against either backend. Retrieval, prompt assembly and every guardrail
    are identical for both; only the generation call differs.

    Args:
        pet: the pet to plan for.
        owner: the pet's owner, for the time budget.
        kb: knowledge base to retrieve from (loaded from disk if omitted).
        client: Anthropic client; injected by tests, else built from the env.
        model: model id; defaults to the backend's default or PAWPAL_MODEL.
        top_k: how many knowledge chunks to retrieve.
        provider: "anthropic" or "ollama"; auto-detected when omitted.

    Returns:
        A SuggestionResult holding accepted tasks, the sources used, the
        model's coverage note, rejections, and token usage.

    Raises:
        PawPalAIError: on any backend or configuration failure, with a message
            that is safe to show the user.
    """
    kb = kb or KnowledgeBase.load()

    # A caller that injects a client is explicitly asking for the Anthropic
    # path (this is what the test suite does), so don't auto-detect past it.
    provider = provider or (PROVIDER_ANTHROPIC if client is not None else resolve_provider())
    model = model or resolve_model(provider)

    query = build_query(pet, owner)
    chunks = kb.search(query, top_k=top_k)
    logger.info(
        "Suggesting tasks for %s (%s, age %s) via %s/%s using sources %s",
        pet.name, pet.species, pet.age, provider, model, [c.id for c in chunks],
    )

    if not chunks:
        return SuggestionResult(
            provider=provider,
            model=model,
            coverage_note=(
                "No care guidance matched this pet, so no tasks were suggested. "
                "Add a species and breed, or extend the knowledge base."
            ),
        )

    system = SYSTEM_PROMPT
    user = build_user_prompt(pet, owner, chunks)

    if provider == PROVIDER_OLLAMA:
        parsed, in_tokens, out_tokens = _generate_ollama(system, user, model)
    else:
        parsed, in_tokens, out_tokens = _generate_anthropic(system, user, model, client)

    accepted, rejected = _validate(
        proposals=parsed.tasks,
        allowed_ids={c.id: c for c in chunks},
        existing_titles={t.title.strip().lower() for t in pet.tasks},
        owner=owner,
    )

    logger.info(
        "Pet %s: %d proposed -> %d accepted, %d rejected (tokens in=%d out=%d)",
        pet.name, len(parsed.tasks), len(accepted), len(rejected), in_tokens, out_tokens,
    )
    for reason in rejected:
        logger.warning("Rejected: %s", reason)

    return SuggestionResult(
        suggestions=accepted,
        sources=chunks,
        coverage_note=parsed.coverage_note.strip(),
        rejected=rejected,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        provider=provider,
        model=model,
    )
