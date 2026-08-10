"""Tests for the RAG layer: retrieval, guardrails, and the end-to-end path.

No test here makes a network call. Retrieval is deterministic local code, and
the one test that exercises suggest_tasks() injects a fake client, so the whole
suite runs offline and for free.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pawpal_ai
from pawpal_ai import (
    MAX_DURATION,
    MAX_TASKS,
    CarePlanSuggestion,
    PawPalAIError,
    ProposedTask,
    SuggestionResult,
    _validate,
    build_user_prompt,
    format_sources,
    suggest_tasks,
)
from pawpal_system import Owner, Pet, Priority, Scheduler, Task
from retriever import Chunk, KnowledgeBase, age_band, build_query, tokenize


# --- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    """The real knowledge base, loaded once for the whole module."""
    return KnowledgeBase.load()


@pytest.fixture
def owner() -> Owner:
    return Owner(name="Jordan", available_minutes=200)


def make_proposal(**overrides) -> ProposedTask:
    """Build a valid ProposedTask, overriding individual fields per test."""
    base = {
        "title": "Morning walk",
        "duration_minutes": 30,
        "priority": "high",
        "frequency": "daily",
        "time": "08:00",
        "category": "exercise",
        "source_id": "dog-walking",
        "rationale": "Adult dogs need 30-60 minutes of walking per day.",
    }
    base.update(overrides)
    return ProposedTask(**base)


def allowed(kb: KnowledgeBase, *ids: str) -> dict[str, Chunk]:
    """Return the {id: Chunk} map _validate() expects, for the given ids."""
    by_id = {c.id: c for c in kb.chunks}
    return {i: by_id[i] for i in ids}


# --- tokenizer -------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Dog and a Cat") == ["dog", "cat"]


def test_tokenize_singularises_long_words_only():
    # "walks" -> "walk", but a short word like "gas" keeps its s
    assert "walk" in tokenize("walks")
    assert "cat" in tokenize("cats")
    assert "gas" in tokenize("gas gas")


def test_tokenize_keeps_double_s_words():
    assert "grass" in tokenize("grass")


def test_tokenize_empty_string():
    assert tokenize("") == []


# --- knowledge base loading ------------------------------------------------


def test_knowledge_base_loads_every_markdown_file(kb):
    ids = {c.id for c in kb.chunks}
    assert len(kb.chunks) >= 12
    for expected in ("dog-walking", "cat-litter", "senior-pets", "medication"):
        assert expected in ids


def test_chunks_parse_title_and_tags(kb):
    chunk = next(c for c in kb.chunks if c.id == "senior-pets")
    assert chunk.title == "Caring for senior pets"
    assert "arthritis" in chunk.tags
    # The tags line must not leak into the retrievable body text.
    assert "tags:" not in chunk.text.lower()


def test_missing_knowledge_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.load(tmp_path / "does-not-exist")


def test_empty_knowledge_directory_raises(tmp_path):
    # An empty KB would silently make the model answer from memory, so this
    # must fail loudly rather than return zero chunks.
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.load(tmp_path)


# --- retrieval quality -----------------------------------------------------


def test_cat_query_does_not_retrieve_dog_guidance(kb):
    pet = Pet(name="Mochi", species="cat", age=3)
    ids = [c.id for c in kb.search(build_query(pet), top_k=4)]
    assert not any(i.startswith("dog-") for i in ids)


def test_dog_query_does_not_retrieve_cat_guidance(kb):
    pet = Pet(name="Biscuit", species="dog", age=4)
    ids = [c.id for c in kb.search(build_query(pet), top_k=4)]
    assert not any(i.startswith("cat-") for i in ids)


def test_senior_pet_retrieves_senior_guidance(kb):
    pet = Pet(name="Rex", species="dog", age=11)
    ids = [c.id for c in kb.search(build_query(pet), top_k=4)]
    assert "senior-pets" in ids


def test_puppy_retrieves_puppy_guidance(kb):
    pet = Pet(name="Scout", species="dog", age=0)
    ids = [c.id for c in kb.search(build_query(pet), top_k=4)]
    assert "puppy-care" in ids


def test_flat_faced_breed_retrieves_breed_warning(kb):
    pet = Pet(name="Nugget", species="dog", breed="pug", age=3)
    ids = [c.id for c in kb.search(build_query(pet), top_k=4)]
    assert "flat-faced-breeds" in ids


def test_tight_budget_retrieves_triage_guidance(kb, owner):
    owner.available_minutes = 30
    pet = Pet(name="Mochi", species="cat", age=3)
    ids = [c.id for c in kb.search(build_query(pet, owner), top_k=4)]
    assert "time-budget-triage" in ids


def test_search_is_deterministic(kb):
    pet = Pet(name="Biscuit", species="dog", breed="labrador", age=4)
    query = build_query(pet)
    first = [c.id for c in kb.search(query, top_k=4)]
    second = [c.id for c in kb.search(query, top_k=4)]
    assert first == second


def test_search_respects_top_k(kb):
    assert len(kb.search("dog walk feeding", top_k=2)) == 2


def test_search_on_empty_query_returns_nothing(kb):
    assert kb.search("") == []


def test_search_min_score_filters_irrelevant(kb):
    # A very high threshold should filter everything out rather than returning
    # weak matches the model would then cite as if authoritative.
    assert kb.search("dog walking", top_k=4, min_score=999) == []


@pytest.mark.parametrize(
    "species, age, expected",
    [
        ("dog", 0, "puppy"),
        ("dog", 8, "senior"),
        ("dog", 4, "adult"),
        ("cat", 0, "kitten"),
        ("cat", 12, "senior"),
        ("cat", 8, "adult"),  # 8 is adult for a cat but senior for a dog
    ],
)
def test_age_band_thresholds(species, age, expected):
    assert expected in age_band(species, age)


def test_age_band_unknown_age_is_empty():
    assert age_band("dog", None) == ""


# --- prompt assembly -------------------------------------------------------


def test_format_sources_labels_every_chunk_with_its_id(kb):
    chunks = kb.search("dog walking feeding", top_k=3)
    rendered = format_sources(chunks)
    for chunk in chunks:
        assert f'id="{chunk.id}"' in rendered


def test_user_prompt_includes_profile_budget_and_sources(kb, owner):
    pet = Pet(name="Biscuit", species="dog", breed="labrador", age=4)
    owner.add_pet(pet)
    chunks = kb.search(build_query(pet, owner), top_k=3)
    prompt = build_user_prompt(pet, owner, chunks)
    assert "Biscuit" in prompt
    assert "labrador" in prompt
    assert "200 minutes" in prompt
    assert chunks[0].text[:40] in prompt


def test_user_prompt_lists_existing_tasks_to_avoid_duplicates(kb, owner):
    pet = Pet(name="Biscuit", species="dog", age=4)
    pet.add_task(Task("Evening walk", 30, Priority.HIGH))
    owner.add_pet(pet)
    prompt = build_user_prompt(pet, owner, kb.search(build_query(pet), top_k=2))
    assert "Evening walk" in prompt


# --- guardrails ------------------------------------------------------------


def test_valid_proposal_is_accepted(kb):
    accepted, rejected = _validate([make_proposal()], allowed(kb, "dog-walking"), set())
    assert rejected == []
    assert len(accepted) == 1
    assert accepted[0].task.title == "Morning walk"
    assert accepted[0].task.duration_minutes == 30
    assert accepted[0].task.priority is Priority.HIGH
    assert accepted[0].source_title == "Daily walking for adult dogs"


def test_ungrounded_source_is_rejected(kb):
    # The core RAG guardrail: a citation we never retrieved means the model
    # went outside its sources.
    proposal = make_proposal(source_id="made-up-source")
    accepted, rejected = _validate([proposal], allowed(kb, "dog-walking"), set())
    assert accepted == []
    assert "made-up-source" in rejected[0]


def test_source_retrieved_for_a_different_pet_is_still_rejected(kb):
    # dog-walking is a real chunk, but it was not among THIS pet's sources.
    proposal = make_proposal(source_id="dog-walking")
    accepted, rejected = _validate([proposal], allowed(kb, "cat-litter"), set())
    assert accepted == []
    assert "not retrieved" in rejected[0]


@pytest.mark.parametrize("duration", [0, -5, MAX_DURATION + 1, 10_000])
def test_out_of_range_durations_are_rejected(kb, duration):
    proposal = make_proposal(duration_minutes=duration)
    accepted, rejected = _validate([proposal], allowed(kb, "dog-walking"), set())
    assert accepted == []
    assert "outside the allowed" in rejected[0]


def test_boundary_durations_are_accepted(kb):
    for duration in (1, MAX_DURATION):
        accepted, _ = _validate(
            [make_proposal(duration_minutes=duration)], allowed(kb, "dog-walking"), set()
        )
        assert len(accepted) == 1


def test_empty_title_is_rejected(kb):
    accepted, rejected = _validate([make_proposal(title="   ")], allowed(kb, "dog-walking"), set())
    assert accepted == []
    assert "no title" in rejected[0]


def test_overlong_title_is_rejected(kb):
    accepted, rejected = _validate(
        [make_proposal(title="x" * 200)], allowed(kb, "dog-walking"), set()
    )
    assert accepted == []
    assert "too long" in rejected[0]


def test_duplicate_of_existing_task_is_rejected(kb):
    accepted, rejected = _validate(
        [make_proposal(title="Morning Walk")], allowed(kb, "dog-walking"), {"morning walk"}
    )
    assert accepted == []
    assert "duplicate" in rejected[0]


def test_duplicate_within_one_batch_is_rejected(kb):
    proposals = [make_proposal(), make_proposal(title="morning walk")]
    accepted, rejected = _validate(proposals, allowed(kb, "dog-walking"), set())
    assert len(accepted) == 1
    assert "duplicate" in rejected[0]


def test_task_count_is_capped(kb):
    proposals = [make_proposal(title=f"Task {i}") for i in range(MAX_TASKS + 4)]
    accepted, rejected = _validate(proposals, allowed(kb, "dog-walking"), set())
    assert len(accepted) == MAX_TASKS
    assert len(rejected) == 4


def test_malformed_time_is_cleared_not_rejected(kb):
    # Losing a good task over a bad timestamp would be worse than dropping the
    # timestamp, so the task survives with no preferred time.
    accepted, rejected = _validate(
        [make_proposal(time="25:99")], allowed(kb, "dog-walking"), set()
    )
    assert rejected == []
    assert accepted[0].task.time == ""


def test_valid_time_is_preserved(kb):
    accepted, _ = _validate([make_proposal(time="07:30")], allowed(kb, "dog-walking"), set())
    assert accepted[0].task.time == "07:30"


def test_one_bad_proposal_does_not_discard_the_good_ones(kb):
    proposals = [
        make_proposal(title="Morning walk"),
        make_proposal(title="Bogus", source_id="not-real"),
        make_proposal(title="Evening meal", source_id="dog-walking"),
    ]
    accepted, rejected = _validate(proposals, allowed(kb, "dog-walking"), set())
    assert [s.task.title for s in accepted] == ["Morning walk", "Evening meal"]
    assert len(rejected) == 1


# --- end-to-end with a fake client ----------------------------------------


class FakeMessages:
    """Stands in for client.messages, returning a canned parsed response."""

    def __init__(self, suggestion: CarePlanSuggestion, stop_reason: str = "end_turn"):
        self.suggestion = suggestion
        self.stop_reason = stop_reason
        self.last_kwargs: dict = {}

    def parse(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            parsed_output=self.suggestion,
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1234, output_tokens=567),
        )


class FakeClient:
    def __init__(self, suggestion: CarePlanSuggestion, stop_reason: str = "end_turn"):
        self.messages = FakeMessages(suggestion, stop_reason)


def test_suggest_tasks_end_to_end(kb, owner):
    pet = Pet(name="Biscuit", species="dog", breed="labrador", age=4)
    owner.add_pet(pet)
    suggestion = CarePlanSuggestion(
        tasks=[
            make_proposal(title="Morning walk", source_id="dog-walking"),
            make_proposal(title="Hallucinated chore", source_id="rabbit-care"),
        ],
        coverage_note="Grooming left out for time.",
    )
    client = FakeClient(suggestion)

    result = suggest_tasks(pet, owner, kb=kb, client=client)

    assert [s.task.title for s in result.suggestions] == ["Morning walk"]
    assert len(result.rejected) == 1
    assert result.coverage_note == "Grooming left out for time."
    assert result.input_tokens == 1234 and result.output_tokens == 567
    # The sources actually retrieved must be reported back for the UI to cite.
    assert "dog-walking" in [c.id for c in result.sources]


def test_suggest_tasks_sends_sources_in_the_prompt(kb, owner):
    pet = Pet(name="Mochi", species="cat", age=12)
    owner.add_pet(pet)
    client = FakeClient(CarePlanSuggestion(tasks=[], coverage_note=""))

    suggest_tasks(pet, owner, kb=kb, client=client)

    sent = client.messages.last_kwargs["messages"][0]["content"]
    # Retrieval must reach the model: every retrieved id appears in the prompt.
    for chunk in kb.search(build_query(pet, owner), top_k=4):
        assert f'id="{chunk.id}"' in sent
    assert client.messages.last_kwargs["output_format"] is CarePlanSuggestion


def test_suggested_tasks_feed_the_existing_scheduler(kb, owner):
    """The whole point of the RAG layer: its output drives the real scheduler."""
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    owner.available_minutes = 45
    suggestion = CarePlanSuggestion(
        tasks=[
            make_proposal(title="Give medication", duration_minutes=5, priority="high"),
            make_proposal(title="Morning walk", duration_minutes=30, priority="high"),
            make_proposal(title="Brush coat", duration_minutes=40, priority="low"),
        ],
        coverage_note="",
    )
    result = suggest_tasks(pet, owner, kb=kb, client=FakeClient(suggestion))
    for s in result.suggestions:
        pet.add_task(s.task)

    plan = Scheduler(start_hour=8).build_plan(owner)

    assert plan.total_minutes <= owner.available_minutes
    scheduled = [i.task.title for i in plan.items]
    assert "Give medication" in scheduled and "Morning walk" in scheduled
    # The 40-minute low-priority groom cannot fit in the 10 minutes left.
    assert [i.task.title for i in plan.skipped] == ["Brush coat"]


def test_refusal_raises_a_friendly_error(kb, owner):
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    client = FakeClient(CarePlanSuggestion(tasks=[], coverage_note=""), stop_reason="refusal")
    with pytest.raises(PawPalAIError, match="declined"):
        suggest_tasks(pet, owner, kb=kb, client=client)


def test_missing_api_key_on_the_anthropic_path_raises(kb, owner, monkeypatch):
    # Explicitly asking for the paid backend without a key must fail fast,
    # before any request is made.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    with pytest.raises(PawPalAIError, match="ANTHROPIC_API_KEY"):
        suggest_tasks(pet, owner, kb=kb, provider="anthropic")


def test_missing_api_key_falls_back_to_the_free_local_backend(kb, owner, monkeypatch):
    # With no key and no explicit provider, the app must route to Ollama rather
    # than erroring — this is what makes it runnable with no account at all.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PAWPAL_PROVIDER", raising=False)
    called = {}

    def fake_ollama(system, user, model):
        called["model"] = model
        return CarePlanSuggestion(tasks=[], coverage_note="local"), 1, 2

    monkeypatch.setattr(pawpal_ai, "_generate_ollama", fake_ollama)
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)

    result = suggest_tasks(pet, owner, kb=kb)

    assert result.provider == "ollama"
    assert called["model"] == pawpal_ai.DEFAULT_OLLAMA_MODEL


def test_no_retrieval_hits_returns_empty_without_calling_the_api(kb, owner, monkeypatch):
    # If retrieval finds nothing there is nothing to ground an answer in, so we
    # must not fall through to the model.
    monkeypatch.setattr(kb, "search", lambda *a, **k: [])
    pet = Pet(name="Ghost", species="", age=None)
    owner.add_pet(pet)

    result = suggest_tasks(pet, owner, kb=kb)

    assert isinstance(result, SuggestionResult)
    assert result.suggestions == []
    assert "No care guidance matched" in result.coverage_note


# --- availability window scheduling ---------------------------------------


def window_owner(minutes=200, start="07:00", end="20:00") -> Owner:
    """An owner with an explicit availability window and one pet."""
    owner = Owner(name="Jordan", available_minutes=minutes, day_start=start, day_end=end)
    owner.add_pet(Pet(name="Biscuit", species="dog"))
    return owner


def test_timed_task_keeps_its_preferred_time():
    owner = window_owner()
    owner.pets[0].add_task(Task("Evening walk", 30, Priority.MEDIUM, time="18:00"))
    plan = Scheduler().build_plan(owner)
    assert plan.items[0].start_time == "18:00"


def test_untimed_tasks_fill_gaps_without_moving_anchors():
    # The bug this guards: placing a high-priority evening task first used to
    # push the entire rest of the routine into the evening behind it.
    owner = window_owner()
    pet = owner.pets[0]
    pet.add_task(Task("Evening walk", 30, Priority.HIGH, time="18:00"))
    pet.add_task(Task("Brush coat", 10, Priority.LOW))
    plan = Scheduler().build_plan(owner)
    by_title = {i.task.title: i.start_time for i in plan.items}
    assert by_title["Evening walk"] == "18:00"
    assert by_title["Brush coat"] == "07:00"  # start of the window, not 18:30


def test_tasks_never_start_before_the_window_opens():
    owner = window_owner(start="09:00")
    owner.pets[0].add_task(Task("Brush coat", 10, Priority.LOW))
    plan = Scheduler().build_plan(owner)
    assert plan.items[0].start_time == "09:00"


def test_task_finishing_after_the_window_is_skipped():
    owner = window_owner(minutes=500, end="20:00")
    owner.pets[0].add_task(Task("Late stroll", 45, Priority.HIGH, time="19:45"))
    plan = Scheduler().build_plan(owner)
    assert plan.items == []
    assert "before 20:00" in plan.skipped[0].reason


def test_untimed_task_too_big_for_the_window_is_skipped():
    owner = window_owner(minutes=900, start="07:00", end="08:00")
    owner.pets[0].add_task(Task("Marathon", 120, Priority.LOW))
    plan = Scheduler().build_plan(owner)
    assert plan.items == []
    assert "no free 120 min slot" in plan.skipped[0].reason


def test_overlapping_preferred_times_are_pushed_later():
    owner = window_owner()
    pet = owner.pets[0]
    pet.add_task(Task("Meal", 30, Priority.HIGH, time="08:00"))
    pet.add_task(Task("Walk", 30, Priority.MEDIUM, time="08:00"))  # clashes
    plan = Scheduler().build_plan(owner)
    starts = sorted(i.start_time for i in plan.items)
    assert starts == ["08:00", "08:30"]  # second one moved, not dropped
    assert any("moved from 08:00" in i.reason for i in plan.items)


def test_budget_and_window_are_independent_limits():
    # A wide window does not grant more hands-on minutes.
    owner = window_owner(minutes=20, start="06:00", end="23:00")
    pet = owner.pets[0]
    pet.add_task(Task("Walk", 15, Priority.HIGH))
    pet.add_task(Task("Groom", 15, Priority.LOW))
    plan = Scheduler().build_plan(owner)
    assert plan.total_minutes == 15
    assert [s.task.title for s in plan.skipped] == ["Groom"]


def test_items_come_back_in_chronological_order():
    owner = window_owner()
    pet = owner.pets[0]
    pet.add_task(Task("Evening", 10, Priority.LOW, time="19:00"))
    pet.add_task(Task("Morning", 10, Priority.HIGH, time="08:00"))
    pet.add_task(Task("Midday", 10, Priority.MEDIUM, time="12:00"))
    plan = Scheduler().build_plan(owner)
    starts = [i.start_time for i in plan.items]
    assert starts == sorted(starts)


def test_reversed_window_is_ignored_rather_than_fatal():
    owner = window_owner(start="20:00", end="06:00")
    owner.pets[0].add_task(Task("Walk", 10, Priority.HIGH))
    plan = Scheduler().build_plan(owner)  # must not raise or return nothing
    assert len(plan.items) == 1


def test_owner_without_a_window_keeps_the_old_behaviour():
    owner = Owner(name="Jordan", available_minutes=100)
    owner.add_pet(Pet(name="Biscuit", species="dog"))
    owner.pets[0].add_task(Task("A", 10, Priority.HIGH))
    owner.pets[0].add_task(Task("B", 20, Priority.LOW))
    plan = Scheduler(start_hour=8).build_plan(owner)
    assert [i.start_time for i in plan.items] == ["08:00", "08:10"]


def test_window_reaches_the_model_prompt(kb):
    owner = window_owner(start="06:30", end="19:15")
    prompt = build_user_prompt(owner.pets[0], owner, kb.search("dog walking", top_k=2))
    assert "06:30" in prompt and "19:15" in prompt


# --- model backend selection ----------------------------------------------


@pytest.mark.parametrize(
    "key, expected",
    [
        ("sk-ant-" + "x" * 90, True),      # looks like a real key
        ("sk-ant-...", False),             # the unedited .env.example placeholder
        ("", False),
        ("not-a-key", False),
        ('"sk-ant-' + "x" * 90 + '"', True),  # quoted in .env
    ],
)
def test_has_usable_api_key(monkeypatch, key, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    assert pawpal_ai.has_usable_api_key() is expected


def test_provider_defaults_to_ollama_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PAWPAL_PROVIDER", raising=False)
    assert pawpal_ai.resolve_provider() == "ollama"


def test_placeholder_key_still_routes_to_ollama(monkeypatch):
    # The trap this guards: the placeholder is non-empty, so a naive check
    # would send the user to the paid backend and fail with a 401.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    monkeypatch.delenv("PAWPAL_PROVIDER", raising=False)
    assert pawpal_ai.resolve_provider() == "ollama"


def test_real_key_routes_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 90)
    monkeypatch.delenv("PAWPAL_PROVIDER", raising=False)
    assert pawpal_ai.resolve_provider() == "anthropic"


def test_explicit_provider_overrides_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 90)
    monkeypatch.setenv("PAWPAL_PROVIDER", "ollama")
    assert pawpal_ai.resolve_provider() == "ollama"


def test_unknown_provider_falls_back_to_autodetect(monkeypatch):
    monkeypatch.setenv("PAWPAL_PROVIDER", "banana")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert pawpal_ai.resolve_provider() == "ollama"


def test_model_defaults_per_provider(monkeypatch):
    monkeypatch.delenv("PAWPAL_MODEL", raising=False)
    assert pawpal_ai.resolve_model("ollama") == pawpal_ai.DEFAULT_OLLAMA_MODEL
    assert pawpal_ai.resolve_model("anthropic") == pawpal_ai.DEFAULT_ANTHROPIC_MODEL


def test_model_env_override_wins(monkeypatch):
    monkeypatch.setenv("PAWPAL_MODEL", "qwen2.5:7b")
    assert pawpal_ai.resolve_model("ollama") == "qwen2.5:7b"


# --- time-of-day inference (fallback for models that skip the time field) --


def test_infer_time_of_day_recognises_evening():
    owner = window_owner(start="07:00", end="21:00")
    result = pawpal_ai.infer_time_of_day("Evening walk", "", owner)
    assert result != ""
    hour = int(result.split(":")[0])
    assert hour >= 16  # solidly in the back half of the window


def test_infer_time_of_day_recognises_morning():
    owner = window_owner(start="07:00", end="21:00")
    result = pawpal_ai.infer_time_of_day("Morning meal", "", owner)
    hour = int(result.split(":")[0])
    assert hour <= 9  # solidly in the front of the window


def test_infer_time_of_day_checks_category_too():
    # Some proposals put the time-of-day word in the category, not the title.
    owner = window_owner(start="07:00", end="21:00")
    assert pawpal_ai.infer_time_of_day("Feed", "evening meal", owner) != ""


def test_infer_time_of_day_no_hint_returns_empty():
    owner = window_owner(start="07:00", end="21:00")
    assert pawpal_ai.infer_time_of_day("Brush coat", "grooming", owner) == ""


def test_infer_time_of_day_without_a_window_returns_empty():
    owner = Owner(name="Jordan", available_minutes=100)  # no day_start/day_end
    assert pawpal_ai.infer_time_of_day("Evening walk", "", owner) == ""


def test_infer_time_of_day_is_within_the_window():
    owner = window_owner(start="09:00", end="17:00")
    for title in ("Morning meal", "Midday walk", "Evening brush", "Bedtime check"):
        result = pawpal_ai.infer_time_of_day(title, "", owner)
        minutes = Scheduler.parse_time(result)
        assert Scheduler.parse_time("09:00") <= minutes <= Scheduler.parse_time("17:00")


def test_validate_fills_a_blank_time_from_the_title(kb):
    owner = window_owner(start="07:00", end="21:00")
    proposal = make_proposal(title="Evening walk", time="", source_id="dog-walking")
    accepted, rejected = _validate(
        [proposal], allowed(kb, "dog-walking"), set(), owner=owner
    )
    assert rejected == []
    assert accepted[0].task.time != ""
    assert int(accepted[0].task.time.split(":")[0]) >= 16


def test_validate_without_owner_leaves_blank_time_blank(kb):
    # No owner means no window to place a guess in — this must not crash.
    proposal = make_proposal(title="Evening walk", time="", source_id="dog-walking")
    accepted, _ = _validate([proposal], allowed(kb, "dog-walking"), set())
    assert accepted[0].task.time == ""


def test_validate_does_not_override_a_time_the_model_did_provide(kb):
    owner = window_owner(start="07:00", end="21:00")
    proposal = make_proposal(title="Evening walk", time="08:00", source_id="dog-walking")
    accepted, _ = _validate([proposal], allowed(kb, "dog-walking"), set(), owner=owner)
    assert accepted[0].task.time == "08:00"  # model's explicit time wins


def test_end_to_end_evening_task_lands_in_the_evening(kb, owner):
    """The whole point of the fallback, exercised through suggest_tasks()."""
    owner.day_start, owner.day_end = "07:00", "21:00"
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    suggestion = CarePlanSuggestion(
        tasks=[
            make_proposal(title="Morning walk", time="", duration_minutes=20),
            make_proposal(title="Evening walk", time="", duration_minutes=20),
        ],
        coverage_note="",
    )
    result = suggest_tasks(pet, owner, kb=kb, client=FakeClient(suggestion))
    by_title = {s.task.title: s.task.time for s in result.suggestions}
    assert Scheduler.parse_time(by_title["Evening walk"]) > Scheduler.parse_time(
        by_title["Morning walk"]
    )


# --- ollama backend (no server needed) ------------------------------------


def fake_ollama_response(payload: dict):
    """Build a stand-in for the object urllib.request.urlopen returns."""

    class FakeResponse:
        def read(self):
            return json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return FakeResponse()


def test_ollama_backend_parses_a_valid_reply(monkeypatch):
    reply = {
        "message": {
            "content": CarePlanSuggestion(
                tasks=[make_proposal()], coverage_note="all good"
            ).model_dump_json()
        },
        "prompt_eval_count": 1189,
        "eval_count": 436,
    }
    monkeypatch.setattr(
        pawpal_ai.urllib.request, "urlopen", lambda *a, **k: fake_ollama_response(reply)
    )
    parsed, tin, tout = pawpal_ai._generate_ollama("sys", "user", "llama3.2:3b")
    assert parsed.tasks[0].title == "Morning walk"
    assert (tin, tout) == (1189, 436)


def test_ollama_offline_gives_a_startup_hint(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(pawpal_ai.urllib.request, "urlopen", boom)
    with pytest.raises(PawPalAIError, match="ollama serve"):
        pawpal_ai._generate_ollama("sys", "user", "llama3.2:3b")


def test_ollama_missing_model_tells_you_to_pull_it(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.HTTPError("url", 404, "not found", {}, io.BytesIO(b"no model"))

    monkeypatch.setattr(pawpal_ai.urllib.request, "urlopen", boom)
    with pytest.raises(PawPalAIError, match="ollama pull"):
        pawpal_ai._generate_ollama("sys", "user", "llama3.2:3b")


def test_ollama_malformed_json_is_a_friendly_error(monkeypatch):
    # Small local models sometimes drift off-schema; that must not crash.
    reply = {"message": {"content": '{"tasks": "not a list"}'}}
    monkeypatch.setattr(
        pawpal_ai.urllib.request, "urlopen", lambda *a, **k: fake_ollama_response(reply)
    )
    with pytest.raises(PawPalAIError, match="didn't match the expected"):
        pawpal_ai._generate_ollama("sys", "user", "llama3.2:3b")


def test_ollama_path_runs_the_same_guardrails(kb, owner, monkeypatch):
    """The point of the backend split: guardrails are provider-agnostic."""
    reply = {
        "message": {
            "content": CarePlanSuggestion(
                tasks=[
                    make_proposal(title="Morning walk", source_id="dog-walking"),
                    make_proposal(title="Bogus", source_id="invented-source"),
                ],
                coverage_note="",
            ).model_dump_json()
        },
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    monkeypatch.setattr(
        pawpal_ai.urllib.request, "urlopen", lambda *a, **k: fake_ollama_response(reply)
    )
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)

    result = suggest_tasks(pet, owner, kb=kb, provider="ollama")

    assert [s.task.title for s in result.suggestions] == ["Morning walk"]
    assert len(result.rejected) == 1
    assert result.provider == "ollama"


def test_api_error_becomes_a_user_safe_message(kb, owner, monkeypatch):
    import anthropic

    class ExplodingMessages:
        def parse(self, **kwargs):
            raise anthropic.APIConnectionError(request=SimpleNamespace())

    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    client = SimpleNamespace(messages=ExplodingMessages())

    with pytest.raises(PawPalAIError, match="Could not reach the API"):
        suggest_tasks(pet, owner, kb=kb, client=client)
