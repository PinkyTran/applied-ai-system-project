"""Tests for the RAG layer: retrieval, guardrails, and the end-to-end path.

No test here makes a network call. Retrieval is deterministic local code, and
the one test that exercises suggest_tasks() injects a fake client, so the whole
suite runs offline and for free.
"""

from __future__ import annotations

import sys
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


def test_missing_api_key_raises_before_calling_the_api(kb, owner, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pet = Pet(name="Biscuit", species="dog", age=4)
    owner.add_pet(pet)
    with pytest.raises(PawPalAIError, match="ANTHROPIC_API_KEY"):
        suggest_tasks(pet, owner, kb=kb)  # no client injected


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
