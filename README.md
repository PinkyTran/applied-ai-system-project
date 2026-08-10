# 🐾 PawPal+ — an AI care planner for pet owners

PawPal+ helps a busy pet owner stay consistent with pet care. You describe your
pets and how much time you have today; it **looks up real care guidance for
each pet**, drafts the right tasks from that guidance, then packs them into a
schedule that fits your time budget and explains every decision.

The AI feature is **Retrieval-Augmented Generation (RAG)**: before suggesting
anything, the app retrieves the care guidance that applies to *this* pet — its
species, breed, and life stage — and that retrieved text is the only source the
model is allowed to use. A 12-year-old Persian gets senior-cat and flat-faced
breed guidance; a 4-month-old border collie gets the puppy exercise rule. The
retrieved sources decide the durations, priorities and frequencies that end up
in your schedule, and any task the model can't tie back to a retrieved source is
rejected before it reaches the plan.

---

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
cp .env.example .env               # Windows: copy .env.example .env
# then open .env and paste your key from
# https://console.anthropic.com/settings/keys
```

Your `.env` should contain:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

`.env` is gitignored — don't commit it.

### Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

### Run the tests

```bash
pytest
```

93 tests, all offline — the AI tests inject a fake client, so the suite needs no
API key and costs nothing to run.

### Run the CLI demo (no API key needed)

```bash
python main.py
```

Exercises the scheduler, conflict detection, sorting, filtering and recurring
tasks without touching the AI layer.

---

## 🤖 How the RAG feature works

```
 Pet profile (species, breed, age) + owner's time budget
        │
        │  build_query()          retriever.py
        ▼
 ┌──────────────────┐  TF-IDF search over knowledge/*.md
 │  KnowledgeBase   │ ───────────────────────────────────┐
 └──────────────────┘                                    │
        │  top 4 relevant care chunks                    │
        ▼                                                │
 ┌──────────────────────────────┐                        │
 │  Claude + structured output  │  pawpal_ai.py          │
 │  "use ONLY these sources,    │                        │
 │   cite the id for each task" │                        │
 └──────────────────────────────┘                        │
        │  proposed tasks + source_id + rationale        │
        ▼                                                │
 ┌──────────────────┐   source_id must be one we ◄───────┘
 │   _validate()    │   actually retrieved, or the
 │   guardrails     │   task is dropped
 └──────────────────┘
        │  validated Task objects
        ▼
 ┌──────────────────┐
 │    Scheduler     │  the existing, unchanged domain logic
 │  build_plan()    │
 └──────────────────┘
        │
        ▼  DailyPlan — scheduled, skipped, and why
```

**The retrieval genuinely changes the output.** The model is given no pet-care
knowledge of its own to fall back on: the retrieved chunks are pasted into the
prompt as the sole source of guidance, each labelled with an id, and every task
must cite the id it came from. Ask for a pug and `flat-faced-breeds.md` is
retrieved, so the walk comes back capped at 20 minutes. Ask for a labrador and
it doesn't, so the walk is 30–45. Same code, different sources, different plan.

### Retrieval (`retriever.py`)

Hand-rolled TF-IDF over the markdown files in [`knowledge/`](knowledge/) — no
vector database, no embedding model, no network call. That keeps the retrieval
step **deterministic and reproducible**: the same pet always retrieves the same
sources on any machine, which is why it can be unit-tested on its own.

- Query terms are **repeated to weight them** — species four times, breed twice
  — because `score()` counts query-side repetition. Without this, a cat query
  retrieved dog grooming advice.
- The topic tail is **species-aware** (`litter box` for cats, `walk potty` for
  dogs), which is what pulls the toileting chunk into range.
- `age_band()` converts a number into the vocabulary the corpus actually uses
  ("senior", "puppy"), because the chunks are written in words, not ages.

### Guardrails (`_validate()` in `pawpal_ai.py`)

Nothing reaches the domain model without passing every check:

| Check | Why |
|---|---|
| `source_id` was actually retrieved | **The grounding check.** A citation we never retrieved means the model went outside its sources — the task is dropped and logged as a warning. |
| Duration is 1–240 minutes | A 0-minute task crashes `Task.__init__`; a 10,000-minute one is a model error. |
| Priority / frequency are known values | Enforced by the schema *and* re-checked, so bad data can't reach `Priority[...]`. |
| Title present and ≤ 60 chars | Keeps the UI and plan readable. |
| No duplicates | Against tasks you already added *and* within the same batch. |
| At most 8 tasks | A 20-task day is noise, not help. |
| `time` is `HH:MM` or empty | A malformed time is **cleared, not rejected** — losing a good task over a bad timestamp would be worse. |

Rejections are shown in the UI with a 🛡️ marker, so you can see what the system
threw away and why rather than silently trusting it.

### Error handling

Every expected failure becomes a plain message instead of a stack trace:
missing API key, rejected key, no internet, rate limiting, unknown model, and
safety refusals (`stop_reason == "refusal"`, which arrives as a normal `200`
response and would otherwise crash on empty content).

### Logging

`setup_logging()` writes to **`pawpal.log`** and the console. Every run records
the retrieval query and which chunks scored highest, how many tasks were
proposed / accepted / rejected, each rejection reason, and token usage:

```
2026-08-09 14:22:01 INFO    retriever: Retrieval query='cat cat cat cat persian persian senior old elderly...'
                            -> [('senior-pets', 1.412), ('cat-litter', 1.166), ('cat-grooming', 1.081)]
2026-08-09 14:22:06 INFO    pawpal_ai: Pet Mochi: 6 proposed -> 5 accepted, 1 rejected (tokens in=2871 out=612)
2026-08-09 14:22:06 WARNING pawpal_ai: Rejected: 'Annual vaccination' was dropped: cites unknown source 'vet-visits' that was not retrieved.
```

### The knowledge base

[`knowledge/`](knowledge/) holds 14 markdown chunks — walking, feeding (dog and
cat), litter care, senior pets, puppies, kittens, grooming, medication,
enrichment, flat-faced breeds, time-budget triage, and home health checks. Each
file starts with a `# Title` and one or more `tags:` lines, which are weighted
3× in scoring.

**To extend it, drop another `.md` file in that folder.** No code change and no
re-indexing step — it's picked up on the next app start.

---

## 🏗️ Classes and modules

PawPal+ keeps the domain model, the AI layer, and the UI separate:

| Module | Responsibility |
|---|---|
| `pawpal_system.py` | Domain model and scheduling. **No AI, no I/O** — unchanged by this feature. |
| `retriever.py` | Loads and searches the knowledge base. Deterministic, no network. |
| `pawpal_ai.py` | Prompt assembly, the model call, guardrails, logging. |
| `app.py` | Streamlit UI. |
| `main.py` | CLI demo of the scheduler. |

Inside `pawpal_system.py`:

| Class | Responsibility | Key attributes | Key methods |
|-------|----------------|----------------|-------------|
| `Priority` | Enum ranking a task's importance | `HIGH`, `MEDIUM`, `LOW` | — |
| `Owner` | The pet owner: time budget + pets | `name`, `available_minutes`, `preferences`, `pets` | `add_pet()`, `filter_tasks()` |
| `Pet` | A pet and its care tasks | `name`, `species`, `breed`, `color`, `age`, `tasks` | `add_task()`, `complete_task()` |
| `Task` | One care activity | `title`, `duration_minutes`, `priority`, `time`, `frequency`, `due_date`, `status` | `rank()`, `mark_complete()`, `next_occurrence()` |
| `PlanItem` | One scheduling decision | `task`, `pet`, `start_time`, `included`, `reason` | — |
| `DailyPlan` | The scheduled + skipped result | `owner`, `items`, `skipped`, `total_minutes` | `explain()` |
| `Scheduler` | Ranks and fits tasks across all pets | `start_hour` | `build_plan()`, `sort_by_time()`, `detect_conflicts()` |

And in the AI layer:

| Class | Responsibility |
|---|---|
| `Chunk` | One retrievable knowledge document (`id`, `title`, `tags`, `text`) |
| `KnowledgeBase` | TF-IDF index over the chunks; `load()`, `search()` |
| `ProposedTask` / `CarePlanSuggestion` | The structured-output schema the model fills in |
| `SuggestedTask` | A validated `Task` plus the source that justifies it |
| `SuggestionResult` | One run's output: suggestions, sources, coverage note, rejections, tokens |

Relationships: an `Owner` has many `Pet`s, each `Pet` owns many `Task`s, and the
`Scheduler` reads an `Owner` to produce a `DailyPlan` of `PlanItem`s. See
[`diagrams/uml_final.mmd`](diagrams/uml_final.mmd) for the full class diagram.

---

## 📐 Smarter scheduling

Beyond building a basic plan, PawPal+ implements several scheduling behaviors.
Each is a named method so it can be tested and reused independently of the UI.

| Feature | Method(s) | Notes |
|---------|-----------|-------|
| Priority + time-budget planning | `Scheduler.build_plan()`, `Task.rank()`, `Scheduler._sort_tasks()` | Ranks tasks by priority (HIGH→LOW) then shorter duration first, packs them back-to-back from `start_hour`, and skips any task that doesn't fit the remaining budget (recording the reason on its `PlanItem`) |
| Sorting by time of day | `Scheduler.sort_by_time()` | Returns tasks ordered by their `"HH:MM"` time using a lambda key; untimed tasks sort last via a `"99:99"` sentinel. Does not mutate the input |
| Filtering | `Owner.filter_tasks(status, pet_name)` | Returns tasks filtered by completion status and/or pet name; either filter is optional, so no arguments returns everything |
| Conflict detection | `Scheduler.detect_conflicts()` | Groups all timed tasks by their `"HH:MM"` slot in one O(n) pass and returns a warning string for any slot claimed by 2+ tasks. Catches same-pet **and** cross-pet clashes, and returns warnings as data instead of raising |
| Recurring tasks | `Task.next_occurrence()`, `Pet.complete_task()` | Completing a `daily`/`weekly` task auto-creates the next occurrence, advancing the due date with `timedelta` (handles month/year rollover). The follow-up starts `pending` |

**Sorting** — `sort_by_time()` relies on the fact that zero-padded 24-hour
strings sort chronologically as plain text, so the key is simply
`lambda t: t.time or "99:99"`.

**Conflict detection** uses a *lightweight* strategy: it never raises. It
returns a list like `["⚠️ Conflict at 08:00: 'Morning walk' (Biscuit), 'Backyard
potty' (Biscuit)"]`, leaving the caller (CLI or Streamlit) to decide how to
display it.

**Recurring tasks** — because a `Task` doesn't hold a reference to its `Pet`,
the recurrence is spawned by `Pet.complete_task()` (which owns the task list),
while `Task.next_occurrence()` does the pure date math. A `once` task returns
`None` and nothing is spawned.

---

## 🧪 Tests

```bash
pytest              # all 93
pytest tests/test_rag.py -v      # just the RAG layer
```

`tests/test_pawpal.py` (42 tests) covers the domain model: ranking, sorting by
priority-then-duration with deterministic tie-breaking, chronological
`sort_by_time()`, filtering, conflict detection, the time-budget cutoff and skip
reasons, editing in place, `mark_complete()`, recurring tasks including
month/year rollover, the "today only" planning filter, and edge cases like a
zero-minute budget and an owner with no pets.

`tests/test_rag.py` (51 tests) covers the AI layer:

- **Tokenizer** — lowercasing, stopwords, singularisation edge cases.
- **Loading** — titles and tags parsed, tags kept out of the body, and a
  missing *or empty* knowledge directory raising rather than silently
  returning nothing.
- **Retrieval quality** — a cat query never retrieves dog guidance (and vice
  versa), seniors retrieve senior guidance, puppies retrieve the puppy rule,
  flat-faced breeds retrieve the breathing warning, a tight budget retrieves
  triage, results are deterministic across runs, and `top_k` / `min_score` are
  respected.
- **Prompt assembly** — the profile, budget, existing tasks and every retrieved
  source id reach the model.
- **Guardrails** — each rule above, including that a task citing a *real* chunk
  that wasn't retrieved for *this* pet is still rejected, that one bad proposal
  doesn't discard the good ones, and that boundary durations (1 and 240) pass.
- **End to end** — with a fake client: suggestions flow into `Scheduler` and
  come back correctly scheduled and skipped, refusals and API errors become
  friendly messages, a missing key fails before any request is made, and zero
  retrieval hits short-circuits without calling the model at all.

---

## 📸 Demo walkthrough

1. **Set the owner and time budget** — name, minutes available today (e.g. 200),
   and the hour your day starts (e.g. 8).
2. **Add pets** — name, species, breed, color, age. Try `Mochi / cat / persian /
   12` and `Biscuit / dog / pug / 3` to see breed and age change what's
   retrieved.
3. **Add tasks manually** (optional) — the AI won't duplicate anything you've
   already added.
4. **Get AI suggestions** — pick a pet and click **🤖 Suggest tasks**. Expand
   **📚 Sources retrieved for this pet** to see exactly what the suggestions were
   based on; each task shows its rationale and source. Untick anything you don't
   want and click **Add selected tasks**.
5. **Check the timeline** for clashes, then **Generate schedule** to see the
   scheduled tasks, the skipped ones with reasons, and the full "Why this plan?"
   explanation.

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or link to a demo video here -->
