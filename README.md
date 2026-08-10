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

Requires Python 3.10+. PawPal+ runs against **either** of two model backends —
pick one. Both use the identical retrieval, guardrails and scheduler; only the
model call differs.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up a backend — see Option A or B below

# 4. Run it
python -m streamlit run app.py
```

Use `python -m streamlit run app.py`, not a bare `streamlit run app.py` — if
you have Anaconda or another Python installed, a bare `streamlit` can silently
resolve to *that* interpreter, which won't have this project's packages and
fails with `ModuleNotFoundError`. Running it through `python -m` guarantees
you're using the virtual environment you just activated.

### Option A — free, local, no account (recommended for trying this out)

Runs a small open model on your own machine via [Ollama](https://ollama.com).
No API key, no billing, works offline once the model is downloaded.

```bash
brew install ollama        # macOS; see ollama.com/download for other OSes
brew services start ollama # runs Ollama in the background, restarts on reboot
ollama pull llama3.2:3b    # one-time download, ~2GB

cp .env.example .env       # already defaults to PAWPAL_PROVIDER=ollama
```

Needs roughly 4GB of free RAM. Each suggestion takes 20–50 seconds — slower
than a cloud API because it's your own hardware doing the work.

### Option B — Anthropic API (paid, faster, higher quality)

```bash
cp .env.example .env
```

Open `.env` and switch the two `PAWPAL_*` lines to `anthropic`/`claude-opus-5`,
then add a real key from <https://console.anthropic.com/settings/keys>:

```
PAWPAL_PROVIDER=anthropic
PAWPAL_MODEL=claude-opus-5
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

A Claude.ai subscription does **not** include API access — it's billed
separately, pay-as-you-go, no free tier. One suggestion costs roughly a cent.

`.env` is gitignored either way — don't commit it.

### Which backend is active

The app tells you, right above the **Suggest tasks** button:

- 🖥️ Running locally via Ollama — model `llama3.2:3b` · free, no API key
- ☁️ Running on the Anthropic API — model `claude-opus-5`

If `PAWPAL_PROVIDER` isn't set at all, the app auto-detects: it uses Anthropic
only if `ANTHROPIC_API_KEY` looks like a real key, and falls back to Ollama
otherwise — so an unedited `.env.example` (with its `sk-ant-...` placeholder)
safely defaults to the free path instead of failing with a confusing 401.

### Run the tests

```bash
python -m pytest
```

141 tests, all offline and free — the AI tests inject a fake client (and, for
the Ollama path, a fake HTTP response), so the suite needs no API key, no
running Ollama server, and no network call.

### Run the CLI demo (no AI backend needed at all)

```bash
python main.py
```

Exercises the scheduler, conflict detection, sorting, filtering and recurring
tasks without touching either model backend.

---

## 🤖 How the RAG feature works

Full system diagram (Mermaid source):
[`diagrams/architecture.mmd`](diagrams/architecture.mmd) — components, data
flow, and the three places AI output is checked. Simplified view:

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
 │  Ollama (free) or Anthropic   │  pawpal_ai.py          │
 │  structured output — "use    │                        │
 │  ONLY these sources, cite    │                        │
 │  the id for each task"       │                        │
 └──────────────────────────────┘                        │
        │  proposed tasks + source_id + rationale        │
        ▼                                                │
 ┌──────────────────┐   source_id must be one we ◄───────┘
 │   _validate()    │   actually retrieved, or the
 │   guardrails     │   task is dropped
 └──────────────────┘
        │  validated Task objects, times filled in
        ▼
 ┌──────────────────┐
 │    Scheduler     │  ranks by priority within the budget,
 │  build_plan()    │  places by time within day_start..day_end
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
| Blank `time` gets a guess | If the model leaves `time` empty, `infer_time_of_day()` looks for a time-of-day word in the task's own title (`"Evening walk"` → late in the window) so smaller local models that skip the field still get sensibly ordered days. |

Rejections are shown in the UI with a 🛡️ marker, so you can see what the system
threw away and why rather than silently trusting it.

### Getting a full routine, not one task

Smaller models — especially the local one — can undershoot: asked for a day's
routine, they sometimes return a single task and stop, even when told
explicitly to cover the whole day. Two layers fix this:

1. **The prompt asks for a specific minimum.** `SYSTEM_PROMPT` requires at
   least `MIN_TASKS_REQUESTED` (4) tasks whenever the sources support it, and
   names the categories to cover — feeding, toileting, exercise, medication,
   grooming — instead of leaving "propose some tasks" open to a one-item
   answer.
2. **A one-shot top-up call is the safety net.** If fewer than
   `MIN_ACCEPTED_TASKS` (3) tasks survive validation, `suggest_tasks()`
   automatically sends one follow-up request — `build_topup_prompt()` lists
   what was already accepted and asks the model to add to it rather than
   repeat it, so the retry isn't just a byte-for-byte rerun of the same
   prompt (which would return the same short answer, since the local
   backend runs at `temperature: 0` for reproducibility). Results from both
   calls are merged, deduplicated by title, and capped at `MAX_TASKS`. If the
   top-up call itself fails, the first call's tasks are kept rather than
   losing everything.

This only ever fires **once** per suggestion — it's a correction for a weak
first answer, not a loop chasing a target count. Token usage from both calls
is added together and shown in the UI, so a top-up isn't hidden cost.

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
| `pawpal_system.py` | Domain model and scheduling. **No AI, no I/O.** |
| `retriever.py` | Loads and searches the knowledge base. Deterministic, no network. |
| `pawpal_ai.py` | Prompt assembly, the model call (Ollama or Anthropic), guardrails, logging. |
| `app.py` | Streamlit UI. |
| `main.py` | CLI demo of the scheduler. |

Inside `pawpal_system.py`:

| Class | Responsibility | Key attributes | Key methods |
|-------|----------------|----------------|-------------|
| `Priority` | Enum ranking a task's importance | `HIGH`, `MEDIUM`, `LOW` | — |
| `Owner` | The pet owner: time budget, availability window, pets | `name`, `available_minutes`, `day_start`, `day_end`, `preferences`, `pets` | `add_pet()`, `filter_tasks()` |
| `Pet` | A pet and its care tasks | `name`, `species`, `breed`, `color`, `age`, `tasks` | `add_task()`, `complete_task()` |
| `Task` | One care activity | `title`, `duration_minutes`, `priority`, `time`, `frequency`, `due_date`, `status` | `rank()`, `mark_complete()`, `next_occurrence()` |
| `PlanItem` | One scheduling decision | `task`, `pet`, `start_time`, `included`, `reason` | — |
| `DailyPlan` | The scheduled + skipped result | `owner`, `items`, `skipped`, `total_minutes` | `explain()` |
| `Scheduler` | Chooses *what* fits the budget, then *when* it happens within the window | `start_hour` (fallback only) | `build_plan()`, `sort_by_time()`, `detect_conflicts()`, `parse_time()` |

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

`Scheduler.build_plan()` runs in two phases, in this order:

1. **What to do** — tasks are ranked by priority (high → low, ties broken by
   shorter duration first) and accepted one at a time until
   `owner.available_minutes` runs out. Everything past that point is skipped,
   with the reason recorded on its `PlanItem`.
2. **When to do it** — accepted tasks with a preferred `time` are placed first,
   at that time, so a task like `"18:00"` keeps its slot no matter what else
   gets scheduled. Tasks with no preferred time then fill whatever gaps remain,
   earliest first. Everything is bounded by `owner.day_start`..`owner.day_end`
   (`Scheduler.parse_time()` turns `"HH:MM"` into minutes since midnight for
   the comparison) — a task that would start before the window opens or finish
   after it closes is skipped instead of forced in.

This split is what stops a single high-priority evening task from dragging the
entire rest of the day's routine into the evening behind it: anchored tasks
never move, only the flexible ones get shuffled around them.

| Feature | Method(s) | Notes |
|---------|-----------|-------|
| Priority + time-budget selection | `Scheduler.build_plan()`, `Task.rank()`, `Scheduler._sort_tasks()` | Phase 1 above |
| Time-of-day placement | `Scheduler.build_plan()`, `Scheduler._find_slot()` | Phase 2 above; a clashing preferred time is pushed later rather than dropped |
| Availability window | `Owner.day_start`, `Owner.day_end`, `Scheduler.parse_time()` | Independent from the minute budget — a wide window doesn't grant more hands-on time, and a generous budget doesn't override the cutoff |
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
python -m pytest                     # all 141
python -m pytest tests/test_rag.py   # just the AI layer
```

`tests/test_pawpal.py` (42 tests) covers the domain model: ranking, sorting by
priority-then-duration with deterministic tie-breaking, chronological
`sort_by_time()`, filtering, conflict detection, recurring tasks including
month/year rollover, the "today only" planning filter, and edge cases like a
zero-minute budget and an owner with no pets. It also covers the two-phase
scheduler introduced for time placement: preferred times are kept, clashing
preferred times are pushed later rather than dropped, untimed tasks fill gaps
without disturbing anchored ones, tasks are rejected when they'd finish after
`day_end`, the minute budget and the availability window are enforced as
independent limits, and an owner with no window falls back to the original
back-to-back behaviour unchanged.

`tests/test_rag.py` (99 tests) covers the AI layer:

- **Tokenizer & knowledge loading** — lowercasing, stopwords, singularisation
  edge cases; titles/tags parsed correctly; a missing *or empty* knowledge
  directory raises rather than silently returning nothing.
- **Retrieval quality** — a cat query never retrieves dog guidance (and vice
  versa), seniors retrieve senior guidance, puppies retrieve the puppy rule,
  flat-faced breeds retrieve the breathing warning, a tight budget retrieves
  triage, results are deterministic across runs.
- **Prompt assembly** — the profile, budget, availability window, existing
  tasks and every retrieved source id reach the model.
- **Guardrails** — every rule in the table above, including that a task citing
  a *real* chunk that wasn't retrieved for *this* pet is still rejected, and
  that one bad proposal doesn't discard the good ones.
- **Time-of-day inference** — `infer_time_of_day()` places "Evening walk"-style
  titles late in the window and "Morning meal"-style titles early, checks the
  category field too, never touches a time the model *did* provide, and does
  nothing when there's no window to place a guess in.
- **Model backend selection** — the placeholder key in an unedited
  `.env.example` routes to Ollama rather than attempting (and failing) an
  Anthropic call; `PAWPAL_PROVIDER` overrides auto-detection; per-provider
  model defaults and the `PAWPAL_MODEL` override.
- **Ollama backend** — schema-valid replies parse correctly; an unreachable
  server, a missing model, and an off-schema reply each become a specific,
  actionable message rather than a crash — all via a mocked HTTP layer, so no
  server needs to be running for the suite to pass.
- **Top-up on a short first answer** — a too-short response triggers exactly
  one follow-up call, on both backends; the second prompt is verified to
  actually differ from the first (not a no-op retry at `temperature: 0`);
  results merge and dedupe correctly; the merge respects `MAX_TASKS`; and a
  failed top-up keeps the first call's partial result instead of raising. One
  test disables the trigger and confirms 5 of these fail — proof the suite
  would actually catch this bug coming back.
- **End to end** — on both backends: suggestions flow into `Scheduler` and come
  back correctly scheduled and skipped, refusals and API errors become
  friendly messages, a missing key on the Anthropic path fails before any
  request is made (while the auto-detect path instead falls back to Ollama),
  and zero retrieval hits short-circuits without calling any model at all.

---

## 📸 Demo walkthrough

1. **Set the owner, time budget, and availability window** — name, hands-on
   minutes available today (e.g. 120), and when you're free from/until (e.g.
   `07:00`–`20:00`). The window and the budget are independent: being free for
   13 hours doesn't grant more than 120 minutes of actual task time.
2. **Add pets** — name, species, breed, color, age. Try `Mochi / cat / persian /
   12` and `Biscuit / dog / pug / 3` to see breed and age change what's
   retrieved.
3. **Add tasks manually** (optional) — the AI won't duplicate anything you've
   already added.
4. **Get AI suggestions** — pick a pet and click **🤖 Suggest tasks**. Expand
   **📚 Sources retrieved for this pet** to see exactly what the suggestions were
   based on; each task shows its rationale and source. Untick anything you don't
   want and click **Add selected tasks**.
5. **Check the timeline** for clashes, then **Generate schedule**. Meals,
   medication, and walks land where they belong across your window — morning
   tasks early, evening tasks late — instead of stacked back-to-back at the
   start of the day. The "Why this plan?" section explains every decision.

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or link to a demo video here -->
