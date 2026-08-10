# PawPal+ Project Reflection

> Draft based on how the project actually came together — edit into your own
> voice and add anything personal before submitting.

## 1. System Design

**a. Initial design**

My initial UML modeled seven pieces: a `Priority` enum (HIGH/MEDIUM/LOW), and
classes for `Owner`, `Pet`, `Task`, `PlanItem`, `DailyPlan`, and `Scheduler`.
The responsibilities were split so that the data classes stay simple and the
`Scheduler` holds the logic:

- `Owner` holds the constraint (the daily time budget) and the pets.
- `Pet` holds descriptive traits and its own list of care tasks.
- `Task` describes one activity (duration + priority) and can rank itself.
- `Scheduler` ranks tasks and fits them into the budget — it's stateless.
- `PlanItem` records one decision (which task, which pet, when, and why).
- `DailyPlan` collects the scheduled and skipped items and can `explain()` them.

**b. Design changes**

Yes. The biggest change was the ownership structure. My first draft had the `Scheduler` take tasks as a separate argument and handle one pet. I changed it to an `Owner → Pet → Task` chain (an owner has pets, each pet owns its tasks), so `build_plan(owner)` now collects `(pet, task)` pairs across all pets. That meant adding a `pet` field to `PlanItem` so the plan could still say which pet each task belongs to. I also split `Task` from `PlanItem` deliberately: a `Task` is static input, while a `PlanItem` is the scheduler's decision about it — which kept the "explain why" data out of the raw task.

---

## 2. Scheduling Logic and Tradeoffs

**a. Constraints and priorities**

The scheduler considers two constraints: the owner's available minutes (a hard time budget) and each task's priority. It sorts tasks by priorit
first (HIGH → LOW), breaking ties by shorter duration, then packs them
back-to-back from a start hour until the budget runs out. I decided time and priority mattered most because they're what a busy owner actually feels — have a fixed amount of time and want the important things done first.

**b. Tradeoffs**

The scheduler is greedy: once a task doesn't fit, it keeps filling the
leftover time with smaller tasks — even lower-priority ones. So a LOW-priority 15-minute task can be scheduled while a MEDIUM 30-minute task is skipped. The tradeoff is "don't waste free minutes" vs. "strict priority order." I think it's reasonable here because leaving 15 minutes idle to protect a task that can't fit anyway doesn't help the pet — but it's a real tradeoff, and which task gets skipped is always the longest task in the lowest tier that still has candidates(deterministic, not random).

---

## 3. AI Collaboration

**a. How you used AI**

I used AI across the whole workflow: brainstorming the UML, generating class stubs from the diagram, implementing the scheduling logic incrementally, writing tests, and wiring the logic into the Streamlit UI. The most helpful prompts were specific ones — e.g. asking how `st.session_state` persists objects across reruns, and asking what edge cases the design might miss.

**b. Judgment and verification**

One moment I didn't accept a suggestion as-is: a test I wrote assumed tasks stay in the order I added them, but the scheduler correctly re-sorts by duration within a priority tier, so a shorter task scheduled first. The test failed and the code was right, not the test. I fixed the test's expectation instead of changing correct code. I verified behavior by running `python main.py` and `pytest`, and by reading the plan's explanation to confirm the reasons matched.

---

## 4. Testing and Verification

**a. What you tested**

I tested: priority ranking order; sorting by priority then duration; stable (deterministic) ordering for ties; chronological `sort_by_time()`; filtering by status and pet; conflict detection (same-pet and cross-pet); the budget cutoff and the skip reasons; the greedy "fill leftover time" behavior; `total_minutes` accounting; sequential start times; `mark_complete()` changing status; `add_task()` growing a pet's task count; recurring tasks; and edge cases (zero budget, an owner with no pets, a pet with no tasks). These matter because they're the behaviors a user actually relies on — that the important tasks come first and that nothing silently disappears without a reason.

**b. Confidence**

Fairly confident — 42 tests pass and cover the core paths. With more time I'd test time-sensitive tasks (e.g. medication that must happen at a fixed hour),multiple pets competing for the same budget more aggressively, and rollover past
midnight when the budget is very large.

---

## 5. Reflection

**a. What went well**

Separating the domain logic (`pawpal_system.py`) from the UI (`app.py`) so the same scheduler powers both the CLI demo and the Streamlit app, and so it could be tested without Streamlit at all.

**b. What you would improve**

I'd add fixed-time tasks (so medication can be pinned to 8:00), make the
`preferences` field actually influence the plan instead of being decorative, and give each pet its own share of the time budget instead of one shared pool.

**c. Key takeaway**

Designing the classes and relationships first (UML → stubs → logic) made the implementation much smoother, and keeping the diagram in sync with the code as the design changed kept everything honest. On the AI side, the biggest lesson was to verify suggestions by running them — the failing test caught a wrong assumption I'd have missed by just reading the code.
