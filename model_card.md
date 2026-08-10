# Model Card — PawPal+

This document is the responsible-AI reflection for PawPal+'s AI feature: a
Retrieval-Augmented Generation layer that drafts daily pet-care tasks from a
small hand-written knowledge base, then hands them to a deterministic
scheduler. For the project overview, architecture, and setup instructions,
see [`README.md`](README.md).

**Backends in use:** `llama3.2:3b` via [Ollama](https://ollama.com) (local,
free, default) or `claude-opus-5` via the Anthropic API (paid, opt-in). The
retrieval, validation, and scheduling logic is identical for both — only the
generation call differs. All findings below were observed against both
backends unless a section says otherwise.

---

## Limitations and biases

**The knowledge base is 14 files I wrote myself.** It's not sourced from a
licensed veterinarian or a peer-reviewed reference — it's illustrative
care-guidance prose I authored for this project, styled to read like
professional advice. It could contain my own gaps, oversimplifications, or
outright errors, and the system has no way to detect that: the grounding
check guarantees a suggested task traces back to *something written in the
corpus*, not that the corpus itself is correct. This is the single biggest
limitation of the system and the one I'd fix first before any real-world use.

**Retrieval is keyword-based, not semantic.** The TF-IDF search
(`retriever.py`) matches on tokenized words and a light singularization rule.
A pet described with vocabulary the corpus doesn't use — a misspelled breed,
a species outside `dog`/`cat` (I tested `"rabbit"` during development; it
silently fell back to generic sources with no rabbit-specific guidance and no
warning that coverage was thin), or a synonym the tokenizer doesn't
normalize — gets weaker or wrong retrieval with no visible signal to the user
that the match was poor.

**Quality is backend-dependent.** The free local model (`llama3.2:3b`) is
measurably weaker at following instructions than the paid option — it needed
an explicit minimum-task-count rule and a self-correction retry to reliably
produce a full routine (see Testing Summary in the README), and it still
occasionally proposes a slightly redundant task even with those fixes. Two
people running "the same" app can get different-quality plans purely based on
which backend they chose, which is a real inconsistency I'm choosing to
accept in exchange for the app being runnable at zero cost.

**No individual personalization beyond species, breed, and age.** The system
has no concept of a specific pet's medical history, medication interactions,
disabilities, or an owner's own physical limitations. Two labradors of the
same age get the same suggestions regardless of one having a knee injury.

**Guardrails check structure, not truth.** `_validate()` enforces that a
citation exists, a duration is in bounds, priorities and frequencies are
valid, and there are no duplicates. None of that checks whether the
underlying *veterinary claim* is correct — only that the model didn't invent
it. Structural soundness and factual correctness are different guarantees,
and this project only provides the first one.

---

## Could this be misused, and how is that prevented?

The design already closes off the most obvious misuse path: there is no open
chat interface. The model never sees free-form user text beyond a few
structured fields (pet name, breed, color) — it only ever sees a fixed system
prompt, the retrieved sources, and the pet/owner profile. That's a small
surface area for prompt injection compared to a general-purpose chatbot
wrapped around the same feature.

Realistic misuse I considered:

- **Substituting for real veterinary care.** A user could treat AI-generated
  task suggestions as medical guidance and delay a vet visit for something
  urgent. Mitigation already in place: `knowledge/vet-checkups.md` explicitly
  tells the model that sudden or urgent symptoms are "a same-day vet call,
  not a scheduled task," and the system prompt forbids proposing anything
  outside a normal day's routine. What's *not* yet in place: an explicit,
  persistent disclaimer surfaced in the UI itself. Right now that safety
  framing lives only inside the knowledge base's prose, which the user never
  directly reads — a real gap I'd close before shipping this beyond a class
  project.
- **Unsafe advice reaching an animal.** A hallucinated or out-of-bounds
  duration (e.g., a multi-hour walk) is blocked structurally: `_validate()`
  rejects anything outside 1–240 minutes and anything not tied to a
  retrieved source, regardless of what either model backend outputs. This
  guarantee holds even if the underlying model is replaced or misbehaves.
- **Cost abuse on the paid backend.** Nothing currently rate-limits repeated
  `Suggest tasks` clicks against the Anthropic API. For a class project this
  is a non-issue (the owner holds their own key), but a production version
  serving multiple users would need per-user or per-session rate limiting
  that doesn't exist today.
- **Free-text field injection.** A pet's breed or name field could contain
  text crafted to influence the model's output (e.g., embedding instructions
  in a "breed" field). I have not specifically tested this attack, and there
  is no input sanitization on those fields beyond what the retrieval query
  builder happens to tokenize. This is an honest gap, not a solved problem.

---

## What surprised me while testing reliability

**Grounding worked better than I expected, and something much simpler didn't
work at all.** I went into this expecting hallucinated citations — the model
inventing a source id that was never retrieved — to be the main reliability
risk, and built the grounding check specifically for that. Across every real
run I made against both backends, I never once observed a fabricated
citation. What actually broke was almost embarrassingly simple: the model
sometimes just... stopped after proposing one task, despite the prompt asking
for a full day's routine. The sophisticated safeguard held up perfectly; the
basic requirement ("propose more than one thing") was what needed real
engineering effort to guarantee.

**Retrying at `temperature: 0` does nothing on its own — and I only learned
that from a bug report, not from writing the code.** I'd set the local
backend to temperature 0 for reproducible test runs, which quietly meant a
naive "just call the model again" retry would return the *exact same*
too-short answer every time, since a deterministic model given an identical
prompt makes an identical choice. This wasn't obvious to me until it became a
real reported bug ("it's only, like, one suggestion") and I found the log
line `1 proposed -> 1 accepted` repeating identically across separate runs
for the same pet. The eventual fix had to change what the second prompt
*said* (list what was already covered, ask explicitly for more), not just
re-issue the request.

**Testing an LLM feature needs two different kinds of tests, and I only
planned for one at first.** My instinct going in was to write deterministic
unit tests with a fake model client — fast, free, and they do catch real
logic bugs (they caught a species-contamination bug in retrieval before any
user saw it). But the one-task bug above was invisible to that kind of test,
because the fake client always returns whatever a test tells it to — it
can't reproduce a real model quietly under-delivering. I only found that bug
by reading `pawpal.log` from actual usage. Both kinds of testing turned out
to be load-bearing; neither alone would have caught everything that shipped
with a bug.

---

## Collaboration with AI

I built this project working with Claude (Claude Code) as a pair-programming
collaborator across the RAG feature, the free local-model backend, the
two-phase scheduler, and the bug fixes described above. Two concrete
examples, one in each direction:

**Helpful: the free Ollama backend.** When I said I wanted to try the
project without paying for API access, Claude added a second, fully
interchangeable model backend rather than just pointing me at a cheaper
Anthropic model. It correctly identified that the retrieval, validation, and
scheduling code didn't need to change at all — only the one function that
calls the model needed a second implementation — and it caught a real trap on
its own: the unedited `.env.example` placeholder key (`sk-ant-...`) is
non-empty, so a naive "is a key set?" check would have routed to the paid
API and failed with a confusing authentication error instead of falling back
to the free option. That specific edge case wasn't something I asked for; it
was found and fixed proactively, and it's the kind of detail that determines
whether "free by default" actually works for someone who just clones the
repo.

**Flawed: the `temperature: 0` default on the local backend.** Claude set
this early, with the stated reasoning "deterministic-ish, for
reproducibility" — a defensible-sounding choice for a project with a test
suite. What it didn't account for is that determinism cuts both ways: it also
means a weak answer is a *permanently* weak answer for that exact prompt, so
the simplest possible fix (retry) was quietly useless from the moment that
default was set. This wasn't caught by review or by tests — it surfaced only
because I hit it in real usage and reported it, and diagnosing it took
reading raw logs to notice the output was *identical*, not just similarly
short, across repeated runs. It's a good example of an AI suggestion that
sounded reasonable, had a real justification attached, and still had a
consequence its author (Claude) didn't fully think through until it actually
broke something.

A third thing worth naming honestly, since it happened during this same
project and is directly relevant to this section: while committing a diagram
update, Claude ran `git add -A` and committed without first checking full
`git status` for unrelated changes — which silently swept up and deleted two
UML diagram files from the repository history. It was caught (by me
reviewing this document's requirements, which prompted a fresh look at the
repo state) and reversed before any real damage was done, but it's a clean
example of exactly the kind of oversight this whole reflection is about:
an AI assistant given broad tool access (here, git) can take an
unreviewed, hard-to-notice destructive action, and the fix is the same as
for any other collaborator — check the diff before you commit it, every
time, not just when something looks unusual.
