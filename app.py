from datetime import time as dtime

import streamlit as st

from pawpal_ai import PawPalAIError, setup_logging, suggest_tasks
from pawpal_system import Owner, Pet, Priority, Scheduler, Task
from retriever import KnowledgeBase, build_query

setup_logging()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")
st.caption("Plan your pets' daily care around the time you actually have.")


@st.cache_resource
def load_knowledge_base() -> KnowledgeBase:
    """Load and index the care knowledge base once per server, not per rerun.

    Streamlit re-runs this script top to bottom on every interaction, so
    without the cache the whole corpus would be re-read and re-indexed on every
    click.
    """
    return KnowledgeBase.load()

PRIORITIES = ["high", "medium", "low"]
PRIORITY_ENUM = {"high": Priority.HIGH, "medium": Priority.MEDIUM, "low": Priority.LOW}

# --- Step 2: the session "vault" -------------------------------------------
# Streamlit re-runs this whole script on every interaction. If we created the
# Owner here unconditionally it would be reborn empty each time, losing all the
# pets and tasks. So we create it ONCE and keep the SAME instance in
# st.session_state (a dict-like store that survives reruns), checking whether
# it already exists before making a new one.
if "owner" not in st.session_state:
    st.session_state.owner = Owner(name="Jordan", available_minutes=200)

owner = st.session_state.owner  # the persisted instance we mutate below

# --- owner + constraints ---------------------------------------------------
st.subheader("1. Owner & time budget")
col_a, col_b, col_c = st.columns(3)
with col_a:
    owner.name = st.text_input("Owner name", value=owner.name)
with col_b:
    owner.available_minutes = st.number_input(
        "Time available today (min)", min_value=0, max_value=1440,
        value=owner.available_minutes, step=5,
    )
with col_c:
    start_hour = st.number_input("Start hour", min_value=0, max_value=23, value=8)

st.divider()

# --- Step 3: add a pet -> Owner.add_pet() ----------------------------------
st.subheader("2. Pets")
with st.form("add_pet", clear_on_submit=True):
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        new_pet_name = st.text_input("Pet name", value="")
    with pc2:
        new_pet_species = st.selectbox("Species", ["dog", "cat", "other"])
    with pc3:
        new_pet_breed = st.text_input("Breed", value="")
    pc4, pc5 = st.columns(2)
    with pc4:
        new_pet_color = st.text_input("Color", value="")
    with pc5:
        new_pet_age = st.number_input("Age (years)", min_value=0, max_value=40, value=0)

    if st.form_submit_button("Add pet"):
        name = new_pet_name.strip()
        existing = {p.name for p in owner.pets}
        if not name:
            st.warning("Please enter a pet name.")
        elif name in existing:
            st.warning(f"You already have a pet named {name}.")
        else:
            # The method on the class handles the data; the rerun after this
            # submit re-reads owner.pets and shows the new pet automatically.
            owner.add_pet(
                Pet(
                    name=name,
                    species=new_pet_species,
                    breed=new_pet_breed,
                    color=new_pet_color,
                    age=int(new_pet_age),
                )
            )

if not owner.pets:
    st.info("Add at least one pet to get started.")

# --- Step 3: add a task -> Pet.add_task() ----------------------------------
if owner.pets:
    st.subheader("3. Tasks")
    with st.form("add_task", clear_on_submit=True):
        tc1, tc2, tc3 = st.columns([2, 2, 1])
        with tc1:
            task_pet_name = st.selectbox("For pet", [p.name for p in owner.pets])
        with tc2:
            task_title = st.text_input("Task", value="Morning walk")
        with tc3:
            task_duration = st.number_input("Min", min_value=1, max_value=240, value=20)
        tc4, tc5 = st.columns(2)
        with tc4:
            task_priority = st.selectbox("Priority", PRIORITIES, index=0)
        with tc5:
            task_frequency = st.selectbox("Repeats", ["once", "daily", "weekly"], index=0)
        tc6, tc7 = st.columns([1, 2])
        with tc6:
            # NOTE: don't gate the time input with disabled=. Widgets inside a
            # st.form don't rerun on change, so a disabled= that depends on this
            # checkbox would never re-enable until submit. Instead the input is
            # always editable and the checkbox just decides whether we apply it.
            task_timed = st.checkbox("Set a time")
        with tc7:
            # st.time_input yields a real time object, so we always store a
            # zero-padded "HH:MM" string that sorts and compares correctly.
            task_time_val = st.time_input("Time of day", value=dtime(9, 0))

        if st.form_submit_button("Add task") and task_title.strip():
            pet = next(p for p in owner.pets if p.name == task_pet_name)
            pet.add_task(
                Task(
                    title=task_title.strip(),
                    duration_minutes=int(task_duration),
                    priority=PRIORITY_ENUM[task_priority],
                    frequency=task_frequency,
                    time=task_time_val.strftime("%H:%M") if task_timed else "",
                )
            )

    # show current pets + their tasks (read straight off the Owner object)
    for pet in owner.pets:
        with st.expander(f"🐾 {pet.name} ({pet.species}) — {len(pet.tasks)} task(s)", expanded=True):
            traits = []
            if pet.breed:
                traits.append(f"Breed: {pet.breed}")
            if pet.color:
                traits.append(f"Color: {pet.color}")
            if pet.age:
                traits.append(f"Age: {pet.age} yr")
            if traits:
                st.caption(" · ".join(traits))

            if pet.tasks:
                for idx, t in enumerate(pet.tasks):
                    r1, r2, r3 = st.columns([5, 3, 2])
                    done = t.status == "complete"
                    title = f"~~{t.title}~~" if done else t.title
                    r1.markdown(f"{'✅' if done else '⬜'} {title}")
                    meta = f"{t.duration_minutes} min · {t.priority.value}"
                    if t.time:
                        meta += f" · 🕐 {t.time}"
                    if t.frequency != "once":
                        meta += f" · 🔁 {t.frequency}"
                    r2.caption(meta)
                    if done:
                        r3.caption("done")
                    elif r3.button("Mark done", key=f"done_{pet.name}_{idx}"):
                        # completes the task; recurring tasks auto-spawn the next one
                        nxt = pet.complete_task(t)
                        if nxt is not None:
                            st.toast(f"Next {t.title} scheduled for {nxt.due_date}")
                        st.rerun()

                    # --- edit ("fix") an already-added task ----------------
                    with st.expander("✏️ Edit", expanded=False):
                        with st.form(f"edit_{pet.name}_{idx}", clear_on_submit=False):
                            e_title = st.text_input("Task", value=t.title)
                            ec1, ec2 = st.columns(2)
                            with ec1:
                                e_duration = st.number_input(
                                    "Min", min_value=1, max_value=240,
                                    value=t.duration_minutes,
                                )
                            with ec2:
                                e_priority = st.selectbox(
                                    "Priority", PRIORITIES,
                                    index=PRIORITIES.index(t.priority.value),
                                )
                            ec3, ec4 = st.columns(2)
                            with ec3:
                                e_frequency = st.selectbox(
                                    "Repeats", ["once", "daily", "weekly"],
                                    index=["once", "daily", "weekly"].index(t.frequency),
                                )
                            with ec4:
                                e_timed = st.checkbox(
                                    "Set a time", value=bool(t.time),
                                    key=f"edit_timed_{pet.name}_{idx}",
                                )
                            # Always editable (see note in the add-task form):
                            # the checkbox decides whether the time is applied.
                            e_time_val = st.time_input(
                                "Time of day",
                                value=(
                                    dtime(*map(int, t.time.split(":")))
                                    if t.time else dtime(9, 0)
                                ),
                            )
                            if st.form_submit_button("Save changes") and e_title.strip():
                                # One validated method updates the task in place.
                                t.update(
                                    title=e_title.strip(),
                                    duration_minutes=int(e_duration),
                                    priority=PRIORITY_ENUM[e_priority],
                                    frequency=e_frequency,
                                    time=e_time_val.strftime("%H:%M") if e_timed else "",
                                )
                                st.toast(f"Updated '{t.title}'.")
                                st.rerun()
                if st.button(f"Clear {pet.name}'s tasks", key=f"clear_{pet.name}"):
                    pet.tasks.clear()
                    st.rerun()
            else:
                st.caption("No tasks yet.")

    # --- AI care suggestions (the RAG layer) -------------------------------
    # Retrieval happens first and the retrieved chunks are the only care
    # guidance the model gets, so the sources shown here are what actually
    # produced the durations and priorities below.
    st.divider()
    st.subheader("4. AI care suggestions")
    st.caption(
        "Looks up care guidance for this pet, then drafts tasks from it. "
        "Every task cites the source it came from — anything the model can't "
        "ground in a retrieved source is rejected before it reaches your plan."
    )

    kb = load_knowledge_base()

    ai_col1, ai_col2 = st.columns([2, 1])
    with ai_col1:
        ai_pet_name = st.selectbox(
            "Suggest care tasks for", [p.name for p in owner.pets], key="ai_pet"
        )
    ai_pet = next(p for p in owner.pets if p.name == ai_pet_name)
    with ai_col2:
        st.caption("Sources it will search:")
        st.caption(
            ", ".join(c.id for c in kb.search(build_query(ai_pet, owner), top_k=4)) or "—"
        )

    if st.button(f"🤖 Suggest tasks for {ai_pet.name}", type="secondary"):
        with st.spinner(f"Retrieving care guidance for {ai_pet.name}…"):
            try:
                st.session_state[f"suggestion_{ai_pet.name}"] = suggest_tasks(
                    ai_pet, owner, kb=kb
                )
            except PawPalAIError as exc:
                # Expected, user-fixable failures (no API key, offline, rate
                # limited) are shown as a message rather than a stack trace.
                st.error(str(exc))
            except Exception:
                st.exception(RuntimeError("Unexpected error — see pawpal.log for details."))

    result = st.session_state.get(f"suggestion_{ai_pet.name}")
    if result is not None:
        if result.sources:
            with st.expander("📚 Sources retrieved for this pet", expanded=False):
                for chunk in result.sources:
                    st.markdown(f"**{chunk.title}** · `{chunk.id}`")
                    st.caption(chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""))

        if result.coverage_note:
            st.info(result.coverage_note)

        # Guardrail transparency: show what was thrown away and why.
        for reason in result.rejected:
            st.warning(f"🛡️ {reason}")

        if result.suggestions:
            st.markdown("**Suggested tasks** — untick anything you don't want.")
            with st.form(f"accept_{ai_pet.name}"):
                keep = []
                for idx, s in enumerate(result.suggestions):
                    t = s.task
                    meta = f"{t.duration_minutes} min · {t.priority.value} · {t.frequency}"
                    if t.time:
                        meta += f" · 🕐 {t.time}"
                    keep.append(
                        st.checkbox(f"**{t.title}** — {meta}", value=True, key=f"keep_{ai_pet.name}_{idx}")
                    )
                    st.caption(f"↳ {s.rationale}  \n*Source: {s.source_title}* (`{s.source_id}`)")

                if st.form_submit_button("Add selected tasks", type="primary"):
                    added = 0
                    existing = {t.title.strip().lower() for t in ai_pet.tasks}
                    for wanted, s in zip(keep, result.suggestions):
                        if wanted and s.task.title.strip().lower() not in existing:
                            ai_pet.add_task(s.task)
                            existing.add(s.task.title.strip().lower())
                            added += 1
                    # Drop the suggestion so the list doesn't linger and let the
                    # user add the same tasks twice.
                    st.session_state.pop(f"suggestion_{ai_pet.name}", None)
                    st.toast(f"Added {added} task(s) to {ai_pet.name}.")
                    st.rerun()

            st.caption(
                f"Tokens used: {result.input_tokens} in / {result.output_tokens} out"
            )
        elif not result.rejected:
            st.caption("No tasks suggested for this pet.")

    # --- timeline + conflicts (both straight from the Scheduler) -----------
    st.divider()
    st.subheader("5. Timeline & conflicts")

    # Conflict detection: the Scheduler flags any duplicate time slots for us.
    conflicts = Scheduler().detect_conflicts(owner)
    if conflicts:
        for warning in conflicts:
            st.warning(warning)
    else:
        st.success("No time conflicts — every scheduled time is unique.")

    # Chronological view: Scheduler.sort_by_time orders timed tasks and pushes
    # untimed ones to the end. We map each task back to its pet for display.
    pet_of = {id(task): pet for pet in owner.pets for task in pet.tasks}
    ordered = Scheduler.sort_by_time(owner.filter_tasks())
    if ordered:
        st.table(
            [
                {
                    "Time": task.time or "—",
                    "Task": task.title,
                    "Pet": pet_of[id(task)].name,
                    "Min": task.duration_minutes,
                    "Priority": task.priority.value,
                    "Status": "✅ done" if task.status == "complete" else "pending",
                }
                for task in ordered
            ]
        )
    else:
        st.info("Add a task to see the daily timeline.")

st.divider()

# --- generate the plan -----------------------------------------------------
st.subheader("6. Today's plan")
st.caption("Tasks are scheduled highest-priority first and packed into your time budget.")

if st.button("Generate schedule", type="primary", disabled=not owner.pets):
    # The Owner already holds everything the scheduler needs.
    plan = Scheduler(start_hour=int(start_hour)).build_plan(owner)

    m1, m2, m3 = st.columns(3)
    m1.metric("Time used", f"{plan.total_minutes} min")
    m2.metric("Scheduled", len(plan.items))
    m3.metric("Skipped", len(plan.skipped))

    st.caption(
        "Only tasks due today (or undated/overdue) are planned — completed tasks "
        "and future occurrences of recurring tasks are left out."
    )

    if plan.items:
        st.markdown("#### ✅ Scheduled")
        st.table(
            [
                {
                    "Time": i.start_time,
                    "Task": i.task.title,
                    "Pet": i.pet.name,
                    "Min": i.task.duration_minutes,
                    "Priority": i.task.priority.value,
                    "Repeats": "once" if i.task.frequency == "once" else f"🔁 {i.task.frequency}",
                }
                for i in plan.items
            ]
        )
    else:
        st.warning("Nothing fit in the available time. Try increasing your time budget.")

    if plan.skipped:
        st.markdown("#### ⏭️ Skipped (ran out of time)")
        st.table(
            [
                {
                    "Task": i.task.title,
                    "Pet": i.pet.name,
                    "Min": i.task.duration_minutes,
                    "Priority": i.task.priority.value,
                    "Why": i.reason,
                }
                for i in plan.skipped
            ]
        )

    with st.expander("Why this plan? (full explanation)"):
        st.code(plan.explain(), language="text")
