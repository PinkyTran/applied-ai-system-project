"""Tests for the PawPal+ scheduling system.

Run from the repo root with: pytest
"""

from datetime import date, timedelta

import pytest

from pawpal_system import DailyPlan, Owner, Pet, Priority, Scheduler, Task


# --- helpers ---------------------------------------------------------------

def make_owner(available_minutes, tasks_by_pet):
    """Build an Owner whose pets carry the given tasks.

    tasks_by_pet: list of (pet_name, [Task, ...]) pairs.
    """
    pets = [Pet(name, "dog", tasks=tasks) for name, tasks in tasks_by_pet]
    return Owner("Owner", available_minutes=available_minutes, pets=pets)


# --- Task ------------------------------------------------------------------

def test_rank_orders_high_above_medium_above_low():
    assert Task("a", 10, Priority.HIGH).rank() > Task("b", 10, Priority.MEDIUM).rank()
    assert Task("b", 10, Priority.MEDIUM).rank() > Task("c", 10, Priority.LOW).rank()


def test_task_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        Task("bad", 0, Priority.HIGH)
    with pytest.raises(ValueError):
        Task("bad", -5, Priority.LOW)


def test_mark_complete_changes_status():
    """Task Completion: mark_complete() flips the task's status."""
    task = Task("Feeding", 10, Priority.HIGH)
    assert task.status == "pending"
    task.mark_complete()
    assert task.status == "complete"


def test_update_changes_only_given_fields():
    """Editing a task changes the supplied fields and leaves the rest intact."""
    t = Task("Walk", 30, Priority.LOW, time="08:00", frequency="once")
    t.update(title="Long walk", duration_minutes=45, priority=Priority.HIGH)
    assert t.title == "Long walk"
    assert t.duration_minutes == 45
    assert t.priority is Priority.HIGH
    assert t.time == "08:00"        # untouched
    assert t.frequency == "once"    # untouched


def test_update_can_clear_time_and_rejects_bad_duration():
    """Edge: time can be cleared to ''; a non-positive duration is rejected."""
    t = Task("Meds", 5, Priority.HIGH, time="09:00")
    t.update(time="")
    assert t.time == ""
    with pytest.raises(ValueError):
        t.update(duration_minutes=0)
    assert t.duration_minutes == 5  # unchanged after the failed update


def test_adding_task_increases_pet_task_count():
    """Task Addition: add_task() grows the pet's task list by one."""
    pet = Pet("Biscuit", "dog")
    assert len(pet.tasks) == 0
    pet.add_task(Task("Morning walk", 30, Priority.HIGH))
    assert len(pet.tasks) == 1


# --- next_occurrence (recurrence date math) --------------------------------

def test_next_occurrence_daily_adds_one_day():
    t = Task("Meds", 5, Priority.HIGH, frequency="daily", due_date=date(2026, 7, 7))
    assert t.next_occurrence().due_date == date(2026, 7, 8)


def test_next_occurrence_weekly_adds_seven_days():
    t = Task("Bath", 40, Priority.LOW, frequency="weekly", due_date=date(2026, 7, 7))
    assert t.next_occurrence().due_date == date(2026, 7, 14)


def test_next_occurrence_once_returns_none():
    t = Task("Vet visit", 60, Priority.HIGH, frequency="once", due_date=date(2026, 7, 7))
    assert t.next_occurrence() is None


def test_next_occurrence_default_frequency_is_once():
    # Not specifying frequency should behave as non-repeating.
    assert Task("Walk", 30, Priority.HIGH).next_occurrence() is None


def test_next_occurrence_rolls_over_month_end():
    t = Task("Meds", 5, Priority.HIGH, frequency="daily", due_date=date(2026, 1, 31))
    assert t.next_occurrence().due_date == date(2026, 2, 1)


def test_next_occurrence_rolls_over_year_end():
    t = Task("Meds", 5, Priority.HIGH, frequency="daily", due_date=date(2026, 12, 31))
    assert t.next_occurrence().due_date == date(2027, 1, 1)


def test_next_occurrence_with_no_due_date_uses_today():
    from datetime import date as real_date, timedelta
    t = Task("Meds", 5, Priority.HIGH, frequency="daily")  # due_date is None
    assert t.next_occurrence().due_date == real_date.today() + timedelta(days=1)


def test_next_occurrence_copies_attributes_and_resets_status():
    t = Task("Meds", 5, Priority.HIGH, category="health", time="09:00",
             frequency="daily", due_date=date(2026, 7, 7))
    t.mark_complete()
    nxt = t.next_occurrence()
    assert nxt.title == "Meds"
    assert nxt.duration_minutes == 5
    assert nxt.priority is Priority.HIGH
    assert nxt.category == "health"
    assert nxt.time == "09:00"
    assert nxt.frequency == "daily"
    assert nxt.status == "pending"  # the fresh occurrence is NOT complete


# --- complete_task (recurrence side effects) -------------------------------

def test_complete_task_recurring_appends_next_occurrence():
    pet = Pet("Rex", "dog")
    t = Task("Meds", 5, Priority.HIGH, frequency="daily", due_date=date(2026, 7, 7))
    pet.add_task(t)
    upcoming = pet.complete_task(t)
    assert t.status == "complete"            # original marked done
    assert upcoming is not None
    assert upcoming in pet.tasks             # next occurrence added
    assert len(pet.tasks) == 2
    assert upcoming.due_date == date(2026, 7, 8)


def test_complete_task_once_does_not_add_anything():
    pet = Pet("Rex", "dog")
    t = Task("Vet visit", 60, Priority.HIGH, frequency="once")
    pet.add_task(t)
    upcoming = pet.complete_task(t)
    assert t.status == "complete"
    assert upcoming is None
    assert len(pet.tasks) == 1               # nothing spawned


# --- Scheduler helpers -----------------------------------------------------

def test_collect_tasks_flattens_across_pets():
    owner = make_owner(
        100,
        [
            ("Biscuit", [Task("walk", 30, Priority.HIGH)]),
            ("Mochi", [Task("meds", 5, Priority.HIGH), Task("play", 15, Priority.LOW)]),
        ],
    )
    collected = Scheduler()._collect_tasks(owner)
    assert len(collected) == 3
    # each entry is a (pet, task) pair
    pets = {pet.name for pet, _ in collected}
    assert pets == {"Biscuit", "Mochi"}


def test_sort_tasks_by_priority_then_duration():
    tasks = [
        Task("low-short", 5, Priority.LOW),
        Task("high-long", 40, Priority.HIGH),
        Task("high-short", 10, Priority.HIGH),
        Task("med", 20, Priority.MEDIUM),
    ]
    pet = Pet("P", "dog")
    ordered = Scheduler()._sort_tasks([(pet, t) for t in tasks])
    titles = [t.title for _, t in ordered]
    # HIGH first (shorter before longer), then MEDIUM, then LOW
    assert titles == ["high-short", "high-long", "med", "low-short"]


def test_sort_is_stable_for_full_ties():
    """Same priority and duration -> original order is preserved (deterministic)."""
    pet = Pet("P", "dog")
    tasks = [Task("first", 10, Priority.HIGH), Task("second", 10, Priority.HIGH)]
    ordered = Scheduler()._sort_tasks([(pet, t) for t in tasks])
    assert [t.title for _, t in ordered] == ["first", "second"]


# --- build_plan ------------------------------------------------------------

def test_high_priority_scheduled_before_low():
    owner = make_owner(
        100,
        [("P", [Task("low", 10, Priority.LOW), Task("high", 10, Priority.HIGH)])],
    )
    plan = Scheduler().build_plan(owner)
    assert [i.task.title for i in plan.items] == ["high", "low"]


def test_tasks_over_budget_are_skipped_with_reason():
    owner = make_owner(
        20,
        [("P", [Task("fits", 20, Priority.HIGH), Task("toobig", 30, Priority.HIGH)])],
    )
    plan = Scheduler().build_plan(owner)
    assert [i.task.title for i in plan.items] == ["fits"]
    assert [i.task.title for i in plan.skipped] == ["toobig"]
    assert "30 min" in plan.skipped[0].reason and "0 min" in plan.skipped[0].reason


def test_greedy_fills_leftover_time_with_smaller_task():
    """A big task that doesn't fit is skipped, but a smaller later task still gets in."""
    owner = make_owner(
        25,
        [(
            "P",
            [
                Task("big", 20, Priority.MEDIUM),   # scheduled (5 left)
                Task("huge", 30, Priority.MEDIUM),  # skipped, doesn't fit
                Task("tiny", 5, Priority.MEDIUM),   # still fits in the leftover 5
            ],
        )],
    )
    plan = Scheduler().build_plan(owner)
    scheduled = [i.task.title for i in plan.items]
    assert "tiny" in scheduled
    assert [i.task.title for i in plan.skipped] == ["huge"]


def test_total_minutes_sums_scheduled_only():
    owner = make_owner(
        50,
        [("P", [Task("a", 20, Priority.HIGH), Task("b", 20, Priority.HIGH),
                Task("c", 40, Priority.LOW)])],
    )
    plan = Scheduler().build_plan(owner)
    assert plan.total_minutes == 40  # a + b; c skipped


def test_start_times_are_sequential_from_start_hour():
    # Equal durations -> stable order, so timing is back-to-back and unambiguous.
    owner = make_owner(
        60,
        [("P", [Task("a", 30, Priority.HIGH), Task("b", 30, Priority.HIGH)])],
    )
    plan = Scheduler(start_hour=8).build_plan(owner)
    assert plan.items[0].start_time == "08:00"
    assert plan.items[1].start_time == "08:30"


def test_zero_budget_skips_everything():
    owner = make_owner(0, [("P", [Task("a", 10, Priority.HIGH)])])
    plan = Scheduler().build_plan(owner)
    assert plan.items == []
    assert len(plan.skipped) == 1
    assert plan.total_minutes == 0


def test_plan_excludes_completed_tasks():
    """A completed task is not scheduled into today's plan at all."""
    done = Task("done", 10, Priority.HIGH)
    done.mark_complete()
    owner = make_owner(100, [("P", [done, Task("todo", 10, Priority.HIGH)])])
    plan = Scheduler().build_plan(owner)
    titles = [i.task.title for i in plan.items] + [i.task.title for i in plan.skipped]
    assert titles == ["todo"]


def test_plan_excludes_future_dated_recurring_occurrence():
    """A recurring task's next (future) occurrence stays out of today's plan."""
    today = date(2026, 7, 7)
    pet = Pet("Rex", "dog")
    meds = Task("meds", 5, Priority.HIGH, frequency="daily", due_date=today)
    pet.add_task(meds)
    owner = Owner("O", available_minutes=100, pets=[pet])

    # Completing it marks the original done and spawns tomorrow's occurrence.
    pet.complete_task(meds)
    plan = Scheduler().build_plan(owner, on_date=today)
    # Neither the completed original nor the future occurrence should appear.
    assert plan.items == [] and plan.skipped == []


def test_plan_includes_undated_and_overdue_tasks():
    """Undated tasks and overdue (past-due) tasks are still scheduled today."""
    today = date(2026, 7, 7)
    overdue = Task("overdue", 10, Priority.HIGH, due_date=today - timedelta(days=3))
    undated = Task("undated", 10, Priority.HIGH)
    owner = make_owner(100, [("P", [overdue, undated])])
    plan = Scheduler().build_plan(owner, on_date=today)
    assert {i.task.title for i in plan.items} == {"overdue", "undated"}


def test_owner_with_no_tasks_produces_empty_plan():
    owner = Owner("Solo", available_minutes=100, pets=[Pet("P", "dog")])
    plan = Scheduler().build_plan(owner)
    assert plan.items == [] and plan.skipped == []
    assert isinstance(plan, DailyPlan)


# --- sort_by_time ----------------------------------------------------------

def test_sort_by_time_orders_chronologically():
    """Happy path: timed tasks come back in time-of-day order."""
    tasks = [
        Task("noon", 10, Priority.LOW, time="12:00"),
        Task("morning", 10, Priority.LOW, time="08:30"),
        Task("evening", 10, Priority.LOW, time="18:00"),
    ]
    ordered = Scheduler.sort_by_time(tasks)
    assert [t.title for t in ordered] == ["morning", "noon", "evening"]


def test_sort_by_time_puts_untimed_tasks_last():
    """Edge: tasks with no time fall to the end, not the front."""
    tasks = [
        Task("untimed", 10, Priority.LOW),           # time == ""
        Task("early", 10, Priority.LOW, time="07:00"),
    ]
    ordered = Scheduler.sort_by_time(tasks)
    assert [t.title for t in ordered] == ["early", "untimed"]


def test_sort_by_time_does_not_mutate_input():
    """Edge: sorting returns a new list and leaves the caller's list alone."""
    tasks = [
        Task("late", 10, Priority.LOW, time="20:00"),
        Task("early", 10, Priority.LOW, time="06:00"),
    ]
    original_order = list(tasks)
    Scheduler.sort_by_time(tasks)
    assert tasks == original_order


def test_sort_by_time_empty_list():
    """Edge: no tasks -> empty list, no error."""
    assert Scheduler.sort_by_time([]) == []


# --- detect_conflicts (two tasks at the same time) -------------------------

def test_detect_conflicts_none_when_times_differ():
    """Happy path: distinct time slots -> no warnings."""
    owner = make_owner(
        100,
        [("P", [Task("a", 10, Priority.HIGH, time="08:00"),
                Task("b", 10, Priority.HIGH, time="09:00")])],
    )
    assert Scheduler().detect_conflicts(owner) == []


def test_detect_conflicts_flags_same_time_same_pet():
    """Edge: two tasks on one pet share a slot -> one warning naming both."""
    owner = make_owner(
        100,
        [("Rex", [Task("walk", 10, Priority.HIGH, time="08:00"),
                  Task("meds", 5, Priority.HIGH, time="08:00")])],
    )
    warnings = Scheduler().detect_conflicts(owner)
    assert len(warnings) == 1
    assert "08:00" in warnings[0]
    assert "walk" in warnings[0] and "meds" in warnings[0]


def test_detect_conflicts_flags_same_time_across_pets():
    """Edge: the clash spans two different pets, not just one."""
    owner = make_owner(
        100,
        [("Rex", [Task("walk", 10, Priority.HIGH, time="07:30")]),
         ("Mochi", [Task("feed", 5, Priority.HIGH, time="07:30")])],
    )
    warnings = Scheduler().detect_conflicts(owner)
    assert len(warnings) == 1
    assert "Rex" in warnings[0] and "Mochi" in warnings[0]


def test_detect_conflicts_ignores_untimed_tasks():
    """Edge: untimed tasks can never conflict, even if there are several."""
    owner = make_owner(
        100,
        [("P", [Task("a", 10, Priority.HIGH), Task("b", 10, Priority.HIGH)])],
    )
    assert Scheduler().detect_conflicts(owner) == []


# --- filter_tasks ----------------------------------------------------------

def test_filter_tasks_returns_everything_with_no_filters():
    """Happy path: no filters -> every task across every pet."""
    owner = make_owner(
        100,
        [("Rex", [Task("a", 10, Priority.HIGH)]),
         ("Mochi", [Task("b", 10, Priority.LOW), Task("c", 5, Priority.LOW)])],
    )
    assert len(owner.filter_tasks()) == 3


def test_filter_tasks_by_status_and_pet():
    """Edge: filters combine (AND) across status and pet name."""
    done = Task("done", 10, Priority.HIGH)
    done.mark_complete()
    owner = make_owner(
        100,
        [("Rex", [done, Task("pending", 10, Priority.HIGH)]),
         ("Mochi", [Task("other", 10, Priority.LOW)])],
    )
    assert [t.title for t in owner.filter_tasks(status="complete")] == ["done"]
    assert {t.title for t in owner.filter_tasks(pet_name="Rex")} == {"done", "pending"}
    assert owner.filter_tasks(status="complete", pet_name="Mochi") == []


# --- multi-pet / empty-owner edges -----------------------------------------

def test_build_plan_owner_with_no_pets():
    """Edge: an owner with zero pets plans an empty day without erroring."""
    owner = Owner("Petless", available_minutes=100, pets=[])
    plan = Scheduler().build_plan(owner)
    assert plan.items == [] and plan.skipped == []


def test_build_plan_ranks_across_multiple_pets():
    """Happy path: ranking is global across pets, not per-pet."""
    owner = make_owner(
        100,
        [("Rex", [Task("rex-low", 10, Priority.LOW)]),
         ("Mochi", [Task("mochi-high", 10, Priority.HIGH)])],
    )
    plan = Scheduler().build_plan(owner)
    assert plan.items[0].task.title == "mochi-high"


# --- explain ---------------------------------------------------------------

def test_explain_names_pet_and_reports_budget():
    owner = make_owner(30, [("Biscuit", [Task("Morning walk", 30, Priority.HIGH)])])
    text = Scheduler().build_plan(owner).explain()
    assert "Morning walk" in text
    assert "Biscuit" in text
    assert "budget: 30 min" in text
