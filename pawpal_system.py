"""PawPal+ core system.

Implements the priority + time-budget design from diagrams/uml.mmd: an owner's
pets carry care tasks, and the Scheduler fits them into the owner's time budget.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum


class Priority(Enum):
    """How important a task is. Drives ranking in the scheduler."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Owner:
    """The pet owner: their time budget, preferences, and the pets they care for."""

    def __init__(
        self,
        name: str,
        available_minutes: int,
        preferences: list[str] | None = None,
        pets: list[Pet] | None = None,
    ):
        """Create an owner with a daily time budget, preferences, and pets."""
        self.name = name
        self.available_minutes = available_minutes
        self.preferences = preferences if preferences is not None else []
        self.pets = pets if pets is not None else []

    def add_pet(self, pet: Pet) -> None:
        """Add a pet for this owner to care for."""
        self.pets.append(pet)

    def filter_tasks(self, status: str | None = None, pet_name: str | None = None) -> list[Task]:
        """Return this owner's tasks, optionally filtered by status and/or pet name.

        Walks every pet's task list and keeps a task only if it matches all the
        filters that were supplied. Any filter left as None is ignored, so
        calling with no arguments returns every task the owner has.

        Args:
            status: keep only tasks with this status (e.g. "pending", "complete").
            pet_name: keep only tasks belonging to the pet with this name.

        Returns:
            A flat list of matching Task objects (empty if none match).
        """
        result: list[Task] = []
        for pet in self.pets:
            if pet_name is not None and pet.name != pet_name:
                continue
            for task in pet.tasks:
                if status is not None and task.status != status:
                    continue
                result.append(task)
        return result


class Pet:
    """A pet the owner cares for, along with the tasks to be done for it."""

    def __init__(
        self,
        name: str,
        species: str,
        breed: str = "",
        color: str = "",
        age: int | None = None,
        tasks: list[Task] | None = None,
    ):
        """Create a pet with its descriptive traits and starting task list."""
        self.name = name
        self.species = species
        self.breed = breed
        self.color = color
        self.age = age
        self.tasks = tasks if tasks is not None else []

    def add_task(self, task: Task) -> None:
        """Add a care task for this pet."""
        self.tasks.append(task)

    def complete_task(self, task: Task) -> Task | None:
        """Mark a task complete and, for recurring tasks, spawn the next occurrence.

        This is the completion entry point (used instead of Task.mark_complete()
        directly) because only the Pet knows the task list to add the follow-up
        occurrence to. Non-recurring tasks are simply marked done.

        Args:
            task: the task to complete (expected to already be in self.tasks).

        Returns:
            The newly created next-occurrence Task if the task repeats, else None.
        """
        task.mark_complete()
        upcoming = task.next_occurrence()
        if upcoming is not None:
            self.tasks.append(upcoming)
        return upcoming


class Task:
    """One care activity: what it is, how long it takes, how important."""

    def __init__(
        self,
        title: str,
        duration_minutes: int,
        priority: Priority,
        category: str = "",
        time: str = "",
        frequency: str = "once",
        due_date: date | None = None,
    ):
        """Create a care task; raises ValueError if the duration is not positive."""
        if duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        self.title = title
        self.duration_minutes = duration_minutes
        self.priority = priority
        self.category = category
        self.time = time  # preferred time of day, "HH:MM" (empty = no preference)
        self.frequency = frequency  # "once", "daily", or "weekly"
        self.due_date = due_date  # date this task is due (None = unscheduled)
        self.status = "pending"

    def rank(self) -> int:
        """Return a sortable integer for this task's priority (higher = more important)."""
        return {Priority.HIGH: 3, Priority.MEDIUM: 2, Priority.LOW: 1}[self.priority]

    def mark_complete(self) -> None:
        """Mark this task as done."""
        self.status = "complete"

    def update(
        self,
        title: str | None = None,
        duration_minutes: int | None = None,
        priority: Priority | None = None,
        time: str | None = None,
        frequency: str | None = None,
        category: str | None = None,
    ) -> None:
        """Edit this task's fields in place; only the arguments given are changed.

        Centralizes editing so the same validation as __init__ applies: a
        non-positive duration is rejected. Any argument left as None keeps its
        current value, so callers can change just one field. Status and due_date
        are intentionally not editable here (use mark_complete / recurrence).

        Args:
            title: new title, if changing.
            duration_minutes: new positive duration, if changing.
            priority: new Priority, if changing.
            time: new "HH:MM" time (or "" to clear), if changing.
            frequency: new "once"/"daily"/"weekly", if changing.
            category: new category, if changing.

        Raises:
            ValueError: if duration_minutes is given and is not positive.
        """
        if duration_minutes is not None:
            if duration_minutes <= 0:
                raise ValueError("duration_minutes must be positive")
            self.duration_minutes = duration_minutes
        if title is not None:
            self.title = title
        if priority is not None:
            self.priority = priority
        if time is not None:
            self.time = time
        if frequency is not None:
            self.frequency = frequency
        if category is not None:
            self.category = category

    def next_occurrence(self) -> Task | None:
        """Return a fresh Task for the next due date, or None if it doesn't repeat.

        Advances the due date with datetime.timedelta (1 day for "daily", 7 for
        "weekly"), which handles month and year rollovers correctly. The new task
        copies all attributes but starts with status "pending". If this task has
        no due_date, today's date is used as the base.

        Returns:
            A new Task due on the next date, or None for non-repeating tasks.
        """
        if self.frequency == "daily":
            delta = timedelta(days=1)
        elif self.frequency == "weekly":
            delta = timedelta(weeks=1)
        else:
            return None
        base = self.due_date or date.today()
        return Task(
            title=self.title,
            duration_minutes=self.duration_minutes,
            priority=self.priority,
            category=self.category,
            time=self.time,
            frequency=self.frequency,
            due_date=base + delta,
        )


class PlanItem:
    """A scheduler decision about a single task: which pet, included/skipped, when, and why."""

    def __init__(self, task: Task, pet: Pet, start_time: str, included: bool, reason: str):
        """Record one scheduling decision: the task, its pet, time, and reason."""
        self.task = task
        self.pet = pet
        self.start_time = start_time
        self.included = included
        self.reason = reason


class DailyPlan:
    """The result of scheduling across all of an owner's pets: what got in, what got skipped."""

    def __init__(
        self,
        owner: Owner,
        items: list[PlanItem] | None = None,
        skipped: list[PlanItem] | None = None,
        total_minutes: int = 0,
    ):
        """Hold the scheduled and skipped items for an owner's day."""
        self.owner = owner
        self.items = items if items is not None else []
        self.skipped = skipped if skipped is not None else []
        self.total_minutes = total_minutes

    def explain(self) -> str:
        """Return a human-readable explanation of the plan and its choices."""
        lines = [
            f"Daily plan for {self.owner.name} "
            f"(budget: {self.owner.available_minutes} min, used: {self.total_minutes} min)",
        ]

        lines.append("")
        if self.items:
            lines.append("Scheduled:")
            for item in self.items:
                t = item.task
                lines.append(
                    f"  {item.start_time} — {t.title} for {item.pet.name} "
                    f"({t.duration_minutes} min) [priority: {t.priority.value}] — {item.reason}"
                )
        else:
            lines.append("Scheduled: nothing fit in the available time.")

        if self.skipped:
            lines.append("")
            lines.append("Skipped:")
            for item in self.skipped:
                t = item.task
                lines.append(
                    f"  {t.title} for {item.pet.name} "
                    f"({t.duration_minutes} min) [priority: {t.priority.value}] — {item.reason}"
                )

        return "\n".join(lines)


class Scheduler:
    """Ranks tasks by priority, fits them into the owner's time budget, builds a DailyPlan."""

    def __init__(self, start_hour: int = 8):
        """Create a scheduler that starts placing tasks at start_hour."""
        self.start_hour = start_hour

    def build_plan(self, owner: Owner, on_date: date | None = None) -> DailyPlan:
        """Greedily fit the owner's ranked pet tasks into their time budget.

        Only tasks relevant to the target day are scheduled: a task is skipped
        entirely (not even listed as "skipped") if it is already complete or if
        its due_date is in the future. This keeps recurring tasks honest — after
        a daily/weekly task is completed its next occurrence is due later, so it
        does NOT clutter today's plan. Undated and overdue tasks stay eligible.

        Args:
            owner: the owner whose pets' tasks are scheduled.
            on_date: the day to plan for (defaults to today).
        """
        today = on_date or date.today()
        eligible = [
            (pet, task)
            for pet, task in self._collect_tasks(owner)
            if task.status != "complete"
            and (task.due_date is None or task.due_date <= today)
        ]
        ranked = self._sort_tasks(eligible)

        remaining = owner.available_minutes
        cursor = self.start_hour * 60  # minutes since midnight
        items: list[PlanItem] = []
        skipped: list[PlanItem] = []
        used = 0

        for pet, task in ranked:
            if task.duration_minutes <= remaining:
                items.append(
                    PlanItem(
                        task=task,
                        pet=pet,
                        start_time=self._format_time(cursor),
                        included=True,
                        reason=f"{task.priority.value} priority, fits the remaining time",
                    )
                )
                cursor += task.duration_minutes
                remaining -= task.duration_minutes
                used += task.duration_minutes
            else:
                skipped.append(
                    PlanItem(
                        task=task,
                        pet=pet,
                        start_time="",
                        included=False,
                        reason=f"needs {task.duration_minutes} min but only {remaining} min left",
                    )
                )

        return DailyPlan(owner=owner, items=items, skipped=skipped, total_minutes=used)

    def _collect_tasks(self, owner: Owner) -> list[tuple[Pet, Task]]:
        """Flatten every (pet, task) pair across all of the owner's pets."""
        return [(pet, task) for pet in owner.pets for task in pet.tasks]

    def _sort_tasks(self, pet_tasks: list[tuple[Pet, Task]]) -> list[tuple[Pet, Task]]:
        """Return (pet, task) pairs by priority desc then duration asc (stable order)."""
        return sorted(
            pet_tasks,
            key=lambda pt: (-pt[1].rank(), pt[1].duration_minutes),
        )

    @staticmethod
    def _format_time(minutes_since_midnight: int) -> str:
        """Format minutes-since-midnight as 'HH:MM' (wraps past 24h)."""
        hours, minutes = divmod(minutes_since_midnight % (24 * 60), 60)
        return f"{hours:02d}:{minutes:02d}"

    @staticmethod
    def sort_by_time(tasks: list[Task]) -> list[Task]:
        """Return tasks ordered by their 'HH:MM' time; untimed tasks sort last.

        Zero-padded 24-hour strings sort chronologically as plain text, so a
        simple string key suffices. Tasks with no time get the sentinel "99:99"
        so they fall to the end instead of the front. Does not mutate the input.

        Args:
            tasks: the tasks to order.

        Returns:
            A new list sorted by time of day.
        """
        return sorted(tasks, key=lambda t: t.time or "99:99")

    def detect_conflicts(self, owner: Owner) -> list[str]:
        """Return warning messages for tasks sharing the same 'HH:MM' time (never raises).

        Lightweight strategy: group every timed task by its time slot in a single
        O(n) pass, then report any slot claimed by more than one task. Works
        across all pets, so same-pet and cross-pet clashes are both caught.
        Returns warnings as data rather than raising, so the caller decides how
        to surface them and the program never crashes.

        Args:
            owner: the owner whose pets' tasks are checked.

        Returns:
            A list of warning strings (empty if there are no conflicts).
        """
        by_time: dict[str, list[tuple[Pet, Task]]] = {}
        for pet, task in self._collect_tasks(owner):
            if not task.time:  # untimed tasks can't conflict
                continue
            by_time.setdefault(task.time, []).append((pet, task))

        warnings: list[str] = []
        for slot in sorted(by_time):
            entries = by_time[slot]
            if len(entries) > 1:
                names = ", ".join(f"'{t.title}' ({p.name})" for p, t in entries)
                warnings.append(f"⚠️ Conflict at {slot}: {names}")
        return warnings
