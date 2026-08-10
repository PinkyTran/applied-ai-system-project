"""PawPal+ demo script.

Builds an owner with two pets and several care tasks (added out of time order),
then demonstrates the scheduler, the sort-by-time method, and task filtering.
Run with: python main.py
"""

from datetime import date

from pawpal_system import Owner, Pet, Priority, Scheduler, Task


def build_owner() -> Owner:
    """Create a sample owner with two pets; tasks are added OUT of time order."""
    biscuit = Pet(name="Biscuit", species="dog", breed="Golden Retriever")
    mochi = Pet(name="Mochi", species="cat", breed="Tabby")

    # Deliberately added out of chronological order to prove sort_by_time works.
    biscuit.add_task(Task("Evening walk", 30, Priority.MEDIUM, time="18:00"))
    biscuit.add_task(Task("Morning walk", 30, Priority.HIGH, time="08:00"))
    biscuit.add_task(Task("Grooming", 25, Priority.LOW, time="13:00"))
    biscuit.add_task(Task("Feeding", 10, Priority.HIGH, time="07:30"))

    mochi.add_task(Task("Play session", 15, Priority.LOW, time="16:00"))
    mochi.add_task(Task("Medication", 5, Priority.HIGH, time="09:00"))
    mochi.add_task(Task("Litter box cleaning", 10, Priority.MEDIUM, time="11:00"))

    # Two deliberate conflicts to exercise detect_conflicts():
    #   - same pet: Biscuit already has "Morning walk" at 08:00
    biscuit.add_task(Task("Backyard potty", 5, Priority.MEDIUM, time="08:00"))
    #   - different pets: Biscuit already has "Feeding" at 07:30
    mochi.add_task(Task("Breakfast", 10, Priority.HIGH, time="07:30"))

    owner = Owner(name="Jordan", available_minutes=120)
    owner.add_pet(biscuit)
    owner.add_pet(mochi)
    return owner


def print_tasks(header: str, tasks: list[Task]) -> None:
    """Print a header and a bulleted list of tasks (with time and status)."""
    print(f"\n{header}")
    if not tasks:
        print("  (none)")
        return
    for t in tasks:
        stamp = t.time or "--:--"
        print(f"  {stamp} — {t.title} ({t.duration_minutes} min) "
              f"[{t.priority.value}] status={t.status}")


def main() -> None:
    owner = build_owner()
    scheduler = Scheduler(start_hour=8)

    print("=" * 52)
    print("Today's Schedule")
    print("=" * 52)
    print(scheduler.build_plan(owner).explain())

    # --- conflict detection: warn (don't crash) on same-time tasks --------
    print("\n" + "=" * 52)
    print("Schedule Conflicts")
    print("=" * 52)
    conflicts = scheduler.detect_conflicts(owner)
    if conflicts:
        for warning in conflicts:
            print(warning)
    else:
        print("No conflicts detected.")

    # --- sorting: Biscuit's tasks were added out of order -----------------
    biscuit_tasks = owner.filter_tasks(pet_name="Biscuit")
    print_tasks("Biscuit's tasks as ENTERED (out of order):", biscuit_tasks)
    print_tasks(
        "Biscuit's tasks SORTED BY TIME:",
        scheduler.sort_by_time(biscuit_tasks),
    )

    # --- filtering: mark a couple complete, then filter -------------------
    owner.pets[0].tasks[1].mark_complete()  # Biscuit "Morning walk"
    owner.pets[1].tasks[1].mark_complete()  # Mochi "Medication"

    print_tasks("Only COMPLETED tasks (all pets):", owner.filter_tasks(status="complete"))
    print_tasks("Only PENDING tasks (all pets):", owner.filter_tasks(status="pending"))
    print_tasks("Only Mochi's tasks:", owner.filter_tasks(pet_name="Mochi"))

    # --- recurrence: completing a repeating task spawns the next one ------
    print("\n" + "=" * 52)
    print("Recurring Tasks")
    print("=" * 52)
    rex = Pet(name="Rex", species="dog")
    rex.add_task(Task("Daily meds", 5, Priority.HIGH, frequency="daily", due_date=date.today()))
    rex.add_task(Task("Weekly bath", 40, Priority.LOW, frequency="weekly", due_date=date.today()))

    print(f"Before completing anything, Rex has {len(rex.tasks)} task(s).")
    for task in list(rex.tasks):  # copy: complete_task appends while we iterate
        upcoming = rex.complete_task(task)
        when = upcoming.due_date if upcoming else "n/a (does not repeat)"
        print(f"  Completed '{task.title}' ({task.frequency}) -> next due: {when}")
    print(f"After completing them, Rex has {len(rex.tasks)} task(s) "
          f"(the auto-created next occurrences).")


if __name__ == "__main__":
    main()
