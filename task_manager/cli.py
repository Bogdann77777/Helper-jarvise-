"""
cli.py — Command-line interface for task manager.
Used by Claude (me) via Bash tool to add/list/done/delete tasks.

Usage:
  python -m task_manager.cli add --project 76 --text "договор с Ивановым" --date "до 20 апреля" --priority high
  python -m task_manager.cli list [--project 76]
  python -m task_manager.cli done 42
  python -m task_manager.cli delete 42
  python -m task_manager.cli vacation --start "2026-05-01" --end "2026-05-10"
  python -m task_manager.cli report
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as script from cli helper root
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_manager.db import (
    init_db, add_task, get_tasks, mark_done, delete_task,
    get_todays_report_tasks, set_vacation, list_vacations,
    get_task, update_task, is_vacation_today,
)
from task_manager.date_parser import parse_date, is_urgent, date_to_iso
from task_manager.formatter import morning_report, project_report, full_list_report


def cmd_add(args):
    init_db()
    due_date = None
    if args.date:
        d = parse_date(args.date)
        if d:
            due_date = date_to_iso(d)
    priority = "high" if args.priority == "high" or is_urgent(args.text) else "normal"
    task_id = add_task(args.project, args.text, due_date, priority)
    result = {"id": task_id, "project": args.project, "text": args.text,
              "due_date": due_date, "priority": priority}
    print(json.dumps(result, ensure_ascii=False))


def cmd_list(args):
    init_db()
    tasks = get_tasks(project=args.project, include_done=args.done)
    if args.format == "json":
        print(json.dumps(tasks, ensure_ascii=False, indent=2))
    elif args.project:
        print(project_report(args.project, tasks))
    else:
        print(full_list_report(tasks))


def cmd_done(args):
    init_db()
    ok = mark_done(args.id)
    print("OK" if ok else "NOT_FOUND")


def cmd_delete(args):
    init_db()
    ok = delete_task(args.id)
    print("OK" if ok else "NOT_FOUND")


def cmd_report(args):
    init_db()
    tasks = get_todays_report_tasks()
    print(morning_report(tasks))


def cmd_vacation(args):
    init_db()
    if args.cancel:
        from task_manager.db import cancel_vacation
        ok = cancel_vacation(int(args.cancel))
        print("Отпуск отменён" if ok else "Не найдено")
        return
    if args.list:
        vacs = list_vacations()
        if not vacs:
            print("Нет запланированных отпусков")
        for v in vacs:
            print(f"#{v['id']} {v['start_date']} — {v['end_date']}  {v['note']}")
        return
    vid = set_vacation(args.start, args.end, args.note or "")
    print(f"Отпуск #{vid} записан: {args.start} — {args.end}")


def cmd_update(args):
    init_db()
    kwargs = {}
    if args.text:
        kwargs["text"] = args.text
    if args.date:
        d = parse_date(args.date)
        kwargs["due_date"] = date_to_iso(d) if d else None
    if args.priority:
        kwargs["priority"] = args.priority
    if args.project:
        kwargs["project"] = args.project
    ok = update_task(args.id, **kwargs)
    print("OK" if ok else "NOT_FOUND")


def main():
    parser = argparse.ArgumentParser(prog="task_manager")
    sub = parser.add_subparsers(dest="cmd")

    # add
    p_add = sub.add_parser("add")
    p_add.add_argument("--project", required=True)
    p_add.add_argument("--text",    required=True)
    p_add.add_argument("--date",    default=None)
    p_add.add_argument("--priority", default="normal", choices=["normal", "high"])

    # list
    p_list = sub.add_parser("list")
    p_list.add_argument("--project", default=None)
    p_list.add_argument("--done",    action="store_true")
    p_list.add_argument("--format",  default="text", choices=["text", "json"])

    # done
    p_done = sub.add_parser("done")
    p_done.add_argument("id", type=int)

    # delete
    p_del = sub.add_parser("delete")
    p_del.add_argument("id", type=int)

    # update
    p_upd = sub.add_parser("update")
    p_upd.add_argument("id",         type=int)
    p_upd.add_argument("--text",     default=None)
    p_upd.add_argument("--date",     default=None)
    p_upd.add_argument("--priority", default=None)
    p_upd.add_argument("--project",  default=None)

    # report
    sub.add_parser("report")

    # vacation
    p_vac = sub.add_parser("vacation")
    p_vac.add_argument("--start",  default=None)
    p_vac.add_argument("--end",    default=None)
    p_vac.add_argument("--note",   default=None)
    p_vac.add_argument("--list",   action="store_true")
    p_vac.add_argument("--cancel", default=None)

    args = parser.parse_args()

    dispatch = {
        "add":      cmd_add,
        "list":     cmd_list,
        "done":     cmd_done,
        "delete":   cmd_delete,
        "report":   cmd_report,
        "vacation": cmd_vacation,
        "update":   cmd_update,
    }

    if args.cmd not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
