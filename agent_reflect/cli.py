"""Command line: reflect, consolidate, backfill, rollback (skills or --memory), install, uninstall, status."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from . import (
    baseline, chart, consolidate, cursor_index, digest, guard, install as launchd, journal, memory_archive, memory_map,
    recent, reflect,
    i18n, skills,
)
from .config import RUN_ID_FORMAT, Config, ConfigError, is_run_id
from .i18n import quoted, t
from .ledger import PATTERN_FIELDS, State, load_state, parse_ts, run_lock, save_state, snapshot_baseline
from .llm import ensure_isolation_files
from .memory import (
    MemoryIndexError, MemoryWriteError, go_live_if_room, remove_index_line, sync_ownership, write_lesson,
)
from .guard_hook import shell_text
from .projects import resolve_project_root
from .promptkit import error_text
from .scrub import scrub
from .transcript import list_transcripts

log = logging.getLogger("agent_reflect")
CRASH_NOTIFY_AFTER = 3
REASON_MAX = 160  # of an abort reason in a notification


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def _logging_to(cfg: Config, cmd: str) -> Iterator[None]:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(cfg.logs_dir / f"{cmd}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        yield
    finally:
        log.removeHandler(handler)
        handler.close()


def _trigger() -> str:
    """"schedule" when launchd started the run (then it is the parent process), else "manual"."""
    return "schedule" if os.getppid() == 1 else "manual"


def _journal_run(cfg: Config, state: State, kind: str, *, started: datetime, ended: datetime, result: dict,
                 events: list[dict], failed: str | None = None, digest_path: Path | None = None) -> None:
    """Append the run to the journal; one that cannot be written is logged and noted, it never fails the run."""
    trigger = "manual" if kind == "backfill" else _trigger()
    try:
        records = journal.run_records(kind=kind, trigger=trigger, started=started, ended=ended, result=result,
                                      events=events, failed=failed,
                                      digest=str(digest_path) if digest_path else None,
                                      live=journal.live_counts(state))
        journal.append(cfg, records, _now())
    except Exception as exc:  # a full disk, a lone surrogate that utf-8 refuses, or a bug: the report's, not the run's
        log.error("journal not written: %s", error_text(exc))
        state.event("journal_failed", error=error_text(exc))


def _count_failure(state: State, cmd: str, reason: str = "") -> None:
    """One more failed run in a row; from the third one on, a notification says so."""
    if cmd == "consolidate":
        state.consolidate_failures += 1
        count = state.consolidate_failures
    else:
        state.consecutive_failures += 1
        count = state.consecutive_failures
    if count >= CRASH_NOTIFY_AFTER:
        detail = f": {scrub(reason)[:REASON_MAX]}" if reason else ""
        digest.notify(t(f"agent-reflect: {cmd} has failed {count} times in a row{detail}, see logs/{cmd}.log",
                        f"agent-reflect: {cmd} падает {count} раз подряд{detail}, см. logs/{cmd}.log"))


def _log_traceback(cmd: str, exc: BaseException) -> None:
    log.error("%s crashed\n%s", cmd, scrub("".join(traceback.format_exception(exc))))


def _record_crash(cfg: Config, state: State, cmd: str, exc: BaseException, before: list[dict], start: int) -> None:
    """Note the crash in the state and the journal; `before` and `start` tell which events the run added."""
    _log_traceback(cmd, exc)
    error = scrub(str(exc) or type(exc).__name__)[:300]
    state.event("run_crashed", cmd=cmd, error=error)
    at = _now()
    _journal_run(cfg, state, cmd, started=at, ended=at, result={}, failed=error,
                 events=journal.events_added(state, before, start))
    _count_failure(state, cmd)


def _log_error_events(state: State, before: list[dict], start: int) -> None:
    """Write each error event of this run to the log, one scrubbed line each."""
    for event in journal.events_added(state, before, start):
        if event.get("kind") in consolidate.ERROR_KINDS:
            log.error("event %s", scrub(json.dumps(event, ensure_ascii=False, default=str)))


def _on_sigterm(signum, frame) -> None:
    raise SystemExit(143)


def _guarded(cfg: Config, cmd: str, work: Callable[[State], int], *, save: bool = True) -> int:
    """Lock, load state, run `work`, then save unless `save` is False (a dry run).

    A crash or kill leaves a trace; a dry run's crash is only logged.
    """
    with _logging_to(cfg, cmd):
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        ensure_isolation_files(cfg)
        with run_lock(cfg.lock_file) as acquired:
            if not acquired:
                log.info("already running")
                return 0
            try:
                state = load_state(cfg.state_file)
            except Exception as exc:  # leave the unreadable file untouched
                log.error("state unreadable: %s", scrub(str(exc))[:300])
                digest.notify(t(f"agent-reflect: the state file is unreadable, see logs/{cmd}.log",
                                f"agent-reflect: файл состояния не читается, см. logs/{cmd}.log"))
                return 1
            previous = signal.signal(signal.SIGTERM, _on_sigterm)
            fatal: BaseException | None = None
            code = 1
            events, start = state.events, len(state.events)
            try:
                code = work(state)
            except BaseException as exc:
                if save:
                    _record_crash(cfg, state, cmd, exc, events, start)
                else:
                    _log_traceback(cmd, exc)
                if not isinstance(exc, Exception):
                    fatal = exc
            finally:
                signal.signal(signal.SIGTERM, previous)
                _log_error_events(state, events, start)
                if save:
                    save_state(state, cfg.state_file)
            if fatal is not None:
                raise fatal
            return code


def _cmd_reflect(cfg: Config, args: argparse.Namespace) -> int:
    def work(state: State) -> int:
        started = _now()
        before, start = state.events, len(state.events)
        result = reflect.run(cfg, state, now=started, notify=lambda m: digest.notify(m),
                             max_segments=args.max_segments, dry_run=args.dry_run)
        shown = {key: value for key, value in result.items() if key != "verdicts"}
        if args.dry_run:
            print(json.dumps(shown, ensure_ascii=False, indent=2))
        else:
            # before the `done` line: the first journal write imports the logs, and must not find this run in them
            _journal_run(cfg, state, "reflect", started=started, ended=_now(), result=result,
                         events=journal.events_added(state, before, start))
            _night_report(cfg, state, started)
        log.info("reflect done: %s", json.dumps(shown, ensure_ascii=False))
        return 0
    return _guarded(cfg, "reflect", work, save=not args.dry_run)


def _night_report(cfg: Config, state: State, started: datetime) -> None:
    """When this reflect is the night's (see `recent.window`): the report into reports/ and a notification.

    A run by hand, or a later one on the night's day (launchd runs the job at login too), tells nothing:
    the night was told already. A report that fails is logged; it never fails the run.
    """
    try:
        now = _now()
        records = journal.load(cfg)
        _, nightly, _ = recent.window(records, now, None)
        if nightly is None or nightly.get("run") != journal.run_id(started):
            return
        night = parse_ts(nightly["started_at"]).astimezone().date()
        path = digest.write_night(cfg, recent.render(cfg, state, records, now), night)
        title, subtitle, message = recent.notice(cfg, state, records, now)
        digest.notify(message, path, title=title, subtitle=subtitle, group=digest.NIGHT_GROUP)
    except Exception as exc:
        log.error("night report not written: %s", error_text(exc))


def _consolidate_due(cfg: Config) -> bool:
    """Read-only look at the schedule: True also when the state cannot be read."""
    try:
        state = load_state(cfg.state_file)
    except Exception:
        return True  # the guarded run reports the unreadable state
    return consolidate.is_due(state, _now(), cfg.consolidate_interval_days)


def _week_days(since: str | None, now: datetime) -> tuple[date, date]:
    """The digest's local days: from the previous consolidate (the install for the first one) to today."""
    last = now.astimezone().date()
    began = parse_ts(since)
    return (min(began.astimezone().date(), last) if began else last), last


def _week_chart(cfg: Config, state: State, first: date, last: date) -> list[str]:
    """The digest's chart of the days from `first` to `last`.

    A journal or MEMORY.md that cannot be read, or a bug in the drawing, leaves the chart out: the digest
    is written without it.
    """
    try:
        installed = parse_ts(state.installed_at)
        return digest.week_block(cfg, state.lessons, journal.load(cfg), first, last,
                                 installed.astimezone().date() if installed else None)
    except Exception as exc:
        log.error("week chart not drawn: %s", error_text(exc))
        return []


def _memory_chart(cfg: Config, state: State, summary: dict, first: date, last: date) -> list[str]:
    """The digest's map of memory. A bug in the drawing leaves it out: the digest is written without it."""
    try:
        return memory_map.block(cfg, state.lessons, summary.get("memory_reads"), first, last)
    except Exception as exc:
        log.error("memory map not drawn: %s", error_text(exc))
        return []


def _cmd_consolidate(cfg: Config, args: argparse.Namespace) -> int:
    # launchd launches this daily: a tick that is not due takes no lock, writes no log and no state
    if not args.force and not args.dry_run and not _consolidate_due(cfg):
        return 0

    def work(state: State) -> int:
        now = _now()
        before, start = state.events, len(state.events)
        since = state.last_consolidate_at or state.installed_at  # consolidate.run moves the first to now
        summary = consolidate.run(cfg, state, now=now, force=args.force, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        # the run record takes no counters from the summary: its `errors` is a list, not a reflect run's count
        events = journal.events_added(state, before, start)
        if summary.get("aborted"):
            reason = str(summary["aborted"])
            log.error("consolidate aborted: %s", scrub(reason))
            _journal_run(cfg, state, "consolidate", started=now, ended=_now(), result={}, events=events,
                         failed=scrub(reason))
            _count_failure(state, "consolidate", reason)
            return 1
        state.consolidate_failures = 0
        path = None
        if summary:
            week = digest.week_tag(now)
            first, last = _week_days(since, now)
            path = digest.free_path(cfg, week)  # a second digest of the week keeps the first
            text = digest.render(summary, state.lessons, week, cfg, chart=_week_chart(cfg, state, first, last),
                                 memory=_memory_chart(cfg, state, summary, first, last), path=path)
            digest.write(cfg, text, week, path)
            title, subtitle, message = digest.notice(summary, first, last)
            digest.notify(message, path, title=title, subtitle=subtitle, group=digest.WEEK_GROUP)
        _journal_run(cfg, state, "consolidate", started=now, ended=_now(), result={}, events=events,
                     digest_path=path)
        if path:
            log.info("consolidate done, digest %s", path)
        else:
            log.info("consolidate: nothing to do")
        return 0
    return _guarded(cfg, "consolidate", work, save=not args.dry_run)


def _cmd_backfill(cfg: Config, args: argparse.Namespace) -> int:
    since = datetime.strptime(args.since, "%Y-%m-%d").timestamp()  # local midnight

    def work(state: State) -> int:
        now = _now()
        before, start = state.events, len(state.events)
        transcripts = list_transcripts(cfg.projects_dir, cfg.own_project_prefixes)
        if not state.installed_at:
            snapshot_baseline(state, transcripts, now.strftime("%Y-%m-%dT%H:%M:%SZ"))
        reset = 0
        for path in transcripts:
            try:
                if path.stat().st_mtime >= since:
                    record = state.sessions.setdefault(str(path), {})
                    # the health up to the old offset is counted already: a re-read must not count it again
                    key = str(path)
                    if record.get("offset", 0) > state.health_offsets.get(key, 0):
                        state.health_offsets[key] = record["offset"]
                    record["offset"] = 0
                    # the re-read resolves the project afresh: a cwd stored under an older rule
                    # (a subagent's own scratchpad) would otherwise keep deciding it
                    record.pop("cwd", None)
                    reset += 1
            except OSError:
                continue
        log.info("backfill: %d transcripts reset since %s", reset, args.since)
        result = reflect.run(cfg, state, now=now, notify=lambda m: digest.notify(m),
                             max_segments=args.max_segments)
        shown = {key: value for key, value in result.items() if key != "verdicts"}
        # before the `done` line, as in reflect
        _journal_run(cfg, state, "backfill", started=now, ended=_now(), result=result,
                     events=journal.events_added(state, before, start))
        log.info("backfill done: %s", json.dumps(shown, ensure_ascii=False))
        return 0
    return _guarded(cfg, "backfill", work)


ROLLBACK_USAGE = "usage: agent-reflect rollback RUN_ID | --memory RUN_ID  (pick one of the run ids above)"


def _cmd_rollback(cfg: Config, args: argparse.Namespace) -> int:
    for value in (args.run_id, args.memory):
        if value is not None and not is_run_id(value):
            print(f"agent-reflect: not a run id: {value!r} (expected YYYYMMDDTHHMMSS)", file=sys.stderr)
            return 2
    if args.memory and not args.list and not args.run_id:
        return _rollback_memory(cfg, args.memory)
    backups = skills.list_backups(cfg)
    if args.list or not args.run_id or args.memory:
        for run_id in backups:
            print(run_id)
        for run_id in memory_archive.list_archives(cfg):
            print(f"--memory {run_id}")
        if args.list:
            return 0
        print(ROLLBACK_USAGE, file=sys.stderr)
        return 2
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with _logging_to(cfg, "rollback"), run_lock(cfg.lock_file) as acquired:
        if not acquired:
            log.info("already running")
            print("already running")
            return 0
        targets = _backup_targets(cfg, args.run_id)
        if not targets:
            print(f"nothing to restore for {args.run_id}")
            return 1
        print(f"rolling back {args.run_id}:")
        for target in targets:
            print(f"  {target}")
        folders = {m.parent for m in (cfg.backups_dir / args.run_id).glob("*/manifest.json")}
        restored = skills.rollback(cfg, args.run_id)
        log.info("rollback %s: %d files", args.run_id, len(restored))
        _lift_holds(cfg, args.run_id, folders)
    return 0


def _lift_holds(cfg: Config, run_id: str, folders: set[Path]) -> None:
    """The skill edits these backup folders held are undone: their lessons may be promoted again,
    and the moves they made are watched no more."""
    try:
        state = load_state(cfg.state_file)
    except Exception as exc:  # leave the unreadable file untouched
        print(f"agent-reflect: state unreadable, holds of {run_id} stay: {scrub(str(exc))[:300]}", file=sys.stderr)
        return
    lifted, kept = memory_archive.lift_holds(cfg, state, run_id, folders)
    ended = memory_archive.end_rolled_back_moves(cfg, state, run_id, folders, _now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    if lifted or ended:
        save_state(state, cfg.state_file)
    for lesson_id in ended:
        print(t(f"  move of {lesson_id} undone: its effect is no longer checked",
                f"  перенос {lesson_id} отменён: его эффект больше не проверяется"))
    for lesson_id in lifted:
        print(f"  hold lifted {lesson_id}: it may be promoted again")
    for lesson_id in kept:
        print(f"  hold kept   {lesson_id}: its edit was not one of these files (a PR or another target)")


def _rollback_memory(cfg: Config, run_id: str) -> int:
    """Put back the auto files a consolidate run removed or rewrote.

    First it applies what the user did since the run, exactly as the next night would
    (sync_ownership: a deleted file or line is a rejection, and its line or file goes
    too), printing each as a `user` line and journaling each rejection. Then it prints
    every file, line and status it will restore before writing any of it. Exits 1 when
    it refused part of the archive or a write failed.
    """
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with _logging_to(cfg, "rollback"), run_lock(cfg.lock_file) as acquired:
        if not acquired:
            log.info("already running")
            print("already running")
            return 0
        if not memory_archive.has_archive(cfg, run_id):
            print(f"nothing to restore for memory run {run_id}")
            return 1
        try:
            state = load_state(cfg.state_file)
        except Exception as exc:  # leave the unreadable file untouched
            print(f"agent-reflect: state unreadable: {scrub(str(exc))[:300]}", file=sys.stderr)
            return 1
        print(f"rolling back memory run {run_id}:")
        touched = _sync_user_changes(cfg, state, run_id)
        plan = memory_archive.plan_rollback(cfg, state, run_id)
        for restore in plan.restores:
            print(f"  restore {restore.path}")
            if restore.index_line is not None:
                print(f"  index   {restore.path.parent / 'MEMORY.md'}: {restore.index_line}")
            print(f"  status  {restore.record.id}: {restore.previous_status or '-'} → {restore.status}")
            for note in restore.notes:
                print(f"  note    {restore.record.id}: {note}")
        for note in plan.skipped:
            print(f"  skip {note}")
        for note in plan.refused:
            print(f"  refused {note}")
        problems = memory_archive.apply_rollback(cfg, state, plan) if plan.restores else []
        if not plan.restores:
            print("  nothing to restore")
        if plan.restores or touched:
            save_state(state, cfg.state_file)
        touched |= {(r.project_root, str(r.path.parent)) for r in plan.restores}
        for project_root, memory_dir in sorted(touched):
            cursor_index.sync(cfg, Path(project_root), Path(memory_dir))
        for problem in problems:
            print(f"  failed {problem}", file=sys.stderr)
            log.error("memory rollback %s: %s", run_id, scrub(problem))
        for note in plan.refused:
            log.error("memory rollback %s refused: %s", run_id, scrub(note))
        log.info("memory rollback %s: %d files", run_id, len(plan.restores))
        return 1 if problems or plan.refused else 0


def _sync_user_changes(cfg: Config, state: State, run_id: str) -> set[tuple[str, str]]:
    """sync_ownership with every change it makes printed, and its rejections journaled."""
    rollback_run = _now().strftime(RUN_ID_FORMAT)
    before = {lesson_id: lesson.status for lesson_id, lesson in state.lessons.items()}
    had_file = {lesson_id: (Path(lesson.memory_dir) / lesson.file).exists()
                for lesson_id, lesson in state.lessons.items() if lesson.status in ("live", "staged")}
    touched = set(sync_ownership(state))
    for lesson_id, lesson in state.lessons.items():
        previous = before.get(lesson_id)
        if previous == lesson.status:
            continue
        path = Path(lesson.memory_dir) / lesson.file
        if lesson.status != "rejected":
            print(f"  user    {lesson_id}: {previous} → {lesson.status} (the user edited {path})")
            continue
        if had_file.get(lesson_id) and not path.exists():
            what = f"the user deleted its index line, so {path} is deleted"
        else:
            what = "the user deleted its file, so its index line is removed"
        print(f"  user    {lesson_id}: {previous} → rejected ({what})")
        try:
            memory_archive.append_journal(cfg, {
                "ts": _now().strftime("%Y-%m-%dT%H:%M:%SZ"), "run_id": rollback_run, "kind": "reject",
                "lesson_id": lesson_id, "project_root": lesson.project_root, "previous_status": previous,
                "status": lesson.status, "file": lesson.file, "during": f"rollback --memory {run_id}"})
        except (OSError, ValueError) as exc:
            print(f"  failed  journal line for {lesson_id}: {scrub(str(exc))[:300]}", file=sys.stderr)
    return touched


def _backup_targets(cfg: Config, run_id: str) -> list[str]:
    targets: list[str] = []
    for manifest in sorted((cfg.backups_dir / run_id).glob("*/manifest.json")):
        try:
            targets += [m["src"] for m in json.loads(manifest.read_text())]
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return targets


def _cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    state = load_state(cfg.state_file)
    unread = 0
    for key, info in state.sessions.items():
        try:
            unread += max(0, Path(key).stat().st_size - info.get("offset", 0))
        except OSError:
            continue
    by_status: dict[str, int] = {}
    for lesson in state.lessons.values():
        by_status[lesson.status] = by_status.get(lesson.status, 0) + 1
    print(f"installed_at: {state.installed_at}")
    print(f"tracked transcripts: {len(state.sessions)}, unread bytes: {unread}")
    print("lessons: " + (", ".join(f"{k}={v}" for k, v in sorted(by_status.items())) or "none"))
    print(f"last_consolidate_at: {state.last_consolidate_at}")
    print(f"open PRs: {len(state.open_prs)}")
    for pr in state.open_prs:
        print(f"  {pr.get('url', pr)}")
    print("last events:")
    for event in state.events[-5:]:
        print("  " + json.dumps(event, ensure_ascii=False))
    return 0


def _cmd_recent(cfg: Config, args: argparse.Namespace) -> int:
    print(recent.render(cfg, load_state(cfg.state_file), journal.load(cfg), _now(), hours=args.hours,
                        verbose=args.verbose))
    return 0


def _cmd_install(cfg: Config, args: argparse.Namespace) -> int:
    try:
        paths = launchd.install(cfg)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        print(f"agent-reflect: launchctl failed: {scrub(err)[:500]}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return _guard_on(cfg)


def _guard_on(cfg: Config) -> int:
    """Wire the Bash hook into settings.json and write its rules from the ledger."""
    try:
        changed = guard.install_hook(cfg)
        guard.sync(cfg, load_state(cfg.state_file))
    except (OSError, ValueError) as exc:
        print(f"agent-reflect: guard hook not installed: {scrub(str(exc))[:300]}", file=sys.stderr)
        return 1
    print(f"{cfg.claude_settings}: guard and session-start hooks {'added' if changed else 'already there'}")
    return 0


def _from_agent() -> bool:
    """True in a Claude Code shell: the agent runs the command, not the user."""
    return os.environ.get("CLAUDECODE") == "1"


def _users_only(what: str) -> int:
    """Refuse the agent what only the user decides: setting or lifting a block, or taking a blocking lesson away."""
    print(t(f"agent-reflect: {what}: run from Claude Code (CLAUDECODE=1); only the user does this, in their own "
            f"terminal; tell them the command and why",
            f"agent-reflect: {what}: запущено из Claude Code (CLAUDECODE=1); это делает только пользователь, "
            f"в своём терминале; скажи ему команду и зачем"),
          file=sys.stderr)
    return 2


def _blocks(lesson) -> bool:
    """The lesson's block holds: the hook's rules carry only trusted lessons."""
    return lesson.block and lesson.status in guard.ACTIVE


def _something_blocks(cfg: Config) -> bool:
    try:
        return any(_blocks(lesson) for lesson in load_state(cfg.state_file).lessons.values())
    except Exception:  # unreadable: a block may stand in it, so the agent is refused as if one did
        return True


def _cmd_uninstall(cfg: Config, args: argparse.Namespace) -> int:
    if _from_agent() and _something_blocks(cfg):
        return _users_only("uninstall")
    launchd.uninstall(cfg)
    try:
        guard.uninstall_hook(cfg)
    except OSError as exc:
        print(f"agent-reflect: guard hook not removed: {scrub(str(exc))[:300]}", file=sys.stderr)
        return 1
    return 0


def _guard_list(cfg: Config) -> int:
    state = load_state(cfg.state_file)
    for lesson in sorted(state.lessons.values(), key=lambda item: item.id):
        shown = t("shown", "показывается") if lesson.status in guard.ACTIVE else t("waits", "ждёт") + f" ({lesson.status})"
        if lesson.block:
            shown += t(", blocks", ", блокирует")
        for on, pattern in lesson.patterns():
            print(f"{lesson.id}  {shown}, " + t("fired: ", "сработал: ")
                  + f"{lesson.fires}  {_on_mark(on)} /{pattern}/  {lesson.title}")
    for rule in state.user_guards:
        mark = _on_mark(str(rule.get("on")))
        where = rule.get("project") or _every_project()
        print(f"{guard.USER_PREFIX}{rule.get('name')}  " + t("your rule, fired: ", "твоё правило, сработало: ")
              + f"{rule.get('fires') or 0}  {mark} /{rule.get('pattern')}/  {where}  {rule.get('text')}")
    return 0


def _on_mark(on: str) -> str:
    return {"command": t("command", "команда"), "path": t("file", "файл")}.get(on) or t("error", "ошибка")


def _every_project() -> str:
    return t("every project", "все проекты")


def _rule_project(value: str | None) -> str:
    """A rule's project as sessions name it: absolute, no trailing slash, the repository's root when inside one."""
    if not value:
        return ""
    path = os.path.normpath(os.path.abspath(os.path.expanduser(value)))
    root = resolve_project_root(path) if os.path.isdir(path) else None
    return str(root) if root is not None else path


def _guard_user_rule(cfg: Config, args: argparse.Namespace, on: str) -> int:
    """`guard add <name> <regex> --text …` and `guard remove <name>`: the user's own rules for the hook."""
    name, pattern = (args.args[0] if args.args else ""), " ".join(args.args[1:])
    if not guard.NAME_RE.fullmatch(name):
        print("agent-reflect: guard add|remove needs a name of letters, digits, - and _", file=sys.stderr)
        return 2
    if args.action == "add":
        if why := guard.check_trigger(pattern, on):
            print(f"agent-reflect: rule not added: {why}", file=sys.stderr)
            return 2
        if not (args.text or "").strip():
            print("agent-reflect: rule not added: --text says what the agent is shown", file=sys.stderr)
            return 2

    project = _rule_project(args.project)

    def work(state: State) -> int:
        kept = [rule for rule in state.user_guards if rule.get("name") != name]
        if args.action == "remove":
            if len(kept) == len(state.user_guards):
                print(f"agent-reflect: no rule {name}", file=sys.stderr)
                return 2
            state.user_guards = kept
            print(f"{name}: " + t("removed", "убрано"))
        else:
            state.user_guards = [*kept, {"name": name, "on": on, "pattern": pattern, "text": args.text.strip(),
                                         "project": project, "fires": 0, "last_fired_at": None}]
            print(f"{name}: /{pattern}/ → {project or _every_project()}")
        guard.sync(cfg, state)
        return 0
    return _guarded(cfg, "guard", work)


def _cmd_guard(cfg: Config, args: argparse.Namespace) -> int:
    """What the guard hook shows: list it, set or drop a lesson's trigger by hand, add a rule, try a text."""
    if args.path and args.error:
        print(t("agent-reflect: --path and --error do not go together",
                "agent-reflect: --path и --error вместе не ставятся"), file=sys.stderr)
        return 2
    on = "path" if args.path else "error" if args.error else "command"
    if args.action == "on":
        return _guard_on(cfg)
    if args.action == "off":
        if _from_agent() and _something_blocks(cfg):
            return _users_only("guard off")
        print("guard and session-start hooks removed" if guard.uninstall_hook(cfg)
              else "guard and session-start hooks were not installed")
        return 0
    if args.action == "list":
        return _guard_list(cfg)
    if args.action == "test":
        text = " ".join(args.args)
        if on == "command":
            text = shell_text(text)  # as the hook reads it
        for rule in guard.rules(load_state(cfg.state_file)):
            if rule["on"] == on and re.search(rule["pattern"], text):
                print(f"{rule['id']}  {rule['text']}")
        return 0
    if args.action in ("add", "remove"):
        return _guard_user_rule(cfg, args, on)
    if args.action in ("block", "unblock"):
        return _guard_block(cfg, args)
    if not args.args:
        print("agent-reflect: guard set|unset needs a lesson id", file=sys.stderr)
        return 2
    lesson_id, trigger = args.args[0], " ".join(args.args[1:])
    if args.action == "set" and (why := guard.check_trigger(trigger, on)):
        print(f"agent-reflect: trigger not set: {why}", file=sys.stderr)
        return 2
    field = dict(PATTERN_FIELDS)[on]

    def work(state: State) -> int:
        lesson = state.lessons.get(lesson_id)
        if lesson is None:
            print(f"agent-reflect: no lesson {lesson_id}", file=sys.stderr)
            return 2
        if _blocks(lesson) and _from_agent():
            return _users_only(f"guard {args.action} {lesson_id}")
        setattr(lesson, field, trigger if args.action == "set" else "")
        if field == "trigger" and not lesson.trigger:
            lesson.block = False  # nothing left to block on
        guard.sync(cfg, state)
        now = getattr(lesson, field)
        print(f"{lesson_id}: /{now}/" if now else f"{lesson_id}: " + t("no pattern", "без шаблона"))
        return 0
    return _guarded(cfg, "guard", work)


def _guard_block(cfg: Config, args: argparse.Namespace) -> int:
    """`guard block <id>`: the hook refuses a command the lesson's trigger matches; `unblock` lifts it.

    Only the user turns it on: the weekly digest proposes it for a lesson broken although the hook showed it.
    """
    if len(args.args) != 1:
        print(t(f"agent-reflect: guard {args.action} needs one lesson id",
                f"agent-reflect: guard {args.action} нужен один id урока"), file=sys.stderr)
        return 2
    lesson_id = args.args[0]
    if _from_agent():
        return _users_only(f"guard {args.action}")

    def work(state: State) -> int:
        lesson = state.lessons.get(lesson_id)
        if lesson is None:
            print(f"agent-reflect: {lesson_id}: {_no_lesson()}", file=sys.stderr)
            return 2
        if args.action == "block" and not lesson.trigger:
            print(t(f"agent-reflect: {lesson_id} has no command pattern to block on "
                    f"(`agent-reflect guard set {lesson_id} <regex>`)",
                    f"agent-reflect: у {lesson_id} нет шаблона команды, блокировать не по чему "
                    f"(`agent-reflect guard set {lesson_id} <regex>`)"), file=sys.stderr)
            return 2
        if args.action == "block" and lesson.status not in guard.ACTIVE:
            print(t(f"agent-reflect: {lesson_id} is not shown by the hook (now: {lesson.status})",
                    f"agent-reflect: {lesson_id} хук не показывает (сейчас: {lesson.status})"), file=sys.stderr)
            return 2
        lesson.block = args.action == "block"
        guard.sync(cfg, state)
        print(f"{lesson_id}: " + (t("blocks", "блокирует") if lesson.block else t("does not block", "не блокирует")))
        return 0
    return _guarded(cfg, "guard", work)


_MEMORY_ERRORS = (MemoryWriteError, MemoryIndexError, OSError, UnicodeEncodeError)


def _day_arg(value: str | None) -> date | None:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def _cmd_baseline(cfg: Config, args: argparse.Namespace) -> int:
    """Count how often the user sent work back in past days, with today's detector; memory is not touched."""
    try:
        since, until = _day_arg(args.since), _day_arg(args.until)
    except ValueError:
        print("agent-reflect: --since and --until must be YYYY-MM-DD", file=sys.stderr)
        return 2
    until = until or (_now().astimezone().date() - timedelta(days=1))
    with _logging_to(cfg, "baseline"):
        try:
            state = load_state(cfg.state_file)
        except Exception as exc:
            print(f"agent-reflect: state unreadable: {scrub(str(exc))[:300]}", file=sys.stderr)
            return 1
        ensure_isolation_files(cfg)
        out = baseline.run(cfg, state, since=since, until=until, now=_now(), dry_run=args.dry_run,
                           max_calls=args.max_calls)
        log.info("baseline done: %s", json.dumps({k: v for k, v in out.items() if k != "days"}, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    sessions, calls, errors, left = out["sessions"], out["calls"], out["errors"], out.get("left", 0)
    print(t(f"sessions: {sessions}, model calls: {calls}, unanswered: {errors}",
            f"сессий: {sessions}, вызовов модели: {calls}, без ответа: {errors}")
          + (t(f", left to ask: {left} (run it again)", f", осталось спросить: {left} (запусти ещё раз)")
             if out.get("partial") else ""))
    print(digest.baseline_line(out, reflect.DETECTOR)
          or t("no replies of the user found in these days", "ответов пользователя за эти дни не найдено"))
    if out.get("forms"):
        print(t("by kind: ", "по видам: ") + ", ".join(f"{chart.form_name(f)}: {n}" for f, n in out["forms"].items()))
    print(t("written: ", "записано: ") + str(cfg.baseline_file))
    return 0


def _after_shelf_change(cfg: Config, state: State, lesson) -> None:
    """The hook's rules and the Cursor copy follow a lesson that entered or left the index."""
    guard.sync(cfg, state)
    cursor_index.sync(cfg, Path(lesson.project_root), Path(lesson.memory_dir))


def _no_lesson() -> str:
    return t("no such lesson", "нет такого урока")


def _memory_unwritten() -> str:
    return t("agent-reflect: memory was not written: ", "agent-reflect: память не записалась: ")


def _cmd_approve(cfg: Config, args: argparse.Namespace) -> int:
    """Put a waiting (or retired) lesson into the index now: the user's word stands for the second case."""
    def work(state: State) -> int:
        lesson = state.lessons.get(args.lesson_id)
        if lesson is None or lesson.status not in ("staged", "retired"):
            why = (_no_lesson() if lesson is None else t(f"the lesson waits for no decision (now: {lesson.status})",
                                                          f"урок не ждёт решения (сейчас: {lesson.status})"))
            print(f"agent-reflect: {args.lesson_id}: {why}", file=sys.stderr)
            return 2
        memory_dir = Path(lesson.memory_dir)
        try:
            unchanged = hashlib.sha256((memory_dir / lesson.file).read_bytes()).hexdigest() == lesson.sha256
        except OSError as exc:
            print(t("agent-reflect: the lesson file is unreadable: ", "agent-reflect: файл урока не читается: ")
                  + scrub(str(exc))[:300], file=sys.stderr)
            return 2
        if not unchanged:
            print(t(f"agent-reflect: you edited {lesson.file} yourself — it is yours now, add its MEMORY.md line by hand",
                    f"agent-reflect: {lesson.file} ты правил сам — он уже твой, строку в MEMORY.md добавь руками"),
                  file=sys.stderr)
            return 2
        now_iso = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
        lesson.status, lesson.confidence, lesson.updated_at = "staged", "high", now_iso
        try:
            lesson.sha256 = write_lesson(memory_dir, lesson)
            live = go_live_if_room(cfg, lesson, now_iso, state)
        except _MEMORY_ERRORS as exc:
            print(_memory_unwritten() + scrub(str(exc))[:300], file=sys.stderr)
            return 1
        state.event("lesson_approved", project=lesson.project_root, lesson_id=lesson.id, title=lesson.title, live=live)
        _after_shelf_change(cfg, state, lesson)
        print(f"{lesson.id}: {quoted(lesson.title)} — " + (
            t("in the index, agents see it from their next session", "в индексе, агенты видят его со следующей сессии")
            if live else t("accepted; MEMORY.md has no room now, its line goes in once there is",
                           "принят; в MEMORY.md сейчас нет места, строка встанет, когда оно появится")))
        return 0
    return _guarded(cfg, "approve", work)


def _cmd_retire(cfg: Config, args: argparse.Namespace) -> int:
    """Take a live lesson's line out of the index and keep its file: a repeat of its mistake brings it back."""
    def work(state: State) -> int:
        lesson = state.lessons.get(args.lesson_id)
        if lesson is None or lesson.status != "live":
            why = (_no_lesson() if lesson is None else t(f"the lesson is not in the index (now: {lesson.status})",
                                                          f"урок не в индексе (сейчас: {lesson.status})"))
            print(f"agent-reflect: {args.lesson_id}: {why}", file=sys.stderr)
            return 2
        if _blocks(lesson) and _from_agent():
            return _users_only(f"retire {lesson.id}")
        try:
            remove_index_line(Path(lesson.memory_dir), lesson.file)
        except _MEMORY_ERRORS as exc:
            print(_memory_unwritten() + scrub(str(exc))[:300], file=sys.stderr)
            return 1
        lesson.status = "retired"
        state.event("lesson_retired", project=lesson.project_root, lesson_id=lesson.id, title=lesson.title)
        _after_shelf_change(cfg, state, lesson)
        print(f"{lesson.id}: {quoted(lesson.title)} — " + t(
            f"its line is out of MEMORY.md, the file {lesson.file} stays; to bring it back: agent-reflect approve {lesson.id}",
            f"строка убрана из MEMORY.md, файл {lesson.file} остался; вернуть: agent-reflect approve {lesson.id}"))
        return 0
    return _guarded(cfg, "retire", work)


def _cmd_confirm(cfg: Config, args: argparse.Namespace) -> int:
    """The user checked a fact again and it holds: it leaves the digest's recheck list for another month."""
    def work(state: State) -> int:
        lesson = state.lessons.get(args.lesson_id)
        if lesson is None or lesson.kind != "fact":
            why = _no_lesson() if lesson is None else t("the lesson is not a fact", "урок — не факт")
            print(f"agent-reflect: {args.lesson_id}: {why}", file=sys.stderr)
            return 2
        lesson.rechecked_at = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"{lesson.id}: {quoted(lesson.title)} — " + t("confirmed", "подтверждён"))
        return 0
    return _guarded(cfg, "confirm", work)


def _cmd_staged(cfg: Config, args: argparse.Namespace) -> int:
    """The lessons that wait for a second case, each with what it stands on and how to take it now."""
    state = load_state(cfg.state_file)
    waiting = sorted((lesson for lesson in state.lessons.values() if lesson.status == "staged"),
                     key=lambda lesson: lesson.created_at)
    for lesson in waiting:
        project = Path(lesson.project_root).name or lesson.project_root
        sessions = len({e.get("session") for e in lesson.evidence if isinstance(e, dict)})
        print(f"{lesson.id}  {lesson.title}  ({project}, {lesson.confidence}, " + t("sessions: ", "сессий: ")
              + f"{sessions}, " + t("since ", "с ") + f"{lesson.created_at[:10]})")
        print(t("  when:    ", "  когда:   ") + lesson.cue)
        print(t("  do:      ", "  делать:  ") + lesson.lesson)
        for entry in lesson.evidence[:2]:
            print(t("  grounds: ", "  опора:   ") + scrub(str(entry.get("text", "")))[:300])
        print(t("  accept:  ", "  принять: ") + f"agent-reflect approve {lesson.id}")
        print()
    if not waiting:
        print(t("no lessons are waiting", "уроков в ожидании нет"))
    return 0


def _positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-reflect")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reflect")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-segments", type=_positive)
    p.set_defaults(fn=_cmd_reflect)
    p = sub.add_parser("consolidate")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=_cmd_consolidate)
    p = sub.add_parser("backfill")
    p.add_argument("--since", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--max-segments", type=_positive)
    p.set_defaults(fn=_cmd_backfill)
    p = sub.add_parser("rollback")
    p.add_argument("run_id", nargs="?")
    p.add_argument("--memory", metavar="RUN_ID", help="restore memory from a consolidate run's archive")
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=_cmd_rollback)
    p = sub.add_parser("recent", help="what the last night's runs did")
    p.add_argument("--hours", type=_positive, help="the last N hours instead of the night")
    p.add_argument("-v", "--verbose", action="store_true", help="every decision, with the model's notes")
    p.set_defaults(fn=_cmd_recent)
    p = sub.add_parser("guard", help="what the hook shows when a command, an edited file or an error matches a trigger")
    p.add_argument("action", choices=("list", "set", "unset", "test", "on", "off", "add", "remove", "block", "unblock"))
    p.add_argument("args", nargs="*", help="set: <lesson id> <regex>; unset: <lesson id>; test: <command or path>; "
                                           "add: <name> <regex> --text …; remove: <name>; "
                                           "block, unblock: <lesson id>")
    p.add_argument("--path", action="store_true", help="the regex is over the path of an edited file")
    p.add_argument("--error", action="store_true", help="the regex is over what a failed shell command printed")
    p.add_argument("--text", help="add: what the agent is shown")
    p.add_argument("--project", help="add: the project root the rule is for (every project when left out)")
    p.set_defaults(fn=_cmd_guard)
    p = sub.add_parser("baseline", help="count past days' returned work with today's detector; writes no lesson")
    p.add_argument("--since", required=True, metavar="YYYY-MM-DD")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="the last day counted (yesterday when left out)")
    p.add_argument("--dry-run", action="store_true", help="say how many sessions and calls it would take")
    p.add_argument("--max-calls", type=_positive, help="stop after this many model calls; run again to go on")
    p.set_defaults(fn=_cmd_baseline)
    p = sub.add_parser("approve", help="put a waiting lesson into the index now")
    p.add_argument("lesson_id")
    p.set_defaults(fn=_cmd_approve)
    p = sub.add_parser("confirm", help="a fact was checked again and still holds")
    p.add_argument("lesson_id")
    p.set_defaults(fn=_cmd_confirm)
    p = sub.add_parser("retire", help="take a lesson's line out of the index, keeping its file")
    p.add_argument("lesson_id")
    p.set_defaults(fn=_cmd_retire)
    sub.add_parser("staged", help="the lessons that wait for a second case").set_defaults(fn=_cmd_staged)
    for name, fn in (("install", _cmd_install), ("uninstall", _cmd_uninstall), ("status", _cmd_status)):
        sub.add_parser(name).set_defaults(fn=fn)
    return parser


# the commands launchd starts on its own; the rest are typed by the user
SCHEDULED = tuple(launchd.LABELS)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.cmd == "backfill":
        try:
            datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print("agent-reflect: --since must be YYYY-MM-DD")
            return 2
    try:
        cfg = Config.default()
    except ConfigError as exc:
        print(f"agent-reflect: {exc}", file=sys.stderr)
        if args.cmd in SCHEDULED:  # launchd runs it: stderr alone would be a silent failure
            i18n.use(i18n.resolve("auto"))
            digest.notify(i18n.t(f"agent-reflect: {args.cmd} did not run, the config file is unusable: {exc}",
                                 f"agent-reflect: {args.cmd} не запустился, файл настроек не читается: {exc}"))
        return 2
    i18n.use(cfg.language)
    return args.fn(cfg, args)
