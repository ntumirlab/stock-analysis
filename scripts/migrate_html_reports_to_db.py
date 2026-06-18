"""Migrate existing HTML report files into golden_ai_backtest_reports DB table.

Scans assets/<StrategyDir>/*.html, extracts reportJson + positionJson from
lines 9-10, and INSERTs into golden_ai_backtest_reports.

Usage:
    python3 scripts/migrate_html_reports_to_db.py [options]

Options:
    --strategy weekly|monthly|weekly_4w   Migrate only one strategy (default: all)
    --dry-run                             Count files without writing to DB
    --batch-size N                        Commit every N inserts (default: 500)
"""
import os
import re
import sys
import time
import sqlite3
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(PROJECT_ROOT, 'assets')

STRATEGY_DIRS = {
    'weekly':    'GoldenAITWStrategyWeekly',
    'monthly':   'GoldenAITWStrategyMonthly',
    'weekly_4w': 'GoldenAITWStrategyWeekly4W',
}

FILE_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_Ranks\[(.+?)\](?:_(Week\d))?\.html$'
)


def _extract_json(html_path):
    with open(html_path, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(11)]
    rj = re.search(r'const reportJson = (.+)</script>', lines[8])
    pj = re.search(r'const positionJson = (.+)</script>', lines[9])
    if not rj or not pj:
        return None, None
    return rj.group(1).rstrip('; '), pj.group(1).rstrip('; ')


def migrate(strategy, db_path, dry_run=False, batch_size=500):
    dir_name = STRATEGY_DIRS[strategy]
    report_dir = os.path.join(ASSETS, dir_name)
    if not os.path.isdir(report_dir):
        print(f"  {dir_name}/ not found, skipping")
        return 0

    files = [f for f in os.listdir(report_dir) if f.endswith('.html')]
    total = len(files)
    print(f"  {dir_name}/: {total} HTML files")

    if dry_run or total == 0:
        return total

    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    inserted = 0
    skipped = 0
    errors = 0

    for i, fname in enumerate(files, 1):
        m = FILE_PATTERN.match(fname)
        if not m:
            skipped += 1
            continue

        date_str, time_str, ranks_str, week_str = m.groups()
        timestamp = f"{date_str} {time_str.replace('-', ':')}"

        existing = conn.execute(
            "SELECT 1 FROM golden_ai_backtest_reports "
            "WHERE strategy=? AND timestamp=? AND ranks=? AND week IS ? LIMIT 1",
            (strategy, timestamp, ranks_str, week_str)
        ).fetchone()
        if existing:
            skipped += 1
            continue

        html_path = os.path.join(report_dir, fname)
        try:
            rj, pj = _extract_json(html_path)
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
            errors += 1
            continue

        if not rj:
            print(f"  WARN: could not extract JSON from {fname}")
            errors += 1
            continue

        conn.execute(
            "INSERT INTO golden_ai_backtest_reports "
            "(timestamp, strategy, week, ranks, report_json, position_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, strategy, week_str, ranks_str, rj, pj)
        )
        inserted += 1

        if inserted % batch_size == 0:
            conn.commit()
            print(f"  [{i}/{total}] committed {inserted} rows...")

    conn.commit()
    conn.close()
    print(f"  Done: {inserted} inserted, {skipped} skipped, {errors} errors")
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Migrate HTML reports to DB')
    parser.add_argument('--strategy', choices=['weekly', 'monthly', 'weekly_4w'],
                        help='Migrate only one strategy')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count files without writing')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--db', default=os.path.join(PROJECT_ROOT, 'data_prod.db'))
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else ['weekly', 'monthly', 'weekly_4w']

    print(f"DB: {args.db}")
    print(f"Dry run: {args.dry_run}")
    print()

    t_start = time.monotonic()
    total_inserted = 0
    for s in strategies:
        print(f"[{s}]")
        total_inserted += migrate(s, args.db, dry_run=args.dry_run, batch_size=args.batch_size)

    elapsed = time.monotonic() - t_start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\nTotal: {total_inserted} rows in {mins}m {secs}s")


if __name__ == '__main__':
    main()
