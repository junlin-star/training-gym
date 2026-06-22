"""CLI entry point: ``training-gym <command>``."""

from __future__ import annotations

import argparse
import sys


def main():
    from modal_training_gym.setup import setup, open_dashboard
    from modal_training_gym.cleanup import cleanup

    parser = argparse.ArgumentParser(prog="training-gym")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Deploy the training-gym dashboard to Modal")

    sub.add_parser("open", help="Open the deployed dashboard in your browser")

    cleanup_parser = sub.add_parser(
        "cleanup",
        help="Delete metadata for old failed runs",
    )
    cleanup_parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        metavar="DAYS",
        help="Delete failed runs older than this many days (default: 7)",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "open":
        open_dashboard()
    elif args.command == "cleanup":
        cleanup(older_than_days=args.older_than_days, dry_run=args.dry_run)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
