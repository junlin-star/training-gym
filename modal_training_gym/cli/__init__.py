"""CLI entry point: ``training-gym <command>``."""

from __future__ import annotations

import argparse
import sys


def main():
    from .setup import setup, open_dashboard, set_password, set_proxy_auth
    from .cleanup import cleanup

    parser = argparse.ArgumentParser(prog="training-gym")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("setup", help="Deploy the training-gym dashboard to Modal")

    sub.add_parser("open", help="Open the deployed dashboard in your browser")

    sub.add_parser(
        "set-proxy-auth",
        help="Set/replace the Modal proxy-auth tokens (MODAL_KEY / MODAL_SECRET)",
    )

    password_parser = sub.add_parser(
        "set-password",
        help="Set/clear the dashboard password (Basic Auth) and redeploy",
    )
    password_parser.add_argument(
        "--password",
        default=None,
        metavar="PASSWORD",
        help="Password to set (prompted securely if omitted; empty disables auth)",
    )

    cleanup_parser = sub.add_parser(
        "cleanup",
        help="Delete metadata for old failed/cancelled runs",
    )
    cleanup_parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        metavar="DAYS",
        help="Delete failed/cancelled runs older than this many days (default: 7)",
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
    elif args.command == "set-proxy-auth":
        set_proxy_auth()
    elif args.command == "set-password":
        set_password(password=args.password)
    elif args.command == "cleanup":
        cleanup(older_than_days=args.older_than_days, dry_run=args.dry_run)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
