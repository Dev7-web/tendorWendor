from __future__ import annotations

import argparse
import sys

from app.pipeline.run import main as run_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m app.pipeline")
    parser.add_argument("command", nargs="?", help="run")

    if not argv:
        parser.print_help()
        return 2

    if argv[0] == "run":
        return run_main(argv[1:])

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
