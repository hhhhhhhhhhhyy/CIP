import argparse
import os
import sys
from pathlib import Path

def _is_runner_process() -> bool:
    try:
        stem = Path(sys.argv[0]).stem.lower()
    except Exception:
        return False

    if stem in {"uvicorn", "gunicorn", "hypercorn"}:
        return True

    # When launched via `python -m uvicorn ...`, sys.argv[0] can be `__main__`.
    # Detect typical ASGI invocation patterns to avoid choking on server flags.
    argv = list(sys.argv[1:])
    has_asgi_app = any(
        isinstance(a, str)
        and a
        and not a.startswith("-")
        and ":" in a
        and len(a.split(":", 1)[0]) > 0
        for a in argv
    )
    has_server_flags = any(a in {"--host", "--port", "--reload", "--workers"} for a in argv)
    return bool(has_asgi_app and has_server_flags)


def parse_parameters(argv=None, *, allow_unknown: bool | None = None):
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--mode", type=str, default="azure", help="Select openai or azure")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Select the model to use")
    parser.add_argument("--async_client", type=bool, default=False, help="Use async client or not")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p sampling value")
    parser.add_argument("--test_file_or_url", type=str, default = "Q:\\VSCode\\TianYang\\CIP\\test.xlsx", help="Path to the test file")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of pages to process concurrently")
    parser.add_argument("--work_type", type=str, default="C", help="Type of work: C or O")
    if argv is None:
        argv = sys.argv[1:]

    if allow_unknown is None:
        allow_unknown = _is_runner_process() or os.environ.get("CIP_ALLOW_UNKNOWN_ARGS") == "1"

    args, unknown = parser.parse_known_args(argv)
    if unknown and not allow_unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    return args