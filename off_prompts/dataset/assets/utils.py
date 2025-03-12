"""Useful tools."""


def format_runtime(start: int, finish: int):
    """Format runtime of a script."""
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"
