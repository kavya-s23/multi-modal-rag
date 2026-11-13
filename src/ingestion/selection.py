from .registry import status
from .scanner import FileInfo, ScanResult


def parse_selection(text: str, count: int) -> list[int] | None:
    """'all', '1,3' or '2-4' -> zero-based indexes. None if invalid."""
    text = text.strip().lower()
    if text == "all":
        return list(range(count))
    chosen: set[int] = set()
    try:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = (int(x) for x in part.split("-"))
                chosen.update(range(a, b + 1))
            elif part:
                chosen.add(int(part))
    except ValueError:
        return None
    if not chosen or min(chosen) < 1 or max(chosen) > count:
        return None
    return sorted(i - 1 for i in chosen)


def choose_files(result: ScanResult, cfg: dict) -> list[FileInfo]:
    """A single file is used as it is. A folder asks the user to pick."""
    limit = cfg["ingestion"]["max_files_per_run"]

    if len(result.files) == 1 and not result.skipped:
        return result.files

    print(f"\nFound {len(result.files)} supported file(s):")
    for i, f in enumerate(result.files, start=1):
        tag = {"new": "", "indexed": "  [already indexed]",
               "changed": "  [changed since last index]"}[status(f)]
        print(f" {i}. {f.name}  ({f.size / 1024:.0f} KB){tag}")
    if result.skipped:
        print("Skipped (unsupported):", ", ".join(result.skipped))
    print(f"You can select up to {limit} file(s) at a time.")

    while True:
        picked = parse_selection(
            input("Select files (e.g. 1,3 or 2-4 or all): "), len(result.files))
        if picked is None:
            print("Invalid selection, try again.")
        elif len(picked) > limit:
            print(f"You selected {len(picked)} files; the limit is {limit}. ")
        else:
            return [result.files[i] for i in picked]