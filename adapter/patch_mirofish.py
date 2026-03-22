"""
Patches MiroFish's imports to use local Graphiti adapter instead of Zep Cloud.

Run this once after cloning MiroFish:
    python -m adapter.patch_mirofish

It rewrites `from zep_cloud.client import Zep` → `from adapter import Zep`
and `from zep_cloud import ...` → `from adapter import ...` / `from adapter.zep_types import ...`
across all MiroFish backend Python files.
"""

import re
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "mirofish" / "backend"

# Import rewrites: (pattern, replacement)
REWRITES = [
    # Main client import
    (
        r"from\s+zep_cloud\.client\s+import\s+Zep",
        "from adapter import Zep",
    ),
    # EpisodeData and EntityEdgeSourceTarget
    (
        r"from\s+zep_cloud\s+import\s+EpisodeData\s*,\s*EntityEdgeSourceTarget",
        "from adapter.zep_types import EpisodeData, EntityEdgeSourceTarget",
    ),
    # Single imports from zep_cloud
    (
        r"from\s+zep_cloud\s+import\s+EpisodeData",
        "from adapter.zep_types import EpisodeData",
    ),
    (
        r"from\s+zep_cloud\s+import\s+EntityEdgeSourceTarget",
        "from adapter.zep_types import EntityEdgeSourceTarget",
    ),
    (
        r"from\s+zep_cloud\s+import\s+InternalServerError",
        "from adapter.zep_types import InternalServerError",
    ),
    # Ontology imports — stub them out since Graphiti handles ontology internally
    (
        r"from\s+zep_cloud\.external_clients\.ontology\s+import\s+.*",
        "from adapter.ontology_stubs import EntityModel, EntityText, EdgeModel",
    ),
]


def patch_file(filepath: Path, dry_run: bool = False) -> bool:
    content = filepath.read_text(encoding="utf-8")
    original = content
    for pattern, replacement in REWRITES:
        content = re.sub(pattern, replacement, content)
    if content != original:
        if dry_run:
            print(f"  [DRY RUN] Would patch: {filepath.relative_to(VENDOR_DIR)}")
        else:
            filepath.write_text(content, encoding="utf-8")
            print(f"  Patched: {filepath.relative_to(VENDOR_DIR)}")
        return True
    return False


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("DRY RUN MODE — no files will be modified\n")

    py_files = list(VENDOR_DIR.rglob("*.py"))
    patched = 0
    for f in py_files:
        if patch_file(f, dry_run=dry_run):
            patched += 1

    print(f"\nDone. {patched}/{len(py_files)} files {'would be ' if dry_run else ''}patched.")


if __name__ == "__main__":
    main()
