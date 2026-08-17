#!/usr/bin/env python3
"""Keep the node dropdown in .github/ISSUE_TEMPLATE/bug-report.yml in sync
with the node display names registered in __init__.py.

Source of truth: the string values of NODE_DISPLAY_NAME_MAPPINGS in __init__.py.
We read them via `ast` (no imports), so this runs anywhere without torch/av/etc.
installed. It rewrites only the `options:` block under the dropdown with
`id: node`, leaving every other field in the form untouched.

Exit code is always 0 on success. The final printed line is `changed=0` or
`changed=1`, which the calling workflow uses to decide whether to commit.
"""
import ast
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INIT = ROOT / "__init__.py"
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug-report.yml"
EXTRA_OPTIONS = ["Other / general"]


def display_names():
    """Return the node display names (dict values, order-preserving, de-duped)."""
    tree = ast.parse(INIT.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "NODE_DISPLAY_NAME_MAPPINGS":
                    d = ast.literal_eval(node.value)
                    seen, out = set(), []
                    for name in d.values():
                        if name not in seen:
                            seen.add(name)
                            out.append(name)
                    return out
    sys.exit("ERROR: NODE_DISPLAY_NAME_MAPPINGS not found in __init__.py")


def quote(value: str) -> str:
    """Double-quote an option value so YAML stays valid for names with & ( )."""
    return '"' + value.replace('"', '\\"') + '"'


def main() -> None:
    names = display_names()
    lines = TEMPLATE.read_text().splitlines(keepends=True)

    # Locate the dropdown's `id: node` line.
    id_idx = next(
        (i for i, l in enumerate(lines) if l.strip() == "id: node"), None
    )
    if id_idx is None:
        sys.exit("ERROR: no 'id: node' dropdown found in bug-report.yml")

    # Locate its `options:` line (first one after the id).
    opt_idx = next(
        (j for j in range(id_idx, len(lines)) if lines[j].strip() == "options:"),
        None,
    )
    if opt_idx is None:
        sys.exit("ERROR: no 'options:' block under 'id: node' in bug-report.yml")

    # End of the options list: first line that is not a `- item`.
    end = opt_idx + 1
    while end < len(lines) and re.match(r"^\s+- ", lines[end]):
        end += 1

    new_options = [f"        - {quote(n)}\n" for n in names + EXTRA_OPTIONS]
    new_text = "".join(lines[: opt_idx + 1] + new_options + lines[end:])
    old_text = "".join(lines)

    if new_text != old_text:
        TEMPLATE.write_text(new_text)
        print(f"Synced {len(names)} registered nodes (+{len(EXTRA_OPTIONS)} fallback option) into the issue-form dropdown.")
        print("changed=1")
    else:
        print(f"Dropdown already in sync ({len(names)} nodes).")
        print("changed=0")


if __name__ == "__main__":
    main()
