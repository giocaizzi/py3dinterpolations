"""Generate API reference pages for mkdocstrings."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
root = Path("py3dinterpolations")
ref = Path("reference")

for path in sorted(root.rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = ref / doc_path

    parts = tuple(module_path.parts)

    # Skip __pycache__ and private modules (except __init__)
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav_parts = list(parts)

    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

with mkdocs_gen_files.open(ref / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
