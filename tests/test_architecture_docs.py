"""Drift guard for the two architecture views: the Mermaid diagrams in `docs/architecture.md` (view A) and the
explorer page `docs/architecture-explorer.html` (view B, its data is one JSON block). Both must name only things that
exist in `src/vaice` and must cover every module and every named thread, so a rename, a removal or a new module or
thread fails here instead of leaving a diagram that lies. Everything is read by path from the repository (conftest
chdirs to a temp dir, so the working directory is never used) and the code is parsed with `ast`, never imported."""
import ast
import json
import re
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "vaice"
ARCHITECTURE_MD = ROOT / "docs" / "architecture.md"
EXPLORER_HTML = ROOT / "docs" / "architecture-explorer.html"

PY_MODULES = frozenset(p.stem for p in SOURCE.glob("*.py") if p.stem not in {"__init__", "__main__"})
MODULES = PY_MODULES | {"browse_gate"}  # browse_gate is src/vaice/browse_gate.ts, the Pi extension

DATA_BLOCK = re.compile(r'<script[^>]*\bid="architecture-data"[^>]*>(.*?)</script>', re.DOTALL)
FILE_TOKEN = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\.(py|ts)\b")
THREAD_TOKEN = re.compile(r"vaice-[a-z]+(?:-[a-z]+)*")
REF_TOKEN = re.compile(
    r"(?<![\w/.])(" + "|".join(sorted(PY_MODULES, key=len, reverse=True)) + r")\."
    r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
REF_KEYS = {"ref", "started_by", "runs"}
FILE_EXTENSIONS = {"py", "ts", "json", "md", "html", "txt", "wav", "env"}  # `memory.json` is a file, not a ref


# --- the code side -----------------------------------------------------------------------------------------------

@cache
def parse_module(module: str) -> ast.Module:
    return ast.parse((SOURCE / f"{module}.py").read_text(encoding="utf-8"))


def source_threads() -> set[str]:
    """The `name=` of every `threading.Thread(...)` / `Thread(...)` call in src/vaice that gives a constant string."""
    names = set()
    for module in PY_MODULES:
        for node in ast.walk(parse_module(module)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called != "Thread":
                continue
            for keyword in node.keywords:
                if (keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)):
                    names.add(keyword.value.value)
    return names


def statements(node: ast.AST):
    """The statements directly in a module/class/function body, looking through if/for/while/with/try blocks but not
    into nested definitions (those are looked up by name)."""
    for field in ("body", "orelse", "finalbody"):
        for stmt in getattr(node, field, []):
            yield stmt
            if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                yield from statements(stmt)
    for handler in getattr(node, "handlers", []):
        yield from statements(handler)


def assigned_names(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, ast.AnnAssign):
        targets = [stmt.target]
    else:
        return set()
    return {t.id for t in targets if isinstance(t, ast.Name)}


def init_attributes(cls: ast.ClassDef) -> set[str]:
    """Names set as `self.<name> = ...` / `self.<name>: T = ...` anywhere inside the class's `__init__`."""
    names = set()
    for stmt in statements(cls):
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            for node in ast.walk(stmt):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                else:
                    continue
                names |= {t.attr for t in targets if isinstance(t, ast.Attribute)
                          and isinstance(t.value, ast.Name) and t.value.id == "self"}
    return names


def resolve(module: str, qualname: str) -> bool:
    """Does `<module>.<qualname>` name something in the module's AST (class, function, nested function, assigned
    name, class attribute or a `self.x` set in `__init__`)? Refs into browse_gate are not resolvable."""
    if module not in PY_MODULES or not qualname:
        return False
    node: ast.AST = parse_module(module)
    parts = qualname.split(".")
    for i, part in enumerate(parts):
        leaf = i == len(parts) - 1
        found = None
        for stmt in statements(node):
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and stmt.name == part:
                found = stmt
                break
            if leaf and part in assigned_names(stmt):
                return True
        if found is None:
            return leaf and isinstance(node, ast.ClassDef) and part in init_attributes(node)
        node = found
    return True


def unresolved(ref: str) -> bool:
    module, _, qualname = ref.partition(".")
    return not resolve(module, qualname)


# --- view A: architecture.md -------------------------------------------------------------------------------------

def read_doc(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} does not exist"
    return path.read_text(encoding="utf-8")


def mermaid_blocks(text: str) -> list[str]:
    """The body of every ```mermaid fenced block."""
    return re.findall(r"^```mermaid[ \t]*\n(.*?)^```", text, re.DOTALL | re.MULTILINE)


def view_block(text: str, view: str) -> str:
    """The one mermaid block marked `%% view: <view>`."""
    blocks = [b for b in mermaid_blocks(text) if re.search(rf"^\s*%%\s*view:\s*{view}\s*$", b, re.MULTILINE)]
    assert len(blocks) == 1, f"expected exactly one mermaid block with `%% view: {view}`, found {len(blocks)}"
    return blocks[0]


def block_modules(block: str) -> set[str]:
    return {stem for stem, _ in FILE_TOKEN.findall(block)} & MODULES


def block_threads(block: str) -> set[str]:
    return set(THREAD_TOKEN.findall(block))


def doc_ref_tokens(text: str):
    """(line number, ref) for each ref-shaped token in the mermaid blocks and the inline backtick spans."""
    in_fence = None
    for number, line in enumerate(text.splitlines(), start=1):
        fence = re.match(r"^\s*```(\w*)", line)
        if fence:
            in_fence = None if in_fence is not None else (fence.group(1) or "plain")
            continue
        if in_fence == "mermaid":
            scopes = [line]
        elif in_fence is None:
            scopes = re.findall(r"`([^`]+)`", line)
        else:
            scopes = []
        for scope in scopes:
            for match in REF_TOKEN.finditer(scope):
                if match.group(2) not in FILE_EXTENSIONS:
                    yield number, f"{match.group(1)}.{match.group(2)}"


# --- view B: architecture-explorer.html --------------------------------------------------------------------------

def explorer_data(text: str) -> dict:
    match = DATA_BLOCK.search(text)
    assert match, 'no <script type="application/json" id="architecture-data"> block in the explorer page'
    return json.loads(match.group(1))


def explorer() -> dict:
    return explorer_data(read_doc(EXPLORER_HTML))


def string_fields(value, key=None):
    """Every (key, string) under the JSON, with list items taking their list's key."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from string_fields(v, k)
    elif isinstance(value, list):
        for item in value:
            yield from string_fields(item, key)
    elif isinstance(value, str):
        yield key, value


def schema_problems(data: dict) -> list[str]:
    problems = []

    def need(where: str, obj, keys: dict[str, type]):
        if not isinstance(obj, dict):
            problems.append(f"{where}: not an object")
            return False
        for key, kind in keys.items():
            if key not in obj:
                problems.append(f"{where}: missing `{key}`")
            elif not isinstance(obj[key], kind):
                problems.append(f"{where}: `{key}` is not a {kind.__name__}")
        return True

    problems += [f"top level: `{key}` is missing or not a list" for key in ("modules", "threads", "flows", "states")
                 if not isinstance(data.get(key), list)]
    module_ids = {m.get("id") for m in data.get("modules", []) if isinstance(m, dict)}
    for m in data.get("modules", []):
        where = f"module {m.get('id') if isinstance(m, dict) else m!r}"
        if not need(where, m, {"id": str, "file": str, "layer": str, "summary": str, "role": str,
                               "symbols": list, "rules": list, "uses": list}):
            continue
        if isinstance(m["file"], str) and not (ROOT / m["file"]).is_file():
            problems.append(f"{where}: file `{m['file']}` does not exist")
        for symbol in m["symbols"] if isinstance(m["symbols"], list) else []:
            need(f"{where} symbol {symbol!r}", symbol, {"ref": str, "what": str})
        if not all(isinstance(r, str) for r in m["rules"] if isinstance(m["rules"], list)):
            problems.append(f"{where}: `rules` must hold strings")
        problems += [f"{where}: `uses` names unknown module `{u}`" for u in m["uses"] if u not in module_ids]
    for t in data.get("threads", []):
        need(f"thread {t.get('name') if isinstance(t, dict) else t!r}", t,
             {"name": str, "started_by": str, "runs": str, "lifetime": str, "talks": object})
    for f in data.get("flows", []):
        where = f"flow {f.get('id') if isinstance(f, dict) else f!r}"
        if not need(where, f, {"id": str, "title": str, "summary": str, "lanes": list, "steps": list}):
            continue
        for i, step in enumerate(f["steps"]):
            if need(f"{where} step {i}", step, {"lane": str, "text": str}) and step["lane"] not in f["lanes"]:
                problems.append(f"{where} step {i}: lane `{step['lane']}` is not in the flow's lanes")
    for s in data.get("states", []):
        where = f"state machine {s.get('id') if isinstance(s, dict) else s!r}"
        if not need(where, s, {"id": str, "title": str, "nodes": list, "edges": list}):
            continue
        node_ids = {n.get("id") for n in s["nodes"] if isinstance(n, dict)}
        for n in s["nodes"]:
            need(f"{where} node {n!r}", n, {"id": str, "label": str})
        for e in s["edges"]:
            if need(f"{where} edge {e!r}", e, {"from": str, "to": str, "label": str}):
                problems += [f"{where} edge {e['from']}->{e['to']}: `{end}` is not a node id"
                             for end in (e["from"], e["to"]) if end not in node_ids]
    return problems


def diff_message(what: str, expected: set[str], actual: set[str], expected_name: str, actual_name: str) -> str:
    return (f"{what}: in {expected_name} but not in {actual_name}: {sorted(expected - actual)}; "
            f"in {actual_name} but not in {expected_name}: {sorted(actual - expected)}")


# --- the resolver itself -----------------------------------------------------------------------------------------

def test_resolve_finds_real_symbols():
    for module, qualname in [("realtime", "_converse"), ("assistant", "Assistant._start_browse"),
                             ("interruptible", "LEAD_SECONDS"), ("lifecycle", "stopping"),
                             ("conversation", "Turn.closed")]:
        assert resolve(module, qualname), f"{module}.{qualname} should resolve"


def test_resolve_finds_nested_functions_and_class_attributes():
    assert resolve("app", "build_stream.reset")
    assert resolve("confirm", "Pending.waiter")


def test_resolve_rejects_unknown_names():
    for module, qualname in [("realtime", "nope"), ("nosuchmodule", "x"), ("assistant", "Assistant.nope"),
                             ("assistant", "Nope._start_browse"), ("browse_gate", "x"), ("realtime", "")]:
        assert not resolve(module, qualname), f"{module}.{qualname} should not resolve"


def test_source_threads_are_found():
    assert {"vaice-realtime", "vaice-mcp", "vaice-browse-worker"} <= source_threads()


# --- view B ------------------------------------------------------------------------------------------------------

def test_explorer_data_block_parses():
    data = explorer()
    assert isinstance(data, dict)


def test_explorer_schema():
    problems = schema_problems(explorer())
    assert not problems, "explorer data breaks its schema:\n" + "\n".join(problems)


def test_explorer_modules_match_source():
    ids = {m["id"] for m in explorer()["modules"]}
    assert ids == MODULES, diff_message("explorer modules", MODULES, ids, "src/vaice", "the explorer")


def test_explorer_threads_match_source():
    names = {t["name"] for t in explorer()["threads"]}
    threads = source_threads()
    assert names == threads, diff_message("explorer threads", threads, names, "src/vaice", "the explorer")


def test_explorer_refs_resolve():
    bad = [f"{key}: {value}" for key, value in string_fields(explorer()) if key in REF_KEYS and unresolved(value)]
    assert not bad, "explorer refs that do not resolve in src/vaice:\n" + "\n".join(bad)


# --- view A ------------------------------------------------------------------------------------------------------

def test_architecture_has_mermaid_blocks():
    assert mermaid_blocks(read_doc(ARCHITECTURE_MD)), "docs/architecture.md has no ```mermaid block"


def test_architecture_modules_view_covers_every_module():
    named = block_modules(view_block(read_doc(ARCHITECTURE_MD), "modules"))
    assert named >= MODULES, f"the `%% view: modules` diagram does not mention: {sorted(MODULES - named)}"


def test_architecture_file_tokens_exist():
    bad = sorted({f"{stem}.{ext}" for block in mermaid_blocks(read_doc(ARCHITECTURE_MD))
                  for stem, ext in FILE_TOKEN.findall(block) if not (SOURCE / f"{stem}.{ext}").is_file()})
    assert not bad, f"mermaid blocks name files that are not in src/vaice: {bad}"


def test_architecture_threads_view_matches_source():
    named = block_threads(view_block(read_doc(ARCHITECTURE_MD), "threads"))
    threads = source_threads()
    assert named == threads, diff_message("`%% view: threads` diagram", threads, named, "src/vaice", "the diagram")


def test_architecture_refs_resolve():
    bad = [f"line {number}: {ref}" for number, ref in doc_ref_tokens(read_doc(ARCHITECTURE_MD)) if unresolved(ref)]
    assert not bad, "docs/architecture.md refs that do not resolve in src/vaice:\n" + "\n".join(bad)


# --- both views --------------------------------------------------------------------------------------------------

def test_views_agree_on_modules():
    diagram = block_modules(view_block(read_doc(ARCHITECTURE_MD), "modules"))
    page = {m["id"] for m in explorer()["modules"]}
    assert diagram == page, diff_message("the two views disagree on modules", diagram, page,
                                         "architecture.md", "the explorer")


def test_views_agree_on_threads():
    diagram = block_threads(view_block(read_doc(ARCHITECTURE_MD), "threads"))
    page = {t["name"] for t in explorer()["threads"]}
    assert diagram == page, diff_message("the two views disagree on threads", diagram, page,
                                         "architecture.md", "the explorer")
