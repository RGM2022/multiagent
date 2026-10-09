import ast
import re
from dataclasses import dataclass
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
}
IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
MAX_FILES = 8000
MAX_FILE_BYTES = 512_000
MAX_EVIDENCE = 80

SYMBOL_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:"
    r"(?:public|private|protected|static|abstract|final|virtual|override)\s+)*"
    r"(class|interface|struct|enum|trait|func|function|def|fn|"
    r"module|namespace)\s+([A-Za-z_$][\w$]*)"
)
METHOD_PATTERN = re.compile(
    r"^\s*(?:(?:public|private|protected|static|abstract|final|virtual|"
    r"override|async|synchronized|inline|extern|const)\s+)*"
    r"(?:[\w.<>,?\[\]&:*]+\s+)?([A-Za-z_$][\w$]*)\s*"
    r"\([^;]*\)\s*(?::\s*[\w.<>,?\[\]| &*]+)?\s*(?:\{|=>|$)"
)
CONTROL_WORDS = {"catch", "for", "if", "switch", "while"}
WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")
STOP_WORDS = {
    "analysis",
    "about",
    "after",
    "agent",
    "also",
    "and",
    "are",
    "build",
    "capability",
    "code",
    "could",
    "create",
    "feature",
    "from",
    "have",
    "impact",
    "into",
    "jira",
    "method",
    "need",
    "request",
    "requirement",
    "requirements",
    "should",
    "system",
    "that",
    "their",
    "this",
    "workflow",
    "with",
    "would",
}


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    start_line: int
    end_line: int


def extract_search_terms(text: str) -> list[str]:
    words = {
        match.group(0).lower()
        for match in WORD_PATTERN.finditer(text)
        if match.group(0).lower() not in STOP_WORDS
    }
    return sorted(words)


def _python_symbols(source: str) -> list[Symbol]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[Symbol] = []

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, child.name))
                kind = "class" if isinstance(child, ast.ClassDef) else "method/function"
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                    )
                )
                visit(child, (*parents, child.name))
            else:
                visit(child, parents)

    visit(tree)
    return symbols


def _symbols_for_file(source: str, suffix: str) -> list[Symbol]:
    if suffix == ".py":
        return _python_symbols(source)

    symbols: list[Symbol] = []
    class_stack: list[tuple[int, str]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        indent = len(line) - len(line.lstrip())
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()

        declaration = SYMBOL_PATTERN.match(line)
        if declaration:
            kind, name = declaration.groups()
            if kind == "class":
                class_stack.append((indent, name))
            symbols.append(
                Symbol(
                    name=name,
                    kind="declaration",
                    start_line=line_number,
                    end_line=line_number,
                )
            )
            continue

        method = METHOD_PATTERN.match(line)
        if method:
            name = method.group(1)
            if name in CONTROL_WORDS:
                continue
            qualified_name = (
                f"{class_stack[-1][1]}.{name}" if class_stack else name
            )
            symbols.append(
                Symbol(
                    name=qualified_name,
                    kind="method/function",
                    start_line=line_number,
                    end_line=line_number,
                )
            )
    return symbols


def _symbol_at_line(symbols: list[Symbol], line_number: int) -> str:
    containing = [
        symbol
        for symbol in symbols
        if symbol.start_line <= line_number <= symbol.end_line
    ]
    if containing:
        return min(
            containing,
            key=lambda symbol: symbol.end_line - symbol.start_line,
        ).name
    preceding = [symbol for symbol in symbols if symbol.start_line <= line_number]
    return preceding[-1].name if preceding else ""


def scan_repository(
    repo_path: str,
    search_terms: list[str],
) -> tuple[list[dict[str, object]], list[str]]:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    terms = {term.lower() for term in search_terms if len(term) >= 3}
    evidence: list[dict[str, object]] = []
    warnings: list[str] = []
    scanned_files = 0

    for file_path in root.rglob("*"):
        if any(part in IGNORED_DIRECTORIES for part in file_path.parts):
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if scanned_files >= MAX_FILES:
            warnings.append(
                f"Stopped after scanning {MAX_FILES} source files; narrow the repository path for complete coverage."
            )
            break
        scanned_files += 1

        try:
            if file_path.stat().st_size > MAX_FILE_BYTES:
                continue
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            warnings.append(f"Could not read {file_path.relative_to(root)}: {error}")
            continue

        symbols = _symbols_for_file(source, file_path.suffix.lower())
        for line_number, line in enumerate(source.splitlines(), start=1):
            normalized_line = line.lower()
            matched_terms = sorted(term for term in terms if term in normalized_line)
            if not matched_terms:
                continue

            evidence.append(
                {
                    "id": f"E{len(evidence) + 1:03d}",
                    "path": file_path.relative_to(root).as_posix(),
                    "line": line_number,
                    "symbol": _symbol_at_line(symbols, line_number),
                    "snippet": line.strip()[:300],
                    "matched_terms": matched_terms,
                    "score": len(matched_terms),
                }
            )

    evidence.sort(
        key=lambda item: (
            -int(item["score"]),
            str(item["path"]),
            int(item["line"]),
        )
    )
    evidence = evidence[:MAX_EVIDENCE]
    for index, item in enumerate(evidence, start=1):
        item["id"] = f"E{index:03d}"

    return evidence, warnings
