import argparse
import base64
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from typing import Any

from langgraph.types import Command

from jira_impact.github import clone_github_repo, parse_github_url
from jira_impact.workflow import build_workflow


JIRA_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")


def _extract_adf_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(filter(None, (_extract_adf_text(item) for item in value)))
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        return " ".join(
            filter(None, (_extract_adf_text(item) for item in value.values()))
        )
    return ""


def fetch_jira_issue(issue_key: str) -> tuple[str, str]:
    base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
    email = os.getenv("JIRA_EMAIL", "")
    api_token = os.getenv("JIRA_API_TOKEN", "")
    if not all((base_url, email, api_token)):
        raise RuntimeError(
            "To use a Jira issue key, set JIRA_BASE_URL, JIRA_EMAIL, and "
            "JIRA_API_TOKEN; alternatively, enter a plain-text capability."
        )

    credentials = base64.b64encode(f"{email}:{api_token}".encode("utf-8")).decode(
        "ascii"
    )
    request = urllib.request.Request(
        f"{base_url}/rest/api/3/issue/{issue_key}?fields=summary,description",
        headers={
            "Accept": "application/json",
            "Authorization": f"Basic {credentials}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Jira returned HTTP {error.code} for issue {issue_key}."
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach Jira: {error.reason}") from error

    fields = payload.get("fields", {})
    summary = str(fields.get("summary") or issue_key)
    description = _extract_adf_text(fields.get("description"))
    return issue_key, f"{summary}\n\n{description}".strip()


def _print_review(state: dict[str, Any]) -> None:
    print("\n=== Repository impact analysis ===")
    if state.get("jira_source"):
        print(f"Jira issue: {state['jira_source']}")
    print(f"Requirement: {state.get('requirement_summary', '')}")

    functional_requirements = state.get("functional_requirements", [])
    if functional_requirements:
        print("\nFunctional requirements:")
        for requirement in functional_requirements:
            print(f"  - {requirement}")

    evidence_by_id = {
        str(item["id"]): item for item in state.get("evidence", [])
    }
    proposals = state.get("proposed_changes", [])
    if not proposals:
        print("\nNo evidence-backed file or symbol candidates were found.")

    for number, proposal in enumerate(proposals, start=1):
        print(
            f"\n{number}. {proposal['path']} - "
            f"{proposal.get('symbol') or '(symbol not identified)'}"
        )
        print(f"   Suggested change: {proposal.get('change', '')}")
        print(f"   Reason: {proposal.get('reason', '')}")
        if proposal.get("patch"):
            print("   Proposed code diff:")
            print(
                "\n".join(
                    f"     {line}" for line in str(proposal["patch"]).splitlines()
                )
            )
        for evidence_id in proposal.get("evidence_ids", []):
            item = evidence_by_id[str(evidence_id)]
            print(
                f"   Evidence [{evidence_id}] {item['path']}:{item['line']} "
                f"({item.get('symbol') or 'module'})"
            )
            print(f"     {item['snippet']}")

    for warning in state.get("scan_warnings", []):
        print(f"\nScan warning: {warning}")
    for error in state.get("validation_errors", []):
        print(f"\nValidation: {error}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a Jira feature against a local or GitHub repository. "
            "GitHub changes are applied only after human review and published "
            "as a pull request."
        )
    )
    repository = parser.add_mutually_exclusive_group()
    repository.add_argument(
        "--repo",
        help="Local repository directory to scan (default: current directory).",
    )
    repository.add_argument(
        "--github-url",
        help="GitHub HTTPS repository URL; approved changes are pushed in a new branch and opened as a PR.",
    )
    parser.add_argument(
        "--requirement",
        help="Jira issue key (for example PROJ-123) or plain-text capability.",
    )
    parser.add_argument(
        "--model",
        help="OpenAI-compatible chat model (requires OPENAI_API_KEY).",
    )
    return parser.parse_args(argv)


def _run_analysis(
    repo_path: str,
    requirement: str,
    jira_source: str,
    model_name: str,
    github_url: str = "",
) -> int:
    workflow = build_workflow()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "requirement_input": requirement,
        "repo_path": repo_path,
        "jira_source": jira_source,
        "model_name": model_name,
        "github_url": github_url,
    }
    state = workflow.invoke(initial_state, config=config)

    _print_review(state)
    if github_url and not state.get("proposed_changes"):
        print(
            "\nNo validated code changes are available; no branch or pull "
            "request was created.",
            file=sys.stderr,
        )
        return 2

    if github_url:
        prompt = "\nApprove these code changes, push a branch, and open a PR? [y/N]: "
    else:
        prompt = "\nApprove this impact analysis? [y/N]: "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""
    approved = answer in {"y", "yes"}
    state = workflow.invoke(
        Command(resume={"approved": approved}),
        config=config,
    )
    if not state.get("approved"):
        print("\nImpact analysis rejected. No repository files were modified.")
        return 0
    if github_url:
        print(f"\nPull request created: {state['pull_request_url']}")
    else:
        print("\nImpact analysis approved. No repository files were modified.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    requirement = args.requirement
    if not requirement:
        requirement = input("Enter a Jira issue key or capability: ").strip()
    if not requirement:
        print("A Jira issue key or capability is required.", file=sys.stderr)
        return 2

    jira_source = ""
    if JIRA_KEY_PATTERN.fullmatch(requirement):
        try:
            jira_source, requirement = fetch_jira_issue(requirement)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"Jira lookup failed: {error}", file=sys.stderr)
            return 2

    model_name = args.model or ""
    if args.github_url:
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not token:
            print("Set GITHUB_TOKEN or GH_TOKEN to create a pull request.", file=sys.stderr)
            return 2
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "Set OPENAI_API_KEY to generate actual code changes for the "
                "GitHub pull request.",
                file=sys.stderr,
            )
            return 2

        try:
            parse_github_url(args.github_url)
            with tempfile.TemporaryDirectory(prefix="jira-impact-") as temp_dir:
                repo_path = os.path.join(temp_dir, "repository")
                clone_github_repo(args.github_url, repo_path, token)
                return _run_analysis(
                    repo_path,
                    requirement,
                    jira_source,
                    model_name,
                    github_url=args.github_url,
                )
        except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
            print(f"GitHub workflow failed: {error}", file=sys.stderr)
            return 2

    try:
        return _run_analysis(args.repo or ".", requirement, jira_source, model_name)
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"Repository analysis failed: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Workflow failed: {error}", file=sys.stderr)
        return 1
