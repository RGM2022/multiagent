import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


GITHUB_API = "https://api.github.com"
BRANCH_PREFIX = "copilot/jira-impact"


def parse_github_url(repository_url: str) -> tuple[str, str, str]:
    parsed = urlsplit(repository_url.strip())
    parts = parsed.path.strip("/").removesuffix(".git").split("/")
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
        or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
    ):
        raise ValueError(
            "Use a GitHub HTTPS repository URL such as "
            "https://github.com/owner/repository."
        )
    owner, repository = parts
    clone_url = f"https://github.com/{owner}/{repository}.git"
    return owner, repository, clone_url


def _git_environment(token: str) -> dict[str, str]:
    environment = os.environ.copy()
    credentials = base64.b64encode(f"x-access-token:{token}".encode("utf-8"))
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": (
                f"AUTHORIZATION: basic {credentials.decode('ascii')}"
            ),
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _run_git(
    repo_path: str | Path,
    *arguments: str,
    token: str | None = None,
) -> str:
    command = ["git", "-C", str(repo_path), *arguments]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(token) if token else None,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"Git command failed: {detail}") from error
    return result.stdout.strip()


def clone_github_repo(
    repository_url: str,
    destination: str | Path,
    token: str,
) -> tuple[str, str, str]:
    owner, repository, clone_url = parse_github_url(repository_url)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--", clone_url, str(destination)],
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(token),
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"Could not clone GitHub repository: {detail}") from error
    return owner, repository, clone_url


def _github_api_request(
    method: str,
    endpoint: str,
    token: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{GITHUB_API}{endpoint}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(response_body).get("message", response_body)
        except json.JSONDecodeError:
            message = response_body
        raise RuntimeError(
            f"GitHub API returned HTTP {error.code}: {message}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach GitHub: {error.reason}") from error


def _write_approved_changes(
    repo_path: Path,
    proposals: list[dict[str, object]],
) -> list[str]:
    root = repo_path.resolve()
    changed_paths: list[str] = []
    seen: set[str] = set()

    for proposal in proposals:
        relative_path = str(proposal["path"])
        unresolved_path = root / relative_path
        target = unresolved_path.resolve()
        contains_symlink = any(
            ancestor.is_symlink()
            for ancestor in (unresolved_path, *unresolved_path.parents)
            if ancestor != root and root in ancestor.parents
        )
        if (
            contains_symlink
            or not target.is_relative_to(root)
            or not target.is_file()
        ):
            raise RuntimeError(
                f"Refusing to write outside the repository or to a non-file: "
                f"{relative_path}"
            )
        if relative_path in seen:
            raise RuntimeError(f"Duplicate change proposal for {relative_path}.")
        seen.add(relative_path)

        original_bytes = target.read_bytes()
        original_content = original_bytes.decode("utf-8")
        expected_hash = str(proposal["original_sha256"])
        current_hash = hashlib.sha256(original_bytes).hexdigest()
        if current_hash != expected_hash:
            raise RuntimeError(
                f"{relative_path} changed after review; refusing to overwrite it."
            )

        updated_content = str(proposal["updated_content"])
        normalized_original = original_content.replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        normalized_updated = updated_content.replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        if not normalized_updated.strip() or normalized_updated == normalized_original:
            raise RuntimeError(f"Refusing an empty or unchanged update to {relative_path}.")
        newline = "\r\n" if "\r\n" in original_content else "\n"
        target.write_text(
            normalized_updated.replace("\n", newline),
            encoding="utf-8",
            newline="",
        )
        changed_paths.append(relative_path)

    if not changed_paths:
        raise RuntimeError("No approved file changes are available to publish.")
    _run_git(repo_path, "diff", "--check")
    return changed_paths


def publish_approved_changes(
    repo_path: str,
    repository_url: str,
    requirement: str,
    proposals: list[dict[str, object]],
    jira_source: str = "",
) -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if not token:
        raise RuntimeError("Set GITHUB_TOKEN or GH_TOKEN to push and open a PR.")
    owner, repository, _ = parse_github_url(repository_url)
    repo = Path(repo_path)
    repository_details = _github_api_request(
        "GET",
        f"/repos/{owner}/{repository}",
        token,
    )
    base_branch = str(repository_details["default_branch"])
    title_requirement = " ".join(requirement.split())[:230]

    changed_paths = _write_approved_changes(repo, proposals)
    branch = f"{BRANCH_PREFIX}-{os.urandom(4).hex()}"
    _run_git(repo, "switch", "-c", branch)
    _run_git(repo, "add", "--", *changed_paths)
    _run_git(
        repo,
        "-c",
        "user.name=Copilot Impact Agent",
        "-c",
        "user.email=223556219+Copilot@users.noreply.github.com",
        "commit",
        "-m",
        f"Implement: {title_requirement[:60]}",
    )
    _run_git(repo, "push", "--set-upstream", "origin", branch, token=token)

    summary = "\n".join(
        f"- `{proposal['path']}` — {proposal['reason']}"
        for proposal in proposals
    )
    jira_line = f"\nJira issue: {jira_source}\n" if jira_source else ""
    body = (
        "## Requirement\n"
        f"{requirement}\n"
        f"{jira_line}\n"
        "## Changes\n"
        f"{summary}\n\n"
        "Generated by the Jira repository impact workflow and published only "
        "after human approval."
    )
    try:
        pull_request = _github_api_request(
            "POST",
            f"/repos/{owner}/{repository}/pulls",
            token,
            {
                "title": f"Implement: {title_requirement}",
                "head": branch,
                "base": base_branch,
                "body": body,
            },
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"Changes were pushed to branch {branch}, but the pull request "
            f"could not be created: {error}"
        ) from error
    return str(pull_request["html_url"])
