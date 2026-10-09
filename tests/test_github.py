import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jira_impact.github import (
    _write_approved_changes,
    parse_github_url,
    publish_approved_changes,
)
from jira_impact.workflow import validation_agent


def run_git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class GitHubWorkflowTests(unittest.TestCase):
    def test_parse_github_url_accepts_repository_urls_only(self) -> None:
        self.assertEqual(
            parse_github_url("https://github.com/example/service.git"),
            (
                "example",
                "service",
                "https://github.com/example/service.git",
            ),
        )
        with self.assertRaises(ValueError):
            parse_github_url("https://example.com/example/service")

    def test_validation_creates_diff_for_cited_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            source = "class AccountService:\n    def validate_email(self):\n        return False\n"
            (repository / "accounts.py").write_text(source, encoding="utf-8")
            original_bytes = (repository / "accounts.py").read_bytes()
            result = validation_agent(
                {
                    "repo_path": str(repository),
                    "github_url": "https://github.com/example/service",
                    "editable_paths": ["accounts.py"],
                    "evidence": [
                        {
                            "id": "E001",
                            "path": "accounts.py",
                            "line": 2,
                            "symbol": "AccountService.validate_email",
                        }
                    ],
                    "proposed_changes": [
                        {
                            "path": "accounts.py",
                            "symbol": "AccountService.validate_email",
                            "change": "Validate email addresses",
                            "reason": "This method implements the requested validation.",
                            "evidence_ids": ["E001"],
                            "updated_content": source.replace(
                                "return False", "return True"
                            ),
                        }
                    ],
                }
            )

        self.assertFalse(result["validation_errors"])
        proposal = result["proposed_changes"][0]
        self.assertIn("-        return False", proposal["patch"])
        self.assertIn("+        return True", proposal["patch"])
        self.assertEqual(
            proposal["original_sha256"],
            hashlib.sha256(original_bytes).hexdigest(),
        )

    def test_publisher_refuses_file_modified_since_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            target = repository / "service.py"
            original = b"def validate():\n    return False\n"
            target.write_bytes(original)

            with self.assertRaisesRegex(RuntimeError, "changed after review"):
                _write_approved_changes(
                    repository,
                    [
                        {
                            "path": "service.py",
                            "original_sha256": "incorrect-review-hash",
                            "updated_content": "def validate():\n    return True\n",
                        }
                    ],
                )

            self.assertEqual(target.read_bytes(), original)

    def test_approved_changes_are_committed_pushed_and_opened_as_pr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "service.git"
            seed = root / "seed"
            checkout = root / "checkout"
            run_git("init", "--bare", "--initial-branch=main", str(remote))
            run_git("clone", str(remote), str(seed))
            run_git("-C", str(seed), "config", "user.name", "Test User")
            run_git(
                "-C",
                str(seed),
                "config",
                "user.email",
                "test@example.com",
            )
            source = b"def is_valid_email(value):\r\n    return False\r\n"
            target = seed / "service.py"
            target.write_bytes(source)
            run_git("-C", str(seed), "add", "service.py")
            run_git("-C", str(seed), "commit", "-m", "Initial commit")
            run_git("-C", str(seed), "push", "-u", "origin", "main")
            run_git("clone", str(remote), str(checkout))

            proposal = {
                "path": "service.py",
                "reason": "Implement the requested email validation.",
                "updated_content": (
                    "def is_valid_email(value):\n"
                    "    return '@' in value\n"
                ),
                "original_sha256": hashlib.sha256(source).hexdigest(),
            }
            api_calls: list[tuple[str, str, dict[str, object] | None]] = []

            def mock_api_request(
                method: str,
                endpoint: str,
                token: str,
                payload: dict[str, object] | None = None,
            ) -> dict[str, object]:
                self.assertEqual(token, "test-token")
                api_calls.append((method, endpoint, payload))
                if method == "GET":
                    return {"default_branch": "main"}
                return {"html_url": "https://github.com/example/service/pull/7"}

            with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
                with patch(
                    "jira_impact.github._github_api_request",
                    side_effect=mock_api_request,
                ):
                    pull_request_url = publish_approved_changes(
                        repo_path=str(checkout),
                        repository_url="https://github.com/example/service",
                        requirement="Add email validation",
                        proposals=[proposal],
                    )

            self.assertEqual(
                pull_request_url,
                "https://github.com/example/service/pull/7",
            )
            self.assertIn("return '@' in value", (checkout / "service.py").read_text())
            self.assertEqual([call[0] for call in api_calls], ["GET", "POST"])
            self.assertTrue(api_calls[1][2]["head"].startswith("copilot/jira-impact-"))
            self.assertEqual(api_calls[1][2]["base"], "main")
            remote_branches = run_git(
                "--git-dir",
                str(remote),
                "branch",
                "--list",
                "copilot/jira-impact-*",
            )
            self.assertTrue(remote_branches)


if __name__ == "__main__":
    unittest.main()
