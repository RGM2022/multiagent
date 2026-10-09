import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.types import Command

from jira_impact.repository import scan_repository
from jira_impact.workflow import (
    ImpactAnalysis,
    ProposedChange,
    RequirementBreakdown,
    build_workflow,
)


class WorkflowTests(unittest.TestCase):
    def test_typescript_scan_finds_class_method_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "signup.ts").write_text(
                "export class SignupService {\n"
                "  verifyEmail(email: string): boolean {\n"
                "    return email.includes('@');\n"
                "  }\n"
                "}\n",
                encoding="utf-8",
            )

            evidence, _ = scan_repository(str(repository), ["email"])

        self.assertTrue(evidence)
        self.assertEqual(evidence[0]["symbol"], "SignupService.verifyEmail")

    def test_workflow_pauses_for_review_and_resumes_with_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            source_file = repository / "accounts.py"
            source_file.write_text(
                "class AccountService:\n"
                "    def create_account(self, email: str) -> None:\n"
                "        validate_email(email)\n",
                encoding="utf-8",
            )
            workflow = build_workflow()
            config = {"configurable": {"thread_id": "approval-test"}}

            paused_state = workflow.invoke(
                {
                    "requirement_input": "Add account email validation",
                    "repo_path": str(repository),
                },
                config=config,
            )

            self.assertTrue(paused_state["evidence"])
            self.assertEqual(
                paused_state["proposed_changes"][0]["path"], "accounts.py"
            )
            self.assertFalse(paused_state["validation_errors"])
            self.assertTrue(paused_state.get("__interrupt__"))

            approved_state = workflow.invoke(
                Command(resume={"approved": True}),
                config=config,
            )
            self.assertTrue(approved_state["approved"])

    def test_invalid_evidence_is_not_presented_as_a_change(self) -> None:
        from jira_impact.workflow import validation_agent

        state = {
            "evidence": [
                {
                    "id": "E001",
                    "path": "accounts.py",
                    "line": 2,
                    "symbol": "AccountService.create_account",
                    "snippet": "def create_account(self):",
                }
            ],
            "repo_path": "",
            "proposed_changes": [
                {
                    "path": "other.py",
                    "symbol": "AccountService.create_account",
                    "change": "Change the implementation",
                    "reason": "Requirement match",
                    "evidence_ids": ["E001"],
                }
            ],
        }

        result = validation_agent(state)

        self.assertEqual(result["proposed_changes"], [])
        self.assertTrue(result["validation_errors"])

    def test_workflow_rejection_resumes_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "accounts.py").write_text(
                "def validate_email(email: str) -> bool:\n"
                "    return '@' in email\n",
                encoding="utf-8",
            )
            workflow = build_workflow()
            config = {"configurable": {"thread_id": "rejection-test"}}
            workflow.invoke(
                {
                    "requirement_input": "Validate account email",
                    "repo_path": str(repository),
                },
                config=config,
            )

            rejected_state = workflow.invoke(
                Command(resume={"approved": False}),
                config=config,
            )

            self.assertFalse(rejected_state["approved"])

    def test_github_publisher_runs_only_after_approved_code_diff(self) -> None:
        class FakeStructuredModel:
            def __init__(self, schema: type) -> None:
                self.schema = schema

            def invoke(self, messages: list[tuple[str, str]]) -> object:
                assert messages
                if self.schema is RequirementBreakdown:
                    return RequirementBreakdown(
                        summary="Validate account email",
                        functional_requirements=["Validate account email"],
                        search_terms=["email"],
                    )
                return ImpactAnalysis(
                    proposed_changes=[
                        ProposedChange(
                            path="accounts.py",
                            symbol="validate_email",
                            change="Validate the submitted email address.",
                            reason="This function handles account email validation.",
                            evidence_ids=["E001"],
                            updated_content=(
                                "def validate_email(email):\n"
                                "    return '@' in email\n"
                            ),
                        )
                    ]
                )

        class FakeModel:
            def with_structured_output(self, schema: type) -> FakeStructuredModel:
                return FakeStructuredModel(schema)

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "accounts.py").write_text(
                "def validate_email(email):\n"
                "    return False\n",
                encoding="utf-8",
            )
            with patch(
                "jira_impact.workflow._configured_model",
                return_value=FakeModel(),
            ):
                with patch(
                    "jira_impact.workflow.publish_approved_changes",
                    return_value="https://github.com/example/service/pull/8",
                ) as publish:
                    rejected_workflow = build_workflow()
                    rejected_config = {
                        "configurable": {"thread_id": "github-rejected"}
                    }
                    rejected_state = rejected_workflow.invoke(
                        {
                            "requirement_input": "Validate account email",
                            "repo_path": str(repository),
                            "github_url": "https://github.com/example/service",
                        },
                        config=rejected_config,
                    )
                    self.assertIn(
                        "+    return '@' in email",
                        rejected_state["proposed_changes"][0]["patch"],
                    )
                    self.assertTrue(rejected_state.get("__interrupt__"))
                    publish.assert_not_called()
                    rejected_workflow.invoke(
                        Command(resume={"approved": False}),
                        config=rejected_config,
                    )
                    publish.assert_not_called()

                    approved_workflow = build_workflow()
                    approved_config = {
                        "configurable": {"thread_id": "github-approved"}
                    }
                    approved_workflow.invoke(
                        {
                            "requirement_input": "Validate account email",
                            "repo_path": str(repository),
                            "github_url": "https://github.com/example/service",
                        },
                        config=approved_config,
                    )
                    approved_state = approved_workflow.invoke(
                        Command(resume={"approved": True}),
                        config=approved_config,
                    )

        self.assertEqual(
            approved_state["pull_request_url"],
            "https://github.com/example/service/pull/8",
        )
        publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
