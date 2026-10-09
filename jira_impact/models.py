from typing import TypedDict


class RequirementState(TypedDict, total=False):
    requirement_input: str
    repo_path: str
    jira_source: str
    model_name: str
    github_url: str
    requirement_summary: str
    functional_requirements: list[str]
    search_terms: list[str]
    evidence: list[dict[str, object]]
    editable_paths: list[str]
    proposed_changes: list[dict[str, object]]
    validation_errors: list[str]
    scan_warnings: list[str]
    approved: bool
    pull_request_url: str
