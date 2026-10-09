# Jira repository impact workflow

This command-line workflow turns a Jira feature or plain-text capability into an
evidence-backed repository impact report. It identifies candidate files and
classes/methods, validates citations against scanned source, and pauses for a
human to review the proposed changes. With a GitHub URL, approval creates a
branch, commits the reviewed changes, pushes the branch, and opens a pull
request. Rejection makes no repository changes.

## Agents

1. **Requirement agent** summarizes the request and derives functional
   requirements and search terms.
2. **Code search agent** scans supported source files in the selected
   repository and records matching lines with their nearest class or method.
3. **Impact analysis agent** maps evidence to candidate changes and explains
   the reason. For GitHub pull requests, it also generates complete updated
   file contents and a diff for review.
4. **Validation agent** rejects proposals whose file, symbol, or evidence
   citations do not match the local scan, and prepares a diff only for a
   validated existing file.
5. **Human approval** pauses the LangGraph run until the user approves or
   rejects the validated report in the terminal.
6. **GitHub publisher** runs only after approval. It applies the reviewed
   content in a temporary clone, verifies the files have not changed since
   review, creates a branch, pushes it, and opens a pull request.

If `OPENAI_API_KEY` is set, the first and third agents use the configured
OpenAI-compatible chat model (`OPENAI_MODEL`, default `gpt-4o-mini`). Without
that key, those agents use local, deterministic behavior. Jira fetching is
independent of model configuration.

## Setup

Use Python 3.10 or newer:

```powershell
cd "multiagent system"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To fetch Jira issues, configure credentials in the current terminal session:

```powershell
$env:JIRA_BASE_URL = "https://your-company.atlassian.net"
$env:JIRA_EMAIL = "you@example.com"
$env:JIRA_API_TOKEN = "your-api-token"
```

For model-assisted analysis, set `OPENAI_API_KEY` and optionally `OPENAI_MODEL`.
The API key is read from the environment and is not written to disk by this
application.

To create GitHub pull requests, also set `GITHUB_TOKEN` (or `GH_TOKEN`) to a
token with repository **Contents: read/write** and **Pull requests: read/write**
permissions. Git is required. The token is supplied to Git through the process
environment and is not written into the clone's Git remote configuration.

## Run

From this folder, pass a local directory or a GitHub HTTPS URL, together with a
Jira issue key or plain-text capability:

```powershell
python .\main.py --repo "C:\path\to\repository" --requirement "PROJ-123"
python .\main.py --repo "C:\path\to\repository" --requirement "Add email verification during signup"
python .\main.py --github-url "https://github.com/owner/repository" --requirement "PROJ-123"
python .\main.py --github-url "https://github.com/owner/repository" --requirement "Add email verification during signup"
```

Omit `--requirement` to enter the request interactively. Jira issue keys are
looked up through Jira Cloud's REST API and require `JIRA_BASE_URL`,
`JIRA_EMAIL`, and `JIRA_API_TOKEN`. All other input is treated as plain text.
When `--repo` is used, the workflow only produces an impact report. When
`--github-url` is used, it requires both `OPENAI_API_KEY` and a GitHub token,
shows the complete proposed diff, and asks before pushing or opening a PR. The
GitHub checkout is temporary; changes are delivered through the pull request.

The scanner supports common Python, Java, JavaScript/TypeScript, C/C++, C#,
Go, Rust, Ruby, PHP, Kotlin, Swift, Scala, and SQL source files. It skips
common generated/dependency directories, files larger than 512 KB, and reports
when the scan reaches its 8,000-file limit.

## Tests

From this folder:

```powershell
python -m unittest discover -s tests -v
```
