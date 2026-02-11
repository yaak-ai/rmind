import subprocess  # noqa: S404


def check_git_clean() -> None:

    def has_diff(args: list[str]) -> bool:
        result = subprocess.run(args, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode == 0:
            return False
        if result.returncode == 1:
            return True
        stderr = result.stderr.strip() or f"exit code {result.returncode}"
        msg = f"Failed to check git diff state: {' '.join(args)}\n{stderr}"
        raise RuntimeError(msg)

    has_unstaged_changes = has_diff(["git", "diff", "--quiet"])
    has_staged_changes = has_diff(["git", "diff", "--cached", "--quiet"])

    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    has_untracked_files = bool(result.stdout.strip())

    result = subprocess.run(
        ["git", "rev-list", "@{upstream}..HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    has_unpushed_commits = (
        bool(result.stdout.strip()) if result.returncode == 0 else False
    )

    if has_unstaged_changes or has_staged_changes or has_untracked_files:
        status = subprocess.run(
            ["git", "status", "--short"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        msg = (
            "Uncommitted changes or untracked files detected! "
            "Commit and push before running experiment.\n\n"
            f"{status.stdout}\n"
            "Run: git add -A && git commit -m 'your message' && git push"
        )
        raise RuntimeError(msg)

    if has_unpushed_commits:
        result = subprocess.run(
            ["git", "log", "@{upstream}..HEAD", "--oneline"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        msg = (
            "Unpushed commits detected! Push before running experiment.\n\n"
            f"{result.stdout}\n"
            "Run: git push"
        )
        raise RuntimeError(msg)


def main() -> None:
    try:
        check_git_clean()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
