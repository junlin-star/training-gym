"""Download baseline validation results from `main` for perf comparison.

Runs on a GitHub Actions runner before the summarize step: for each
validatable model, fetch the most recent `validate-result-<model>` artifact
produced by a workflow run on `main` and unzip it into the baseline
directory consumed by `validate_model_configs.py summarize --baseline-dir`.

Missing or failing artifacts are skipped with a warning so a stale baseline
never blocks the comment; the baseline directory is created regardless.
"""

import argparse
import io
import os
import zipfile
from pathlib import Path

import requests
from github import Auth, Github
from github.Artifact import Artifact
from github.Repository import Repository

from validate_model_configs import available_model_names

ARTIFACT_PREFIX = "validate-result-"
BASELINE_BRANCH = "main"


def artifact_name_for_model(model_name: str) -> str:
    """Mirror the artifact naming in validate-models.yml ('/' -> '-')."""
    return ARTIFACT_PREFIX + model_name.replace("/", "-")


def find_baseline_artifact(repo: Repository, artifact_name: str) -> Artifact | None:
    """Latest non-expired artifact with this name from a run on main.

    ``get_artifacts`` returns newest-first and paginates transparently, so the
    first match is the latest baseline.
    """
    for artifact in repo.get_artifacts(name=artifact_name):
        return artifact
        if artifact.expired:
            continue
        run = artifact.workflow_run
        if run is not None and run.head_branch == BASELINE_BRANCH:
            return artifact
    return None


def download_and_extract(artifact: Artifact, token: str, dest_dir: Path) -> None:
    """Unzip the artifact's files into dest_dir.

    ``requests`` follows the API's redirect to blob storage and drops the
    Authorization header across hosts, which the signed URL requires.
    """
    response = requests.get(
        artifact.archive_download_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(dest_dir)


def download_baselines(baseline_dir: Path) -> None:
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPOSITORY"]

    baseline_dir.mkdir(parents=True, exist_ok=True)
    repo = Github(auth=Auth.Token(token)).get_repo(repo_name)

    downloaded = 0
    models = available_model_names()
    for model_name in models:
        artifact_name = artifact_name_for_model(model_name)
        try:
            artifact = find_baseline_artifact(repo, artifact_name)
            if artifact is None:
                print(
                    f"warning: no {artifact_name!r} artifact found from a run "
                    f"on {BASELINE_BRANCH!r}; skipping baseline for {model_name}"
                )
                continue
            download_and_extract(artifact, token, baseline_dir)
        except Exception as exc:
            print(
                f"warning: failed to download baseline {artifact_name!r} "
                f"for {model_name}: {exc}"
            )
            continue
        downloaded += 1
        print(f"downloaded baseline {artifact_name!r} (artifact id {artifact.id})")

    print(f"downloaded {downloaded}/{len(models)} baselines into {baseline_dir}")


def __main__():
    parser = argparse.ArgumentParser(
        description="Download baseline validation artifacts from main for each model."
    )
    parser.add_argument(
        "-d",
        "--baseline-dir",
        default="baseline",
        help="Directory to unzip baseline result JSON files into. Defaults to 'baseline'.",
    )
    args = parser.parse_args()
    download_baselines(Path(args.baseline_dir))


if __name__ == "__main__":
    __main__()
