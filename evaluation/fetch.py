"""Download original benchmark files (not Hugging Face) into benchmarks/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"

MMLU_TAR = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
ARC_ZIP = "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip"
GSM8K_REPO = "https://github.com/openai/grade-school-math.git"
HELLASWAG_REPO = "https://github.com/rowanz/hellaswag.git"

# Direct originals if git is unavailable (same files as the repos above).
GSM8K_RAW = {
    "test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
}
HELLASWAG_RAW = {
    "hellaswag_val.jsonl": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "hellaswag_test.jsonl": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def _download(url: str, dest: Path, *, retries: int = 6) -> None:
    """Download with resume. Prefer curl; fall back to urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}")
    curl = shutil.which("curl")
    if curl:
        proc = subprocess.run(
            [
                curl,
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "-C",
                "-",
                "-o",
                str(dest),
                url,
            ]
        )
        if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
            return
        print("curl failed, trying urllib…", file=sys.stderr)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except (urllib.error.ContentTooShortError, urllib.error.URLError, OSError) as exc:
            last = exc
            print(f"download retry {attempt + 1}/{retries}: {exc}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 30))
    raise last or RuntimeError(f"failed to download {url}")


def _git_clone(url: str, dest: Path) -> bool:
    if dest.is_dir() and any(dest.iterdir()):
        print(f"already cloned: {dest}")
        return True
    git = shutil.which("git")
    if not git:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"git clone --depth 1 {url}")
    proc = subprocess.run(
        [git, "clone", "--depth", "1", url, str(dest)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr[-500:], file=sys.stderr)
        return False
    return True


def fetch_mmlu(root: Path = BENCH) -> Path:
    """Hendrycks original CSVs: benchmarks/mmlu/{dev,val,test}/*_{split}.csv"""
    dest = root / "mmlu"
    test_dir = dest / "test"
    if test_dir.is_dir() and any(test_dir.glob("*_test.csv")):
        print(f"MMLU already present: {test_dir}")
        return dest
    tar_path = dest / "_data.tar"
    _download(MMLU_TAR, tar_path)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(dest / "_extract", filter="data")
        except TypeError:
            tf.extractall(dest / "_extract")
    extracted = dest / "_extract" / "data"
    if not extracted.is_dir():
        # some tars unpack as data/ at dest/_extract/data or dest/_extract/*/data
        matches = list((dest / "_extract").rglob("test"))
        extracted = matches[0].parent if matches else extracted
    for split in ("dev", "val", "test"):
        src = extracted / split
        if src.is_dir():
            shutil.copytree(src, dest / split, dirs_exist_ok=True)
    tar_path.unlink(missing_ok=True)
    shutil.rmtree(dest / "_extract", ignore_errors=True)
    print(f"MMLU test CSVs → {test_dir}")
    return dest


def fetch_gsm8k(root: Path = BENCH) -> Path:
    """OpenAI original JSONL: benchmarks/gsm8k/test.jsonl (and train.jsonl for shots)."""
    dest = root / "gsm8k"
    test = dest / "test.jsonl"
    if test.is_file():
        print(f"GSM8K already present: {test}")
        return dest
    clone = dest / "_repo"
    if _git_clone(GSM8K_REPO, clone):
        data = clone / "grade_school_math" / "data"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / "test.jsonl", dest / "test.jsonl")
        shutil.copy2(data / "train.jsonl", dest / "train.jsonl")
        shutil.rmtree(clone, ignore_errors=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for name, url in GSM8K_RAW.items():
            _download(url, dest / name)
    print(f"GSM8K test JSONL → {test}")
    return dest


def fetch_hellaswag(root: Path = BENCH) -> Path:
    """Original JSONL. Test labels are hidden; scoring uses val (same as published local evals)."""
    dest = root / "hellaswag"
    val = dest / "hellaswag_val.jsonl"
    if val.is_file():
        print(f"HellaSwag already present: {val}")
        return dest
    clone = dest / "_repo"
    if _git_clone(HELLASWAG_REPO, clone):
        data = clone / "data"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("hellaswag_train.jsonl", "hellaswag_val.jsonl", "hellaswag_test.jsonl"):
            src = data / name
            if src.is_file():
                shutil.copy2(src, dest / name)
        shutil.rmtree(clone, ignore_errors=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for name, url in HELLASWAG_RAW.items():
            _download(url, dest / name)
    labeled = dest / "hellaswag_val.jsonl"
    if labeled.is_file():
        shutil.copy2(labeled, dest / "val.jsonl")
    orig_test = dest / "hellaswag_test.jsonl"
    if orig_test.is_file():
        shutil.copy2(orig_test, dest / "test.jsonl")
    print(f"HellaSwag JSONL → {dest}  (score val; original test has no public labels)")
    return dest


def fetch_arc(root: Path = BENCH) -> Path:
    """AI2 original ARC-Challenge JSONL (not the 14M-sentence corpus)."""
    dest = root / "arc" / "challenge"
    test = dest / "ARC-Challenge-Test.jsonl"
    if test.is_file():
        print(f"ARC-Challenge already present: {test}")
        return dest
    zip_path = root / "arc" / "_ARC-V1-Feb2018.zip"
    _download(ARC_ZIP, zip_path)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        keep = (
            "ARC-Challenge-Test.jsonl",
            "ARC-Challenge-Dev.jsonl",
            "ARC-Challenge-Train.jsonl",
        )
        for info in zf.infolist():
            name = Path(info.filename).name
            if name in keep:
                with zf.open(info) as src, open(dest / name, "wb") as out:
                    shutil.copyfileobj(src, out)
    zip_path.unlink(missing_ok=True)
    print(f"ARC-Challenge JSONL → {dest}")
    return dest


def link_models(root: Path = ROOT) -> None:
    """Point models/ at the local GGUFs already produced under benchmark/models/."""
    src_dir = root / "benchmark" / "models"
    dest = root / "models"
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "original.gguf": "functiongemma-270m-bf16.gguf",
        "q4_k_m.gguf": "functiongemma-270m-q4_k_m.gguf",
        "q5_k_m.gguf": "functiongemma-270m-q5_k_m.gguf",
        "q6_k.gguf": "functiongemma-270m-q6_k.gguf",
        "opendynamic.gguf": "functiongemma-270m-odg.gguf",
    }
    for alias, name in mapping.items():
        src = src_dir / name
        dst = dest / alias
        if not src.is_file():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
        print(f"model {alias} → {src.name}")


def fetch_all(tasks: list[str] | None = None) -> None:
    wanted = tasks or ["mmlu", "gsm8k", "hellaswag", "arc"]
    BENCH.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    runners = {
        "mmlu": fetch_mmlu,
        "gsm8k": fetch_gsm8k,
        "hellaswag": fetch_hellaswag,
        "arc": fetch_arc,
    }
    for name in wanted:
        fn = runners.get(name)
        if fn is None:
            errors.append(f"unknown task {name}")
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            print(f"ERROR fetching {name}: {exc}", file=sys.stderr)
    link_models()
    if errors:
        raise RuntimeError("fetch failed: " + " | ".join(errors))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch original (non-HF) benchmark files.")
    p.add_argument(
        "--tasks",
        default="mmlu,gsm8k,hellaswag,arc",
        help="Comma-separated: mmlu,gsm8k,hellaswag,arc",
    )
    args = p.parse_args(argv)
    fetch_all([t.strip() for t in args.tasks.split(",") if t.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
