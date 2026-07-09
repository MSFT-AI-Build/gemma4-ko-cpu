#!/usr/bin/env python3
"""
파이썬 코딩 벤치마크 - gemma4 모델 크기별 비교
한국어 지시 기반 파이썬 함수 구현 (HumanEval 스타일, pass@1)

각 문제는 함수 시그니처 + 한국어 docstring이 주어지고, 모델이 함수를 완성합니다.
생성된 코드는 격리된 서브프로세스에서 테스트와 함께 실행되어 기능 정확성을 평가합니다.

사용법:
  python3 benchmark_python.py                 # 전체 문제, 4개 모델
  python3 benchmark_python.py --small         # 앞 5문제만 (빠른 테스트)
  python3 benchmark_python.py --models e2b,e4b # 특정 모델만
"""

import json
import os
import re
import sys
import time
import subprocess
import tempfile
import datetime
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 600            # 모델 응답 타임아웃(초)
EXEC_TIMEOUT = 10        # 생성 코드 실행 타임아웃(초)
NUM_PREDICT = 1024

MODELS = [
    {"name": "e2b (2.3B)", "tag": "gemma4:e2b"},
    {"name": "e4b (4.5B)", "tag": "gemma4:latest"},
    {"name": "26b MoE",    "tag": "gemma4:26b"},
    {"name": "31b Dense",  "tag": "gemma4:31b"},
]

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "problems.json"
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
LOG_FILE = LOG_DIR / "benchmark_run.log"
PROGRESS_FILE = LOG_DIR / "benchmark_progress.md"

# ─── Logging ─────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_progress(model_name, done, total, results_so_far):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# 파이썬 코딩 벤치마크 진행 현황 ({ts})\n",
             f"**현재**: {model_name} — {done}/{total}\n",
             "\n## 완료된 결과\n",
             "| Model | pass@1 | 통과/전체 | 평균 지연 |",
             "|-------|--------|-----------|-----------|"]
    for r in results_so_far:
        lines.append(f"| {r['model']} | {r['pass_at_1']:.1f}% | "
                     f"{r['passed']}/{r['total']} | {r['avg_time']:.2f}s |")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ─── Ollama API ──────────────────────────────────────────────────────────────
def query_ollama(model_tag, prompt, temperature=0):
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": NUM_PREDICT},
        "think": False,
    }
    try:
        import urllib.request
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"


# ─── Code extraction & execution ─────────────────────────────────────────────
def extract_code(response):
    """모델 응답에서 파이썬 코드를 추출합니다."""
    # ```python ... ``` 또는 ``` ... ``` 코드 블록 우선
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if fences:
        # 가장 긴 코드 블록 사용 (함수 정의를 포함할 가능성이 큼)
        return max(fences, key=len).strip()
    return response.strip()


def build_prompt(problem):
    return (
        "다음 파이썬 함수를 완성하세요. 시그니처를 포함한 함수 전체를 파이썬 코드로만 출력하세요. "
        "설명이나 예시는 쓰지 마세요.\n\n"
        "```python\n" + problem["prompt"] + "```\n"
    )


def run_solution(code, problem):
    """생성된 코드를 테스트와 함께 격리된 서브프로세스에서 실행합니다.
    반환: (passed: bool, detail: str)
    """
    entry = problem["entry_point"]
    program = (
        code
        + "\n\n"
        + problem["test"]
        + f"\n\ncheck({entry})\n"
        + "print('__ALL_TESTS_PASSED__')\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as tf:
            tf.write(program)
            tmp_path = tf.name
    except Exception as e:
        return False, f"tempfile error: {e}"

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=EXEC_TIMEOUT,
        )
        if "__ALL_TESTS_PASSED__" in proc.stdout and proc.returncode == 0:
            return True, "ok"
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return False, err[-1] if err else f"exit={proc.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"timeout(>{EXEC_TIMEOUT}s)"
    except Exception as e:
        return False, f"exec error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Benchmark ───────────────────────────────────────────────────────────────
def evaluate_model(model, problems):
    tag, name = model["tag"], model["name"]
    log(f"{'='*60}")
    log(f"모델 평가 시작: {name} ({tag})")
    log(f"{'='*60}")

    # 웜업
    _ = query_ollama(tag, "print('hi')")

    passed = 0
    total = len(problems)
    times = []
    details = []
    by_diff = {}

    for i, prob in enumerate(problems):
        t0 = time.time()
        resp = query_ollama(tag, build_prompt(prob))
        dt = time.time() - t0
        times.append(dt)

        code = extract_code(resp)
        ok, detail = run_solution(code, prob)
        if ok:
            passed += 1
        diff = prob.get("difficulty", "unknown")
        d = by_diff.setdefault(diff, {"passed": 0, "total": 0})
        d["total"] += 1
        if ok:
            d["passed"] += 1

        details.append({
            "task_id": prob["task_id"],
            "difficulty": diff,
            "passed": ok,
            "detail": detail,
            "time": round(dt, 2),
            "response": resp,
            "extracted_code": code,
        })
        mark = "✅" if ok else "❌"
        log(f"  {mark} [{i+1}/{total}] {prob['task_id']} ({diff}) "
            f"{dt:.1f}s {'' if ok else '- ' + detail}")

    avg_time = sum(times) / len(times) if times else 0
    pass_at_1 = passed / total * 100 if total else 0
    log(f"  ▶ {name}: pass@1={pass_at_1:.1f}% ({passed}/{total}), 평균 {avg_time:.2f}s")

    return {
        "model": name,
        "tag": tag,
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "avg_time": avg_time,
        "by_difficulty": by_diff,
        "details": details,
    }


# ─── Reporting ───────────────────────────────────────────────────────────────
def generate_report(results, problems):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diffs = sorted({p.get("difficulty", "unknown") for p in problems},
                   key=lambda d: {"easy": 0, "medium": 1, "hard": 2}.get(d, 9))
    lines = [
        "# 파이썬 코딩 벤치마크 결과",
        "",
        f"**실행 시간**: {ts}",
        f"**문제 수**: {len(problems)}개 (한국어 지시 기반 파이썬 함수 구현, pass@1)",
        f"**평가 방식**: 생성 코드를 격리 서브프로세스에서 테스트 실행 (기능 정확성)",
        "",
        "## 종합 결과 (pass@1)",
        "",
        "| Model | pass@1 | 통과/전체 | 평균 지연 |",
        "|-------|:------:|:---------:|:---------:|",
    ]
    for r in results:
        lines.append(f"| {r['model']} | **{r['pass_at_1']:.1f}%** | "
                     f"{r['passed']}/{r['total']} | {r['avg_time']:.2f}s |")

    # 난이도별
    lines += ["", "## 난이도별 정확도 (%)", "",
              "| Model | " + " | ".join(diffs) + " |",
              "|-------|" + "|".join([":---:"] * len(diffs)) + "|"]
    for r in results:
        cells = []
        for d in diffs:
            bd = r["by_difficulty"].get(d)
            if bd and bd["total"]:
                cells.append(f"{bd['passed']/bd['total']*100:.0f}")
            else:
                cells.append("-")
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")

    # 문제별 통과 여부
    lines += ["", "## 문제별 통과 여부", "",
              "| Task | 난이도 | " + " | ".join(r["model"] for r in results) + " |",
              "|------|--------|" + "|".join([":---:"] * len(results)) + "|"]
    detail_map = {r["model"]: {d["task_id"]: d["passed"] for d in r["details"]}
                  for r in results}
    for p in problems:
        row = [p["task_id"], p.get("difficulty", "")]
        for r in results:
            row.append("✅" if detail_map[r["model"]].get(p["task_id"]) else "❌")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = set(sys.argv[1:])
    problems = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if "--small" in args:
        problems = problems[:5]

    models = MODELS
    for a in sys.argv[1:]:
        if a.startswith("--models"):
            wanted = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            keys = {k.strip() for k in wanted.split(",")}
            alias = {"e2b": "gemma4:e2b", "e4b": "gemma4:latest",
                     "26b": "gemma4:26b", "31b": "gemma4:31b"}
            tags = {alias.get(k, k) for k in keys}
            models = [m for m in MODELS if m["tag"] in tags]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("파이썬 코딩 벤치마크 시작")
    log(f"문제 {len(problems)}개 | 모델 {len(models)}개")
    log("=" * 60)

    results = []
    for model in models:
        r = evaluate_model(model, problems)
        results.append(r)
        update_progress(model["name"], len(results), len(models), results)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_DIR / f"results_raw_{ts}.json"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    (RESULTS_DIR / "results_latest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    report = generate_report(results, problems)
    (RESULTS_DIR / f"RESULTS_{ts}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "RESULTS_LATEST.md").write_text(report, encoding="utf-8")

    log("벤치마크 완료! 리포트 생성...")
    log(f"결과: {RESULTS_DIR / 'RESULTS_LATEST.md'}")
    log("완료!")


if __name__ == "__main__":
    main()
