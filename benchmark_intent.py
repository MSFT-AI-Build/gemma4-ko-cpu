#!/usr/bin/env python3
"""
한글 Intent 분석 성능 벤치마크 - gemma4 모델 크기별 비교
고객 서비스 도메인 (주문/배송/환불/문의)

사용법:
  python3 benchmark_intent.py                  # 3000개, 공정+최대성능 둘 다
  python3 benchmark_intent.py --small          # 30개, 빠른 테스트
  python3 benchmark_intent.py --fair-only      # 공정 비교만
  python3 benchmark_intent.py --max-only       # 최대 성능만
"""

import json
import time
import sys
import os
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tabulate import tabulate
from test_cases_3000 import TEST_CASES as TEST_CASES_3000

OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
PARALLEL_WORKERS = 4

MODELS = [
    "gemma4:e2b",     # Edge 2.3B
    "gemma4:latest",  # Edge 4.5B (e4b)
    "gemma4:26b",     # MoE 26B (3.8B active)
    "gemma4:31b",     # Dense 31B
]

MODEL_LABELS = {
    "gemma4:e2b": "e2b (2.3B)",
    "gemma4:latest": "e4b (4.5B)",
    "gemma4:26b": "26b MoE",
    "gemma4:31b": "31b Dense",
}

SYSTEM_PROMPT = """당신은 고객 서비스 챗봇의 Intent 분류기입니다.
사용자의 메시지를 읽고, 아래 Intent 중 하나로 분류하세요.

가능한 Intent 목록:
- 주문조회: 주문 상태, 주문 내역, 주문 번호 확인
- 배송조회: 배송 상태, 배송 위치, 도착 예정일 확인
- 환불요청: 환불, 취소, 반품 요청
- 교환요청: 상품 교환 요청
- 상품문의: 상품 정보, 재고, 가격, 사이즈 문의
- 결제문의: 결제 수단, 결제 오류, 할부, 카드 문의
- 불만접수: 불만, 클레임, 서비스 불만족
- 회원정보: 회원가입, 비밀번호 변경, 개인정보 수정
- 쿠폰/적립금: 쿠폰 사용, 적립금 조회, 할인 문의
- 기타: 위 카테고리에 해당하지 않는 경우

반드시 Intent 이름만 출력하세요. 다른 설명은 하지 마세요."""

TEST_CASES_SMALL = [
    ("제가 지난주에 주문한 거 어떻게 됐나요?", "주문조회"),
    ("주문번호 2024-1234 확인 좀 해주세요", "주문조회"),
    ("내가 산 거 목록 좀 보여줘", "주문조회"),
    ("택배 언제 오나요?", "배송조회"),
    ("배송 추적 좀 해주세요", "배송조회"),
    ("오늘 도착한다고 했는데 아직 안 왔어요", "배송조회"),
    ("이거 환불하고 싶습니다", "환불요청"),
    ("주문 취소 어떻게 해요?", "환불요청"),
    ("반품 접수 하려고요", "환불요청"),
    ("사이즈가 안 맞아서 교환하고 싶어요", "교환요청"),
    ("색상 다른 걸로 바꿀 수 있나요?", "교환요청"),
    ("불량이라 교환 요청합니다", "교환요청"),
    ("이 제품 재고 있어요?", "상품문의"),
    ("XL 사이즈 있나요?", "상품문의"),
    ("이 상품 언제 입고되나요?", "상품문의"),
    ("카드 결제가 안 돼요", "결제문의"),
    ("무이자 할부 되나요?", "결제문의"),
    ("결제 수단 변경하고 싶어요", "결제문의"),
    ("서비스가 너무 불친절해요", "불만접수"),
    ("도대체 왜 이렇게 늦는 거예요? 화가 납니다", "불만접수"),
    ("엉뚱한 상품이 왔는데 어떻게 된 거예요?", "불만접수"),
    ("비밀번호를 잊어버렸어요", "회원정보"),
    ("전화번호 변경하려고요", "회원정보"),
    ("탈퇴하고 싶습니다", "회원정보"),
    ("쿠폰 어떻게 쓰나요?", "쿠폰/적립금"),
    ("적립금 얼마 있는지 확인해주세요", "쿠폰/적립금"),
    ("신규 가입 할인 쿠폰 있나요?", "쿠폰/적립금"),
    ("영업시간이 어떻게 되나요?", "기타"),
    ("매장 위치 알려주세요", "기타"),
    ("고마워요 잘 해결됐어요", "기타"),
]

VALID_INTENTS = [
    "주문조회", "배송조회", "환불요청", "교환요청", "상품문의",
    "결제문의", "불만접수", "회원정보", "쿠폰/적립금", "기타",
]

BASE_DIR = "/home/azureuser/workspace/h01"


# ─────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────

def query_model(model: str, user_message: str, think: bool = False,
                num_predict: int = 50) -> dict:
    """모델에 쿼리하고 상세 결과를 반환"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_CHAT_API, json=payload, timeout=600)
        elapsed = time.time() - start
        data = resp.json()
        msg_data = data.get("message", {})
        answer = msg_data.get("content", "").strip()
        thinking = msg_data.get("thinking", "")
        return {
            "answer": answer,
            "thinking": thinking,
            "elapsed": elapsed,
            "done_reason": data.get("done_reason", ""),
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "thinking": "",
            "elapsed": time.time() - start,
            "done_reason": "error",
        }


def normalize_intent(raw: str) -> str:
    """응답에서 Intent 이름을 추출"""
    raw = raw.strip().strip('"').strip("'").strip()
    for intent in VALID_INTENTS:
        if intent in raw:
            return intent
    return raw[:20]


def warmup_model(model: str):
    """모델 로딩을 위한 웜업 호출"""
    query_model(model, "테스트", think=False, num_predict=5)


def run_single_test(args):
    """병렬 실행용 단일 테스트 (index, msg, expected, model, think, num_predict)"""
    idx, msg, expected, model, think, num_predict = args
    result = query_model(model, msg, think=think, num_predict=num_predict)
    predicted = normalize_intent(result["answer"])
    return {
        "index": idx,
        "input": msg,
        "expected": expected,
        "predicted": predicted,
        "raw": result["answer"],
        "thinking": result["thinking"],
        "correct": predicted == expected,
        "time": result["elapsed"],
        "done_reason": result["done_reason"],
    }


def run_model_benchmark(model: str, test_cases: list, think: bool,
                        num_predict: int, label: str) -> dict:
    """단일 모델 벤치마크 실행 (병렬)"""
    print(f"\n🔄 [{MODEL_LABELS[model]}] 테스트 시작... (think={'ON' if think else 'OFF'}, "
          f"workers={PARALLEL_WORKERS})")

    # 웜업
    warmup_model(model)

    tasks = [
        (i, msg, expected, model, think, num_predict)
        for i, (msg, expected) in enumerate(test_cases)
    ]

    details = []
    completed = 0
    correct = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(run_single_test, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            details.append(r)
            completed += 1
            if r["correct"]:
                correct += 1

            if completed % 100 == 0 or completed == len(test_cases):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(test_cases) - completed) / rate if rate > 0 else 0
                print(f"  📊 {completed}/{len(test_cases)} "
                      f"({correct}/{completed} correct, {correct/completed*100:.1f}%) "
                      f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    total_time = time.time() - start_time
    details.sort(key=lambda x: x["index"])

    accuracy = correct / len(test_cases) * 100
    avg_time = sum(d["time"] for d in details) / len(details)

    print(f"  ✅ [{MODEL_LABELS[model]}] 완료: {accuracy:.1f}% ({correct}/{len(test_cases)}), "
          f"평균 {avg_time:.2f}s/건, 총 {total_time:.0f}s (wall)")

    return {
        "model": model,
        "label": MODEL_LABELS[model],
        "think": think,
        "num_predict": num_predict,
        "details": details,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "avg_time": avg_time,
        "total_time": total_time,
        "wall_time": total_time,
    }


# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

def save_raw_log(all_results: dict, path: str):
    """모든 raw 응답을 포함한 JSON 로그 저장"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"💾 Raw 로그 저장: {path}")


def generate_report(bench_name: str, bench_label: str, results: list,
                    test_cases: list) -> str:
    """단일 벤치마크에 대한 마크다운 리포트 생성"""
    lines = []
    lines.append(f"## {bench_label}")
    lines.append("")

    think_mode = "ON (26b/31b)" if bench_name == "max" else "OFF (전체)"
    np_val = "1000 (26b/31b) / 50 (e2b/e4b)" if bench_name == "max" else "50"
    lines.append(f"- **Thinking**: {think_mode}")
    lines.append(f"- **num_predict**: {np_val}")
    lines.append(f"- **병렬 workers**: {PARALLEL_WORKERS}")
    lines.append(f"- **테스트 케이스**: {len(test_cases)}개")
    lines.append("")

    # 종합 비교표
    lines.append("### 종합 결과")
    lines.append("")
    lines.append("| 모델 | 정확도 | 정답수 | 평균응답시간 | 총소요시간(wall) |")
    lines.append("|------|--------|--------|-------------|-----------------|")
    for r in results:
        lines.append(f"| {r['label']} | **{r['accuracy']:.1f}%** | "
                     f"{r['correct']}/{r['total']} | {r['avg_time']:.2f}s | "
                     f"{r['wall_time']:.0f}s |")
    lines.append("")

    # Intent별 정확도
    intents = list(dict.fromkeys(exp for _, exp in test_cases))
    lines.append("### Intent별 정확도")
    lines.append("")
    header = "| Intent | " + " | ".join(r["label"] for r in results) + " |"
    sep = "|--------|" + "|".join("------" for _ in results) + "|"
    lines.append(header)
    lines.append(sep)
    for intent in intents:
        row = f"| {intent} |"
        for r in results:
            relevant = [d for d in r["details"] if d["expected"] == intent]
            c = sum(1 for d in relevant if d["correct"])
            total = len(relevant)
            pct = c / total * 100 if total else 0
            row += f" {c}/{total} ({pct:.0f}%) |"
        lines.append(row)
    lines.append("")

    # 오답 요약 (모델별 상위 혼동 패턴)
    lines.append("### 오답 패턴 분석")
    lines.append("")
    for r in results:
        errors = [d for d in r["details"] if not d["correct"]]
        lines.append(f"**{r['label']}** — 오답 {len(errors)}건 "
                     f"({len(errors)/r['total']*100:.1f}%)")
        lines.append("")
        if errors:
            confusion = Counter((d["expected"], d["predicted"]) for d in errors)
            lines.append("| 정답 → 예측 | 건수 | 예시 입력 |")
            lines.append("|------------|------|----------|")
            for (exp, pred), cnt in confusion.most_common(10):
                example = next(d["input"] for d in errors
                             if d["expected"] == exp and d["predicted"] == pred)
                lines.append(f"| {exp} → {pred} | {cnt} | {example[:40]} |")
            lines.append("")
        else:
            lines.append("오답 없음! 🎉\n")

    return "\n".join(lines)


def generate_full_report(all_results: dict, test_cases: list) -> str:
    """전체 마크다운 리포트 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# Gemma4 한글 Intent 분류 벤치마크 결과")
    lines.append("")
    lines.append(f"**실행 시각**: {now}")
    lines.append(f"**테스트 케이스**: {len(test_cases)}개 (10개 Intent × "
                 f"{len(test_cases)//10}문장)")
    lines.append(f"**환경**: Azure VM, 31GB RAM, CPU only, Ollama v0.20.4")
    lines.append(f"**병렬 workers**: {PARALLEL_WORKERS}")
    lines.append("")

    lines.append("## 모델 정보")
    lines.append("")
    lines.append("| 태그 | 파라미터 | 크기 | Context |")
    lines.append("|------|---------|------|---------|")
    lines.append("| gemma4:e2b | 2.3B | 7.2 GB | 128K |")
    lines.append("| gemma4:latest (e4b) | 4.5B | 9.6 GB | 128K |")
    lines.append("| gemma4:26b | 26B MoE (3.8B active) | 17 GB | 256K |")
    lines.append("| gemma4:31b | 31B Dense | 19 GB | 256K |")
    lines.append("")

    for bench_name, bench_data in all_results.items():
        label = bench_data["label"]
        results = bench_data["results"]
        lines.append(generate_report(bench_name, label, results, test_cases))

    # 두 벤치마크 간 비교 (둘 다 존재할 때)
    if "fair" in all_results and "max" in all_results:
        lines.append("## 공정 vs 최대성능 비교")
        lines.append("")
        lines.append("| 모델 | 공정 (think OFF) | 최대성능 (think ON) | 차이 |")
        lines.append("|------|-----------------|-------------------|------|")
        fair_map = {r["model"]: r for r in all_results["fair"]["results"]}
        max_map = {r["model"]: r for r in all_results["max"]["results"]}
        for model in MODELS:
            f = fair_map[model]
            m = max_map[model]
            diff = m["accuracy"] - f["accuracy"]
            sign = "+" if diff >= 0 else ""
            lines.append(f"| {MODEL_LABELS[model]} | {f['accuracy']:.1f}% | "
                         f"{m['accuracy']:.1f}% | {sign}{diff:.1f}%p |")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = set(sys.argv[1:])
    test_cases = TEST_CASES_SMALL if "--small" in args else TEST_CASES_3000
    run_fair = "--max-only" not in args
    run_max = "--fair-only" not in args

    size_label = f"{len(test_cases)}개"
    print("=" * 70)
    print("  한글 Intent 분석 벤치마크 - gemma4 모델 크기별 비교")
    print(f"  테스트: {size_label} | "
          f"모드: {'공정+최대성능' if run_fair and run_max else '공정' if run_fair else '최대성능'}")
    print(f"  병렬: {PARALLEL_WORKERS} workers")
    print("=" * 70)

    all_results = {}

    # ── 벤치마크 1: 공정 비교 (think OFF 통일) ──
    if run_fair:
        print("\n" + "=" * 70)
        print("  🏁 벤치마크 1: 공정 비교 (think OFF, num_predict=50)")
        print("=" * 70)
        fair_results = []
        for model in MODELS:
            r = run_model_benchmark(model, test_cases, think=False,
                                    num_predict=50, label="fair")
            fair_results.append(r)
        all_results["fair"] = {
            "label": "벤치마크 1: 공정 비교 (think OFF 통일)",
            "results": fair_results,
        }

    # ── 벤치마크 2: 최대 성능 (26b/31b think ON) ──
    if run_max:
        print("\n" + "=" * 70)
        print("  🏁 벤치마크 2: 최대 성능 (26b/31b think ON, num_predict=1000)")
        print("=" * 70)
        max_results = []
        for model in MODELS:
            use_think = model in ("gemma4:26b", "gemma4:31b")
            np = 1000 if use_think else 50
            r = run_model_benchmark(model, test_cases, think=use_think,
                                    num_predict=np, label="max")
            max_results.append(r)
        all_results["max"] = {
            "label": "벤치마크 2: 최대 성능 (26b/31b think ON)",
            "results": max_results,
        }

    # ── 결과 저장 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw JSON 로그
    log_path = os.path.join(BASE_DIR, f"benchmark_raw_{timestamp}.json")
    save_raw_log(all_results, log_path)

    # 마크다운 리포트
    report = generate_full_report(all_results, test_cases)
    md_path = os.path.join(BASE_DIR, f"BENCHMARK_RESULTS_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📝 리포트 저장: {md_path}")

    # latest 심볼릭 (편의용)
    latest_md = os.path.join(BASE_DIR, "BENCHMARK_RESULTS.md")
    latest_json = os.path.join(BASE_DIR, "benchmark_results.json")
    for src, dst in [(md_path, latest_md), (log_path, latest_json)]:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    print(f"\n🔗 최신 결과: {latest_md}")
    print("✅ 벤치마크 완료!")


if __name__ == "__main__":
    main()
