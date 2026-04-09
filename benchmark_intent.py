#!/usr/bin/env python3
"""
한글 Intent 분석 성능 벤치마크 - gemma4 모델 크기별 비교
고객 서비스 도메인 (주문/배송/환불/문의)
"""

import json
import time
import requests
from tabulate import tabulate
from test_cases_3000 import TEST_CASES as TEST_CASES_3000

OLLAMA_API = "http://localhost:11434/api/generate"
MODELS = [
    "gemma4:e2b",     # Edge 2.3B
    "gemma4:latest",  # Edge 4.5B (e4b)
    "gemma4:26b",     # MoE 26B (3.8B active)
    "gemma4:31b",     # Dense 31B
]

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

# (입력 문장, 정답 Intent) - 기본 30개 + 확장 3000개
TEST_CASES_SMALL = [
    # 주문조회
    ("제가 지난주에 주문한 거 어떻게 됐나요?", "주문조회"),
    ("주문번호 2024-1234 확인 좀 해주세요", "주문조회"),
    ("내가 산 거 목록 좀 보여줘", "주문조회"),

    # 배송조회
    ("택배 언제 오나요?", "배송조회"),
    ("배송 추적 좀 해주세요", "배송조회"),
    ("오늘 도착한다고 했는데 아직 안 왔어요", "배송조회"),

    # 환불요청
    ("이거 환불하고 싶습니다", "환불요청"),
    ("주문 취소 어떻게 해요?", "환불요청"),
    ("반품 접수 하려고요", "환불요청"),

    # 교환요청
    ("사이즈가 안 맞아서 교환하고 싶어요", "교환요청"),
    ("색상 다른 걸로 바꿀 수 있나요?", "교환요청"),
    ("불량이라 교환 요청합니다", "교환요청"),

    # 상품문의
    ("이 제품 재고 있어요?", "상품문의"),
    ("XL 사이즈 있나요?", "상품문의"),
    ("이 상품 언제 입고되나요?", "상품문의"),

    # 결제문의
    ("카드 결제가 안 돼요", "결제문의"),
    ("무이자 할부 되나요?", "결제문의"),
    ("결제 수단 변경하고 싶어요", "결제문의"),

    # 불만접수
    ("서비스가 너무 불친절해요", "불만접수"),
    ("도대체 왜 이렇게 늦는 거예요? 화가 납니다", "불만접수"),
    ("엉뚱한 상품이 왔는데 어떻게 된 거예요?", "불만접수"),

    # 회원정보
    ("비밀번호를 잊어버렸어요", "회원정보"),
    ("전화번호 변경하려고요", "회원정보"),
    ("탈퇴하고 싶습니다", "회원정보"),

    # 쿠폰/적립금
    ("쿠폰 어떻게 쓰나요?", "쿠폰/적립금"),
    ("적립금 얼마 있는지 확인해주세요", "쿠폰/적립금"),
    ("신규 가입 할인 쿠폰 있나요?", "쿠폰/적립금"),

    # 기타
    ("영업시간이 어떻게 되나요?", "기타"),
    ("매장 위치 알려주세요", "기타"),
    ("고마워요 잘 해결됐어요", "기타"),
]

TEST_CASES = TEST_CASES_3000  # 기본: 3000개 사용


OLLAMA_CHAT_API = "http://localhost:11434/api/chat"


def query_model(model: str, user_message: str) -> tuple[str, float]:
    """모델에 쿼리하고 (응답, 소요시간)을 반환 - /api/chat + think:false 사용"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 50},
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_CHAT_API, json=payload, timeout=300)
        elapsed = time.time() - start
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        return answer, elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def normalize_intent(raw: str) -> str:
    """응답에서 Intent 이름을 추출"""
    raw = raw.strip().strip('"').strip("'").strip()
    valid_intents = [
        "주문조회", "배송조회", "환불요청", "교환요청", "상품문의",
        "결제문의", "불만접수", "회원정보", "쿠폰/적립금", "기타",
    ]
    for intent in valid_intents:
        if intent in raw:
            return intent
    return raw[:20]


def run_benchmark():
    print("=" * 70)
    print("  한글 Intent 분석 벤치마크 - gemma4 모델 크기별 비교")
    print("  도메인: 고객 서비스 (주문/배송/환불/문의 등)")
    print(f"  테스트 케이스: {len(TEST_CASES)}개")
    print("=" * 70)

    results = {}

    for model in MODELS:
        print(f"\n🔄 [{model}] 테스트 시작...")
        model_results = []
        correct = 0
        total_time = 0

        for i, (msg, expected) in enumerate(TEST_CASES):
            raw_answer, elapsed = query_model(model, msg)
            predicted = normalize_intent(raw_answer)
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            total_time += elapsed

            icon = "✅" if is_correct else "❌"
            print(f"  {icon} [{i+1:02d}/{len(TEST_CASES)}] {msg[:30]:30s} "
                  f"→ {predicted:10s} (정답: {expected:10s}) [{elapsed:.1f}s]")

            model_results.append({
                "input": msg,
                "expected": expected,
                "predicted": predicted,
                "raw": raw_answer,
                "correct": is_correct,
                "time": elapsed,
            })

        accuracy = correct / len(TEST_CASES) * 100
        avg_time = total_time / len(TEST_CASES)
        results[model] = {
            "details": model_results,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(TEST_CASES),
            "avg_time": avg_time,
            "total_time": total_time,
        }
        print(f"\n  📊 [{model}] 정확도: {accuracy:.1f}% ({correct}/{len(TEST_CASES)}), "
              f"평균 응답시간: {avg_time:.1f}s")

    # 종합 비교표
    print("\n" + "=" * 70)
    print("  📋 종합 비교 결과")
    print("=" * 70)

    summary_table = []
    for model in MODELS:
        r = results[model]
        summary_table.append([
            model, f"{r['accuracy']:.1f}%",
            f"{r['correct']}/{r['total']}",
            f"{r['avg_time']:.1f}s",
            f"{r['total_time']:.0f}s",
        ])

    print(tabulate(
        summary_table,
        headers=["모델", "정확도", "정답수", "평균응답시간", "총소요시간"],
        tablefmt="grid",
    ))

    # Intent별 정확도 비교
    print("\n" + "=" * 70)
    print("  📋 Intent별 정확도 비교")
    print("=" * 70)

    intents = list(dict.fromkeys(exp for _, exp in TEST_CASES))
    intent_table = []
    for intent in intents:
        row = [intent]
        for model in MODELS:
            details = results[model]["details"]
            relevant = [d for d in details if d["expected"] == intent]
            c = sum(1 for d in relevant if d["correct"])
            row.append(f"{c}/{len(relevant)}")
        intent_table.append(row)

    print(tabulate(
        intent_table,
        headers=["Intent"] + MODELS,
        tablefmt="grid",
    ))

    # 오답 상세 분석
    print("\n" + "=" * 70)
    print("  🔍 오답 상세 분석")
    print("=" * 70)

    for model in MODELS:
        errors = [d for d in results[model]["details"] if not d["correct"]]
        if errors:
            print(f"\n  [{model}] 오답 {len(errors)}건:")
            for e in errors:
                print(f"    입력: {e['input']}")
                print(f"    정답: {e['expected']} → 예측: {e['predicted']} (원문: {e['raw'][:50]})")
                print()
        else:
            print(f"\n  [{model}] 오답 없음! 🎉")

    # JSON 결과 저장
    output_path = "/home/azureuser/workspace/h01/benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 상세 결과 저장: {output_path}")


if __name__ == "__main__":
    import sys
    if "--small" in sys.argv:
        TEST_CASES = TEST_CASES_SMALL
        print("📌 소규모 테스트 (30개) 사용")
    else:
        print("📌 대규모 테스트 (3000개) 사용  (--small 옵션으로 30개 모드 가능)")
    run_benchmark()
