#!/usr/bin/env python3
"""
Gemma4 한국어 실용 유즈케이스 벤치마크
- Task 1: PII/NER 탐지 (개인정보·개체명 추출)
- Task 2: 문서 라우팅 (부서별 자동 분류)
- Task 3: 스팸 탐지 (스팸/정상 이메일 분류)

Usage: python3 benchmark_usecase.py
"""

import json
import os
import re
import random
import time
import datetime
import subprocess
import traceback
from pathlib import Path

random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
MODELS = [
    {"name": "e2b (2.3B)", "tag": "gemma4:e2b"},
    {"name": "e4b (4.5B)", "tag": "gemma4:latest"},
    {"name": "26b MoE",    "tag": "gemma4:26b"},
    {"name": "31b Dense",  "tag": "gemma4:31b"},
]
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 600
SAMPLES_PER_TASK = 100
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "awesomekorean_data"
LOG_FILE = BASE_DIR / "benchmark_run.log"
PROGRESS_FILE = BASE_DIR / "benchmark_progress.md"

# ─── Logging ─────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def update_progress(task_name, model_name, done, total, results_so_far=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# Benchmark Progress (updated {ts})\n"]
    lines.append(f"**Current**: {task_name} / {model_name} — {done}/{total}\n")
    if results_so_far:
        lines.append("\n## Completed Results\n")
        lines.append("| Task | Model | Score | Avg Latency |")
        lines.append("|------|-------|-------|-------------|")
        for r in results_so_far:
            lines.append(f"| {r['task']} | {r['model']} | {r['score']:.1f}% | {r['avg_time']:.2f}s |")
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ─── Ollama API ──────────────────────────────────────────────────────────────
def query_ollama(model_tag, prompt, temperature=0):
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
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

# ─── TASK 1: PII/NER Detection ──────────────────────────────────────────────
def load_ner_data():
    """Parse Korean NER corpus and create PII detection test cases."""
    ner_dir = DATA_DIR / "NER" / "말뭉치 - 형태소_개체명"
    if not ner_dir.exists():
        log(f"NER directory not found: {ner_dir}")
        return []

    PII_TYPES = {"PER", "LOC", "ORG", "DAT", "TIM"}
    TYPE_KR = {"PER": "인명", "LOC": "장소", "ORG": "기관", "DAT": "날짜", "TIM": "시간"}

    samples = []
    files = sorted([f for f in ner_dir.iterdir() if f.suffix == ".txt"])
    random.shuffle(files)

    for fpath in files:
        if len(samples) >= SAMPLES_PER_TASK * 3:
            break
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not (line.startswith("## ") and "<" in line and ":" in line):
                        continue
                    entities = re.findall(r"<([^>]+):([A-Z]+)>", line)
                    pii_entities = [(name, etype) for name, etype in entities if etype in PII_TYPES]
                    if not pii_entities:
                        continue
                    raw_text = re.sub(r"<([^>]+):[A-Z]+>", r"\1", line[3:])
                    if len(raw_text) < 10 or len(raw_text) > 300:
                        continue
                    expected = []
                    for name, etype in pii_entities:
                        expected.append({"entity": name, "type": etype, "type_kr": TYPE_KR.get(etype, etype)})
                    samples.append({"text": raw_text, "entities": expected})
        except Exception:
            continue

    random.shuffle(samples)
    return samples[:SAMPLES_PER_TASK]

def evaluate_ner(model_tag, samples, all_results):
    """Evaluate NER/PII detection accuracy."""
    correct = 0
    total = len(samples)
    details = []

    for i, sample in enumerate(samples):
        text = sample["text"]
        expected = sample["entities"]
        expected_str = ", ".join([f"{e['entity']}({e['type_kr']})" for e in expected])

        prompt = f"""다음 한국어 문장에서 개인정보 및 개체명(인명, 장소, 기관, 날짜, 시간)을 추출하세요.

문장: {text}

응답 형식 (JSON 배열):
[{{"entity": "추출된_개체", "type": "인명|장소|기관|날짜|시간"}}]

개체명만 JSON으로 답하세요:"""

        t0 = time.time()
        response = query_ollama(model_tag, prompt)
        elapsed = time.time() - t0

        # Score: check if expected entities appear in response
        found_count = 0
        for ent in expected:
            if ent["entity"] in response:
                found_count += 1

        score = found_count / len(expected) if expected else 0
        is_correct = score >= 0.5  # at least half of entities found
        if is_correct:
            correct += 1

        details.append({
            "index": i,
            "text": text[:100],
            "expected": expected_str,
            "response": response[:300],
            "found_ratio": f"{found_count}/{len(expected)}",
            "correct": is_correct,
            "time": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            log(f"  NER [{i+1}/{total}] running_acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) last_time={elapsed:.1f}s")
            update_progress("PII/NER 탐지", model_tag, i + 1, total, all_results)

    return correct, total, details

# ─── TASK 2: Document Routing ────────────────────────────────────────────────
def generate_doc_routing_data():
    """Generate synthetic Korean business documents for routing classification."""
    categories = {
        "인사": [
            "신규 입사자 {name}의 온보딩 일정을 공유드립니다. {date}부터 OJT가 시작됩니다.",
            "연차 사용 신청합니다. {date} ~ {date2} 개인 사유로 휴가를 신청드립니다.",
            "이번 분기 성과평가 결과를 첨부합니다. 이의신청 기간은 {date}까지입니다.",
            "{name} 사원의 수습 평가 결과 정규직 전환을 추천합니다.",
            "복리후생 제도 변경 안내: {date}부터 건강검진 지원 범위가 확대됩니다.",
            "퇴직금 정산 관련 문의드립니다. {name} 대리 퇴사일은 {date}입니다.",
            "조직개편에 따라 {name} 과장이 마케팅팀에서 기획팀으로 이동합니다.",
            "채용 공고 게시 요청합니다. 개발팀 시니어 엔지니어 {n}명 채용 예정입니다.",
            "직원 교육 프로그램 안내: {date} 리더십 워크숍이 개최됩니다.",
            "급여 명세서 관련 문의입니다. {month}월 야근 수당이 누락된 것 같습니다.",
        ],
        "법무": [
            "{company}와의 업무위탁 계약서 검토를 요청드립니다. 계약 기간은 1년입니다.",
            "특허 출원 준비 중입니다. 발명의 명칭은 '{invention}'이며 출원일은 {date}입니다.",
            "개인정보 처리방침 개정안을 첨부합니다. 법률팀 검토 부탁드립니다.",
            "소송 관련 진행 상황 보고: {company} 건 1심 판결이 {date} 예정입니다.",
            "NDA(비밀유지계약) 체결 요청합니다. 거래처: {company}",
            "상표권 침해 의심 사례를 발견했습니다. 경쟁사 {company}의 유사 상표 관련입니다.",
            "이용약관 변경 공지 초안을 검토해주세요. 시행일은 {date}입니다.",
            "분쟁 조정 회의록을 공유합니다. {company}와의 계약 해지 건입니다.",
            "GDPR 및 개인정보보호법 준수 현황 점검 보고서입니다.",
            "라이선스 계약 갱신 관련입니다. {company}와의 소프트웨어 라이선스 만료일은 {date}입니다.",
        ],
        "재무": [
            "{month}월 매출 보고서를 공유드립니다. 전월 대비 {n}% 증가했습니다.",
            "법인카드 사용 내역 정산 요청합니다. 총 금액은 {amount}원입니다.",
            "내년도 예산 편성 계획안을 첨부합니다. 각 부서별 검토 부탁드립니다.",
            "세금계산서 발행 요청합니다. 거래처: {company}, 금액: {amount}원",
            "분기 실적 보고: 영업이익이 전분기 대비 {n}% 감소했습니다.",
            "출장비 정산 신청합니다. {date} {location} 출장, 총 {amount}원입니다.",
            "미수금 회수 현황입니다. {company}의 미결제 금액은 {amount}원입니다.",
            "투자 수익률 분석 보고서입니다. 올해 ROI는 {n}%입니다.",
            "감사 대비 자료 준비 요청입니다. 외부 감사일은 {date}입니다.",
            "원가 절감 방안을 제안합니다. 연간 약 {amount}원 절감 예상됩니다.",
        ],
        "IT": [
            "서버 장애 보고: {date} {time}부터 웹서버 응답 지연이 발생하고 있습니다.",
            "보안 패치 적용 안내: 전 사 PC에 {date}까지 업데이트를 완료해주세요.",
            "신규 시스템 도입 제안서입니다. ERP 고도화 프로젝트 예산은 {amount}원입니다.",
            "VPN 접속 장애 문의입니다. 재택근무 중 사내망 접속이 안 됩니다.",
            "데이터 백업 정책 변경 안내: 주 1회에서 일 1회로 변경됩니다.",
            "클라우드 마이그레이션 진행 상황입니다. 현재 {n}% 완료되었습니다.",
            "정보보안 교육 이수 안내: {date}까지 전 직원 필수 이수 바랍니다.",
            "네트워크 인프라 증설 요청입니다. {location} 사무실 Wi-Fi 커버리지 확대가 필요합니다.",
            "소프트웨어 라이선스 만료 알림: {software} 라이선스가 {date} 만료됩니다.",
            "사이버 보안 사고 대응 보고서입니다. 피싱 이메일 {n}건이 탐지되었습니다.",
        ],
        "마케팅": [
            "신제품 런칭 캠페인 기획안입니다. 출시일은 {date}이며 예산은 {amount}원입니다.",
            "SNS 마케팅 성과 보고: 이번 달 팔로워 {n}명 증가, 도달률 {n2}% 상승",
            "{date} 프로모션 이벤트 기획안입니다. 할인율 {n}%로 진행 예정입니다.",
            "브랜드 인지도 조사 결과입니다. 목표 대비 {n}% 달성했습니다.",
            "인플루언서 협업 제안서입니다. {name}과 콘텐츠 제작 협업을 제안합니다.",
            "광고 매체 분석 보고서입니다. 네이버 키워드 광고 ROAS가 {n}%입니다.",
            "고객 만족도 설문 결과를 공유합니다. 전체 만족도는 {n}점(5점 만점)입니다.",
            "경쟁사 분석 보고서입니다. {company}의 신규 서비스 출시가 확인되었습니다.",
            "이메일 마케팅 성과: 개봉률 {n}%, 클릭률 {n2}%, 전환율 {n3}%",
            "전시회 참가 보고서입니다. {date} {location}에서 열린 박람회에 참가했습니다.",
        ],
        "고객지원": [
            "고객 {name}님의 불만 접수: 배송 지연으로 인한 환불 요청입니다.",
            "제품 교환 요청 건입니다. 주문번호 {order_no}, 불량 사유: 파손",
            "VIP 고객 관리 보고서입니다. 이번 달 VIP 이탈률은 {n}%입니다.",
            "고객센터 응대 품질 보고: 평균 응답시간 {n}초, 만족도 {n2}점입니다.",
            "반품 처리 요청입니다. 사유: 상품 불일치. 반품 접수번호: {order_no}",
            "자주 묻는 질문 업데이트 요청입니다. 신규 FAQ {n}건을 추가해주세요.",
            "서비스 장애 관련 고객 문의가 {n}건 접수되었습니다. 일괄 안내가 필요합니다.",
            "고객 {name}님의 개인정보 삭제 요청입니다. 처리 기한은 {date}입니다.",
            "AS 접수 현황입니다. 금주 접수 {n}건 중 {n2}건 처리 완료되었습니다.",
            "고객 리텐션 분석 보고서: 재구매율이 전월 대비 {n}% 하락했습니다.",
        ],
    }

    names = ["김민수", "이서연", "박지호", "최유진", "정현우", "강수빈", "조태영", "윤미래", "임재현", "한소희"]
    companies = ["테크솔루션", "글로벌트레이드", "넥스트이노", "코리아텍", "퓨처시스템", "스마트코퍼", "디지털웨이브", "한빛소프트"]
    locations = ["서울 강남", "부산 해운대", "판교 테크노밸리", "여의도", "세종시"]
    inventions = ["AI 기반 음성인식 시스템", "블록체인 결제 솔루션", "자율주행 센서 모듈"]
    softwares = ["Microsoft 365", "Adobe Creative Cloud", "Jira", "Slack Enterprise"]

    samples = []
    for category, templates in categories.items():
        for _ in range(SAMPLES_PER_TASK // len(categories) + 5):
            template = random.choice(templates)
            text = template.format(
                name=random.choice(names),
                date=f"2026년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                date2=f"2026년 {random.randint(1,12)}월 {random.randint(1,28)}일",
                month=random.randint(1, 12),
                n=random.randint(1, 50),
                n2=random.randint(1, 50),
                n3=random.randint(1, 10),
                amount=f"{random.randint(1,99) * 100:,}",
                company=random.choice(companies),
                location=random.choice(locations),
                invention=random.choice(inventions),
                software=random.choice(softwares),
                order_no=f"ORD-{random.randint(100000,999999)}",
                time=f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
            )
            samples.append({"text": text, "category": category})

    random.shuffle(samples)
    # Balance: equal per category
    balanced = []
    per_cat = SAMPLES_PER_TASK // len(categories)
    cat_count = {c: 0 for c in categories}
    for s in samples:
        if cat_count[s["category"]] < per_cat:
            balanced.append(s)
            cat_count[s["category"]] += 1
        if len(balanced) >= SAMPLES_PER_TASK:
            break
    return balanced

def evaluate_doc_routing(model_tag, samples, all_results):
    """Evaluate document routing classification accuracy."""
    categories_list = ["인사", "법무", "재무", "IT", "마케팅", "고객지원"]
    correct = 0
    total = len(samples)
    details = []

    for i, sample in enumerate(samples):
        text = sample["text"]
        expected = sample["category"]

        prompt = f"""다음 업무 문서를 읽고 해당 부서를 분류하세요.

부서 목록: {', '.join(categories_list)}

문서: {text}

위 부서 중 하나만 답하세요:"""

        t0 = time.time()
        response = query_ollama(model_tag, prompt)
        elapsed = time.time() - t0

        predicted = response.strip()
        is_correct = expected in predicted

        # Also check if exact category appears
        if not is_correct:
            for cat in categories_list:
                if cat in predicted:
                    is_correct = (cat == expected)
                    break

        if is_correct:
            correct += 1

        details.append({
            "index": i,
            "text": text[:80],
            "expected": expected,
            "predicted": predicted[:100],
            "correct": is_correct,
            "time": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            log(f"  DocRoute [{i+1}/{total}] running_acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) last_time={elapsed:.1f}s")
            update_progress("문서 라우팅", model_tag, i + 1, total, all_results)

    return correct, total, details

# ─── TASK 3: Spam Detection ──────────────────────────────────────────────────
def generate_spam_data():
    """Generate synthetic Korean spam/ham email classification data."""
    spam_templates = [
        "[긴급] 축하합니다! 이벤트 당첨금 {amount}만원을 수령하세요. 링크: http://bit.ly/{code}",
        "【무료 체험】 다이어트 보조제 {n}일 무료! 지금 신청하면 {n2}% 할인 → {url}",
        "대출 한도 조회 결과: {amount}만원 승인! 금리 {n}% 최저금리 상담 {phone}",
        "미수령 택배가 있습니다. 배송조회: {url} (본인확인 필요)",
        "{name}님 카드 해외결제 {amount}만원 승인. 본인 아닐 시 즉시 신고 {phone}",
        "★특가★ 명품가방 {n}% 할인! 한정수량 {n2}개! 구매링크: {url}",
        "건강보험 환급금 {amount}만원이 미수령 상태입니다. 확인하기: {url}",
        "재택부업으로 월 {amount}만원! 하루 {n}시간 투자로 수익 보장 {phone}",
        "국세청 세금 환급 안내: {amount}만원 미수령. 본인인증 후 수령 {url}",
        "[경고] 회원님의 계정이 비정상 로그인되었습니다. 비밀번호 변경: {url}",
        "코인 투자 수익률 {n}%! 전문가의 무료 리딩방 참여하세요 {url}",
        "무료 상담! 이혼/재산분할 전문 법률사무소 {phone}",
        "카지노 신규가입 보너스 {amount}만원! VIP 무료체험 {url}",
        "보험료 비교견적 무료! 최대 {n}% 절약 가능. 상담 {phone}",
        "대환대출 가능! 기존 {n}% → {n2}% 금리 인하. 즉시 상담 {phone}",
    ]
    ham_templates = [
        "{name}님, 내일 {time} 회의 참석 확인 부탁드립니다. 회의실: {location}",
        "안녕하세요, {date} 납품 일정 관련하여 확인 부탁드립니다. 감사합니다.",
        "프로젝트 주간보고서를 첨부합니다. 이번 주 진행 사항을 확인해주세요.",
        "{name} 팀장님, 계약서 수정본 전달드립니다. 검토 후 회신 부탁드립니다.",
        "이번 달 팀 회식은 {date} {time}에 {location}에서 진행합니다. 참석 여부를 알려주세요.",
        "출장 보고서를 제출합니다. {location} 현장 방문 결과를 정리했습니다.",
        "보고서 리뷰 감사합니다. 피드백 반영하여 수정본을 다시 보내드리겠습니다.",
        "{name}님, 신규 프로젝트 킥오프 미팅 일정을 조율하고 싶습니다. 가능한 시간을 알려주세요.",
        "안녕하세요, 지난 미팅에서 논의한 API 연동 건 진행 상황을 공유드립니다.",
        "인수인계 자료를 정리하여 공유 폴더에 업로드했습니다. 확인 부탁드립니다.",
        "{name}님, 고객사 미팅이 {date}로 확정되었습니다. 발표 자료 준비를 부탁드립니다.",
        "안녕하세요, 면접 일정 안내드립니다. {date} {time} 온라인 면접입니다.",
        "부서 이동 안내: {date}부로 {name}님이 우리 팀에 합류하십니다. 환영해주세요.",
        "이번 분기 OKR 설정 마감일은 {date}입니다. 기한 내 입력 부탁드립니다.",
        "팀 세미나 안내: {date} {time}, 주제 '클라우드 아키텍처 설계'. 많은 참여 바랍니다.",
    ]

    names = ["김민수", "이서연", "박지호", "최유진", "정현우", "강수빈", "조태영"]
    locations = ["강남 본사 3층", "판교 오피스", "부산 지사", "여의도 회의실"]
    urls = ["http://bit.ly/a1b2c3", "https://t.co/xyz123", "http://cutt.ly/promo"]
    phones = ["010-1234-5678", "1588-0000", "02-555-1234"]

    samples = []
    for _ in range(SAMPLES_PER_TASK // 2 + 5):
        template = random.choice(spam_templates)
        text = template.format(
            name=random.choice(names), amount=random.randint(10, 500),
            n=random.randint(2, 90), n2=random.randint(2, 50),
            code=f"{''.join(random.choices('abcdef1234', k=6))}",
            url=random.choice(urls), phone=random.choice(phones),
            date=f"{random.randint(1,12)}월 {random.randint(1,28)}일",
            time=f"{random.randint(9,18)}시", location=random.choice(locations),
        )
        samples.append({"text": text, "label": "스팸"})

    for _ in range(SAMPLES_PER_TASK // 2 + 5):
        template = random.choice(ham_templates)
        text = template.format(
            name=random.choice(names), amount=random.randint(10, 500),
            n=random.randint(2, 90), n2=random.randint(2, 50),
            date=f"{random.randint(1,12)}월 {random.randint(1,28)}일",
            time=f"{random.randint(9,18)}시", location=random.choice(locations),
        )
        samples.append({"text": text, "label": "정상"})

    random.shuffle(samples)
    return samples[:SAMPLES_PER_TASK]

def evaluate_spam(model_tag, samples, all_results):
    """Evaluate spam detection accuracy."""
    correct = 0
    total = len(samples)
    details = []

    for i, sample in enumerate(samples):
        text = sample["text"]
        expected = sample["label"]

        prompt = f"""다음 메시지가 스팸인지 정상인지 판별하세요.

메시지: {text}

"스팸" 또는 "정상" 중 하나만 답하세요:"""

        t0 = time.time()
        response = query_ollama(model_tag, prompt)
        elapsed = time.time() - t0

        predicted = response.strip()
        is_correct = expected in predicted

        if is_correct:
            correct += 1

        details.append({
            "index": i,
            "text": text[:80],
            "expected": expected,
            "predicted": predicted[:100],
            "correct": is_correct,
            "time": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            log(f"  Spam [{i+1}/{total}] running_acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) last_time={elapsed:.1f}s")
            update_progress("스팸 탐지", model_tag, i + 1, total, all_results)

    return correct, total, details

# ─── Report Generation ───────────────────────────────────────────────────────
def generate_report(all_results, all_details, start_time):
    elapsed_total = time.time() - start_time
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    report = []
    report.append("# Gemma4 한국어 실용 유즈케이스 벤치마크 결과\n")
    report.append(f"**실행 시간**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**총 소요 시간**: {elapsed_total/3600:.1f}시간")
    report.append(f"**환경**: Azure VM, 8 vCPU, 31GB RAM, CPU only, Ollama")
    report.append(f"**모드**: think OFF, temperature 0, 태스크당 {SAMPLES_PER_TASK}샘플\n")

    # Summary table
    report.append("## 종합 결과 (Accuracy %)\n")
    report.append("| Task | Use Case | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |")
    report.append("|------|----------|:---:|:---:|:---:|:---:|")

    tasks = ["PII/NER 탐지", "문서 라우팅", "스팸 탐지"]
    task_use = {"PII/NER 탐지": "개인정보 탐지", "문서 라우팅": "부서 자동 분류", "스팸 탐지": "스팸 메일 필터링"}
    for task in tasks:
        row = f"| {task} | {task_use.get(task, '')} |"
        for model in MODELS:
            r = next((x for x in all_results if x["task"] == task and x["model"] == model["name"]), None)
            if r:
                score = r["score"]
                bold = "**" if score == max(x["score"] for x in all_results if x["task"] == task) else ""
                row += f" {bold}{score:.1f}{bold} |"
            else:
                row += " - |"
        report.append(row)

    # Average
    row = "| **평균** | |"
    for model in MODELS:
        scores = [x["score"] for x in all_results if x["model"] == model["name"]]
        avg = sum(scores) / len(scores) if scores else 0
        row += f" **{avg:.1f}** |"
    report.append(row)

    # Speed table
    report.append("\n## 응답 속도 (avg seconds/query)\n")
    report.append("| Task | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |")
    report.append("|------|:---:|:---:|:---:|:---:|")
    for task in tasks:
        row = f"| {task} |"
        for model in MODELS:
            r = next((x for x in all_results if x["task"] == task and x["model"] == model["name"]), None)
            if r:
                row += f" {r['avg_time']:.2f}s |"
            else:
                row += " - |"
        report.append(row)

    # Error analysis per task
    report.append("\n## 태스크별 분석\n")
    for task in tasks:
        report.append(f"### {task}\n")
        for model in MODELS:
            r = next((x for x in all_results if x["task"] == task and x["model"] == model["name"]), None)
            if r:
                report.append(f"- **{model['name']}**: {r['score']:.1f}% ({r['correct']}/{r['total']}), avg {r['avg_time']:.2f}s/query")
        report.append("")

    # Recommendations
    report.append("## 유즈케이스별 모델 추천\n")
    report.append("| 유즈케이스 | 추천 모델 | 정확도 | 이유 |")
    report.append("|-----------|----------|--------|------|")
    for task in tasks:
        task_results = [x for x in all_results if x["task"] == task]
        if task_results:
            best = max(task_results, key=lambda x: x["score"])
            # Find best cost-effective (>85% and fastest)
            good_enough = [x for x in task_results if x["score"] >= 85]
            if good_enough:
                fastest = min(good_enough, key=lambda x: x["avg_time"])
                report.append(f"| {task} | {fastest['model']} | {fastest['score']:.1f}% | 속도-정확도 밸런스 |")
            else:
                report.append(f"| {task} | {best['model']} | {best['score']:.1f}% | 최고 정확도 |")

    report_text = "\n".join(report) + "\n"

    # Save report
    report_path = BASE_DIR / f"RESULTS_{ts}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log(f"Report saved: {report_path}")

    # Also save as latest
    with open(BASE_DIR / "RESULTS_LATEST.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save raw JSON
    raw = {
        "timestamp": ts,
        "config": {"models": [m["name"] for m in MODELS], "samples_per_task": SAMPLES_PER_TASK},
        "results": all_results,
        "details": {k: v for k, v in all_details.items()},
    }
    json_path = BASE_DIR / f"results_raw_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    log(f"Raw JSON saved: {json_path}")

    return report_text, ts

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log("=" * 60)
    log("Gemma4 한국어 실용 유즈케이스 벤치마크 시작")
    log("=" * 60)

    # Load/generate data
    log("데이터 준비 중...")
    ner_data = load_ner_data()
    doc_data = generate_doc_routing_data()
    spam_data = generate_spam_data()
    log(f"  NER: {len(ner_data)} samples, DocRoute: {len(doc_data)} samples, Spam: {len(spam_data)} samples")

    tasks = [
        ("PII/NER 탐지", ner_data, evaluate_ner),
        ("문서 라우팅", doc_data, evaluate_doc_routing),
        ("스팸 탐지", spam_data, evaluate_spam),
    ]

    all_results = []
    all_details = {}

    for task_name, data, eval_fn in tasks:
        log(f"\n{'='*60}")
        log(f"Task: {task_name}")
        log(f"{'='*60}")

        for model in MODELS:
            log(f"\n--- {model['name']} ({model['tag']}) ---")

            # Warmup
            log(f"  Warming up {model['tag']}...")
            warmup_resp = query_ollama(model["tag"], "안녕하세요")
            log(f"  Warmup response: {warmup_resp[:50]}")

            t0 = time.time()
            correct, total, details = eval_fn(model["tag"], data, all_results)
            wall_time = time.time() - t0
            avg_time = wall_time / total if total > 0 else 0
            score = 100 * correct / total if total > 0 else 0

            result = {
                "task": task_name,
                "model": model["name"],
                "score": round(score, 1),
                "correct": correct,
                "total": total,
                "avg_time": round(avg_time, 2),
                "wall_time": round(wall_time, 1),
            }
            all_results.append(result)
            all_details[f"{task_name}_{model['name']}"] = details

            log(f"  ✅ {model['name']}: {score:.1f}% ({correct}/{total}) avg={avg_time:.2f}s wall={wall_time:.0f}s")

            # Save intermediate
            with open(BASE_DIR / "results_intermediate.json", "w", encoding="utf-8") as f:
                json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)

    # Generate report
    log("\n" + "=" * 60)
    log("벤치마크 완료! 리포트 생성 중...")
    report_text, ts = generate_report(all_results, all_details, start_time)

    total_time = time.time() - start_time
    log(f"총 소요 시간: {total_time/3600:.1f}시간 ({total_time:.0f}초)")
    log("완료!")

if __name__ == "__main__":
    main()
