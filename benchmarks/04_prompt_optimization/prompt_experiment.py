#!/usr/bin/env python3
"""
프롬프트 최적화 Before/After 비교 실험
- Baseline: 기존 프롬프트 (단순 지시)
- Optimized: System Prompt + Few-shot + 출력 제약 강화

3개 태스크 × 3개 전략 × 2개 모델 (e4b, 31b)
태스크당 50샘플로 빠르게 비교
"""

import json
import os
import re
import random
import time
import datetime
import urllib.request
from pathlib import Path

random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
MODELS = [
    {"name": "e4b (4.5B)", "tag": "gemma4:latest"},
    {"name": "31b Dense",  "tag": "gemma4:31b"},
]
OLLAMA_URL = "http://localhost:11434/api/generate"
TIMEOUT = 600
SAMPLES = 50
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "awesomekorean_data"
LOG_FILE = BASE_DIR / "prompt_experiment.log"
PROGRESS_FILE = BASE_DIR / "experiment_progress.md"

STRATEGIES = ["baseline", "system_prompt", "few_shot"]
STRATEGY_KR = {
    "baseline": "기본 프롬프트",
    "system_prompt": "시스템 프롬프트 추가",
    "few_shot": "시스템 프롬프트 + Few-shot 예시",
}

# ─── System Prompts ──────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "ner": "당신은 한국어 개체명 인식(NER) 전문가입니다. 텍스트에서 인명, 장소, 기관, 날짜, 시간을 정확하게 추출합니다. 반드시 JSON 형식으로만 답하세요. 설명이나 부연은 절대 하지 마세요.",
    "doc_routing": "당신은 한국어 문서 분류 전문가입니다. 업무 문서를 읽고 해당 부서를 정확히 판별합니다. 부서명 한 단어만 답하세요. 설명이나 이유는 절대 쓰지 마세요.",
    "spam": "당신은 한국어 스팸 탐지 전문가입니다. 메시지를 분석하여 스팸 여부를 판별합니다. '스팸' 또는 '정상' 한 단어만 답하세요. 설명이나 이유는 절대 쓰지 마세요.",
}

# ─── Few-shot Examples ───────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = {
    "ner": [
        {
            "input": "삼성전자 이재용 회장이 2024년 3월 서울에서 기자회견을 열었다.",
            "output": '[{"entity": "삼성전자", "type": "기관"}, {"entity": "이재용", "type": "인명"}, {"entity": "2024년 3월", "type": "날짜"}, {"entity": "서울", "type": "장소"}]',
        },
        {
            "input": "오후 2시에 한국은행 총재가 부산 벡스코에서 발표할 예정이다.",
            "output": '[{"entity": "오후 2시", "type": "시간"}, {"entity": "한국은행", "type": "기관"}, {"entity": "부산", "type": "장소"}, {"entity": "벡스코", "type": "장소"}]',
        },
    ],
    "doc_routing": [
        {"input": "신규 채용 면접 일정을 조율해주세요. 지원자 3명의 면접이 내주에 예정되어 있습니다.", "output": "인사"},
        {"input": "이번 분기 매출 보고서를 작성했습니다. 전분기 대비 15% 성장했습니다.", "output": "재무"},
        {"input": "거래처와의 NDA 계약서 초안을 검토해주세요.", "output": "법무"},
    ],
    "spam": [
        {"input": "내일 오전 10시 회의 참석 부탁드립니다. 안건은 Q2 계획입니다.", "output": "정상"},
        {"input": "축하합니다! 100만원 당첨! 지금 클릭하세요 → http://bit.ly/xyz", "output": "스팸"},
        {"input": "프로젝트 진행 보고서를 첨부합니다. 확인 부탁드립니다.", "output": "정상"},
    ],
}

# ─── Logging ─────────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def update_progress(info, all_results):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# 프롬프트 최적화 실험 진행 현황 ({ts})\n"]
    lines.append(f"**현재**: {info}\n")
    if all_results:
        lines.append("\n## 완료된 결과\n")
        lines.append("| Task | Model | Strategy | Accuracy | Avg Latency |")
        lines.append("|------|-------|----------|----------|-------------|")
        for r in all_results:
            lines.append(f"| {r['task']} | {r['model']} | {r['strategy_kr']} | {r['accuracy']:.1f}% | {r['avg_time']:.2f}s |")
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# ─── Ollama API ──────────────────────────────────────────────────────────────
def query_ollama(model_tag, prompt, system=None):
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 512},
        "think": False,
    }
    if system:
        payload["system"] = system
    try:
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

# ─── Data Loading (same as benchmark_usecase.py) ─────────────────────────────
def load_ner_data():
    ner_dir = DATA_DIR / "NER" / "말뭉치 - 형태소_개체명"
    PII_TYPES = {"PER", "LOC", "ORG", "DAT", "TIM"}
    TYPE_KR = {"PER": "인명", "LOC": "장소", "ORG": "기관", "DAT": "날짜", "TIM": "시간"}
    samples = []
    files = sorted([f for f in ner_dir.iterdir() if f.suffix == ".txt"])
    random.shuffle(files)
    for fpath in files:
        if len(samples) >= SAMPLES * 3:
            break
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not (line.startswith("## ") and "<" in line and ":" in line):
                        continue
                    entities = re.findall(r"<([^>]+):([A-Z]+)>", line)
                    pii = [(n, t) for n, t in entities if t in PII_TYPES]
                    if not pii:
                        continue
                    raw = re.sub(r"<([^>]+):[A-Z]+>", r"\1", line[3:])
                    if len(raw) < 10 or len(raw) > 300:
                        continue
                    expected = [{"entity": n, "type": t, "type_kr": TYPE_KR.get(t, t)} for n, t in pii]
                    samples.append({"text": raw, "entities": expected})
        except Exception:
            continue
    random.shuffle(samples)
    return samples[:SAMPLES]

def generate_doc_routing_data():
    categories = {
        "인사": [
            "신규 입사자 김민수의 온보딩 일정을 공유드립니다. 2026년 5월부터 OJT가 시작됩니다.",
            "연차 사용 신청합니다. 7월 1일 ~ 7월 3일 개인 사유로 휴가를 신청드립니다.",
            "이번 분기 성과평가 결과를 첨부합니다. 이의신청 기간은 6월 30일까지입니다.",
            "채용 공고 게시 요청합니다. 개발팀 시니어 엔지니어 2명 채용 예정입니다.",
            "직원 교육 프로그램 안내: 8월 리더십 워크숍이 개최됩니다.",
            "급여 명세서 관련 문의입니다. 3월 야근 수당이 누락된 것 같습니다.",
            "퇴직금 정산 관련 문의드립니다. 이서연 대리 퇴사일은 4월 15일입니다.",
            "조직개편에 따라 박지호 과장이 마케팅팀에서 기획팀으로 이동합니다.",
            "복리후생 제도 변경 안내: 5월부터 건강검진 지원 범위가 확대됩니다.",
            "수습 평가 결과 정규직 전환을 추천합니다. 업무 역량이 우수합니다.",
        ],
        "법무": [
            "테크솔루션과의 업무위탁 계약서 검토를 요청드립니다. 계약 기간은 1년입니다.",
            "특허 출원 준비 중입니다. 발명의 명칭은 'AI 음성인식 시스템'이며 출원일은 6월입니다.",
            "개인정보 처리방침 개정안을 첨부합니다. 법률팀 검토 부탁드립니다.",
            "소송 관련 진행 상황 보고: 글로벌트레이드 건 1심 판결이 9월 예정입니다.",
            "NDA(비밀유지계약) 체결 요청합니다. 거래처: 넥스트이노",
            "상표권 침해 의심 사례를 발견했습니다. 경쟁사의 유사 상표 관련입니다.",
            "이용약관 변경 공지 초안을 검토해주세요. 시행일은 7월 1일입니다.",
            "GDPR 및 개인정보보호법 준수 현황 점검 보고서입니다.",
            "라이선스 계약 갱신 관련입니다. 소프트웨어 라이선스 만료일은 12월입니다.",
            "분쟁 조정 회의록을 공유합니다. 계약 해지 건에 대한 합의가 필요합니다.",
        ],
        "재무": [
            "4월 매출 보고서를 공유드립니다. 전월 대비 12% 증가했습니다.",
            "법인카드 사용 내역 정산 요청합니다. 총 금액은 350만원입니다.",
            "내년도 예산 편성 계획안을 첨부합니다. 각 부서별 검토 부탁드립니다.",
            "세금계산서 발행 요청합니다. 거래처: 코리아텍, 금액: 500만원",
            "분기 실적 보고: 영업이익이 전분기 대비 8% 감소했습니다.",
            "출장비 정산 신청합니다. 부산 출장, 총 45만원입니다.",
            "미수금 회수 현황입니다. 퓨처시스템의 미결제 금액은 1,200만원입니다.",
            "투자 수익률 분석 보고서입니다. 올해 ROI는 15%입니다.",
            "감사 대비 자료 준비 요청입니다. 외부 감사일은 11월입니다.",
            "원가 절감 방안을 제안합니다. 연간 약 3,000만원 절감 예상됩니다.",
        ],
        "IT": [
            "서버 장애 보고: 오늘 오전 9시부터 웹서버 응답 지연이 발생하고 있습니다.",
            "보안 패치 적용 안내: 전 사 PC에 금주까지 업데이트를 완료해주세요.",
            "신규 시스템 도입 제안서입니다. ERP 고도화 프로젝트 예산은 2억원입니다.",
            "VPN 접속 장애 문의입니다. 재택근무 중 사내망 접속이 안 됩니다.",
            "데이터 백업 정책 변경 안내: 주 1회에서 일 1회로 변경됩니다.",
            "클라우드 마이그레이션 진행 상황입니다. 현재 75% 완료되었습니다.",
            "정보보안 교육 이수 안내: 6월까지 전 직원 필수 이수 바랍니다.",
            "네트워크 인프라 증설 요청입니다. 판교 사무실 Wi-Fi 커버리지 확대가 필요합니다.",
            "소프트웨어 라이선스 만료 알림: Jira 라이선스가 8월 만료됩니다.",
            "사이버 보안 사고 대응 보고서입니다. 피싱 이메일 15건이 탐지되었습니다.",
        ],
        "마케팅": [
            "신제품 런칭 캠페인 기획안입니다. 출시일은 9월이며 예산은 5,000만원입니다.",
            "SNS 마케팅 성과 보고: 이번 달 팔로워 2,000명 증가, 도달률 35% 상승",
            "프로모션 이벤트 기획안입니다. 할인율 20%로 진행 예정입니다.",
            "브랜드 인지도 조사 결과입니다. 목표 대비 85% 달성했습니다.",
            "인플루언서 협업 제안서입니다. 유명 유튜버와 콘텐츠 제작 협업을 제안합니다.",
            "광고 매체 분석 보고서입니다. 네이버 키워드 광고 ROAS가 320%입니다.",
            "고객 만족도 설문 결과를 공유합니다. 전체 만족도는 4.2점(5점 만점)입니다.",
            "경쟁사 분석 보고서입니다. 디지털웨이브의 신규 서비스 출시가 확인되었습니다.",
            "이메일 마케팅 성과: 개봉률 25%, 클릭률 8%, 전환율 3%",
            "전시회 참가 보고서입니다. 서울 코엑스에서 열린 박람회에 참가했습니다.",
        ],
        "고객지원": [
            "고객 최유진님의 불만 접수: 배송 지연으로 인한 환불 요청입니다.",
            "제품 교환 요청 건입니다. 주문번호 ORD-456789, 불량 사유: 파손",
            "VIP 고객 관리 보고서입니다. 이번 달 VIP 이탈률은 5%입니다.",
            "고객센터 응대 품질 보고: 평균 응답시간 30초, 만족도 4.5점입니다.",
            "반품 처리 요청입니다. 사유: 상품 불일치. 반품 접수번호: ORD-123456",
            "자주 묻는 질문 업데이트 요청입니다. 신규 FAQ 5건을 추가해주세요.",
            "서비스 장애 관련 고객 문의가 50건 접수되었습니다. 일괄 안내가 필요합니다.",
            "고객 정현우님의 개인정보 삭제 요청입니다. 처리 기한은 7월 15일입니다.",
            "AS 접수 현황입니다. 금주 접수 120건 중 95건 처리 완료되었습니다.",
            "고객 리텐션 분석 보고서: 재구매율이 전월 대비 3% 하락했습니다.",
        ],
    }
    samples = []
    per_cat = SAMPLES // len(categories)
    for cat, texts in categories.items():
        chosen = random.sample(texts, min(per_cat, len(texts)))
        for text in chosen:
            samples.append({"text": text, "category": cat})
    random.shuffle(samples)
    return samples[:SAMPLES]

def generate_spam_data():
    spam_texts = [
        "[긴급] 축하합니다! 이벤트 당첨금 200만원을 수령하세요. 링크: http://bit.ly/abc123",
        "【무료 체험】 다이어트 보조제 7일 무료! 지금 신청하면 50% 할인 → http://cutt.ly/promo",
        "대출 한도 조회 결과: 3,000만원 승인! 금리 2.9% 최저금리 상담 010-1234-5678",
        "미수령 택배가 있습니다. 배송조회: http://t.co/xyz (본인확인 필요)",
        "카드 해외결제 150만원 승인. 본인 아닐 시 즉시 신고 1588-0000",
        "★특가★ 명품가방 70% 할인! 한정수량 50개! 구매링크: http://bit.ly/sale",
        "건강보험 환급금 45만원이 미수령 상태입니다. 확인하기: http://cutt.ly/check",
        "재택부업으로 월 300만원! 하루 2시간 투자로 수익 보장 010-9876-5432",
        "국세청 세금 환급 안내: 80만원 미수령. 본인인증 후 수령 http://t.co/tax",
        "[경고] 회원님의 계정이 비정상 로그인되었습니다. 비밀번호 변경: http://bit.ly/pwd",
        "코인 투자 수익률 500%! 전문가의 무료 리딩방 참여하세요 http://t.co/coin",
        "카지노 신규가입 보너스 100만원! VIP 무료체험 http://bit.ly/casino",
        "보험료 비교견적 무료! 최대 40% 절약 가능. 상담 1588-1234",
    ]
    ham_texts = [
        "내일 오전 10시 회의 참석 확인 부탁드립니다. 회의실: 강남 본사 3층",
        "안녕하세요, 5월 납품 일정 관련하여 확인 부탁드립니다. 감사합니다.",
        "프로젝트 주간보고서를 첨부합니다. 이번 주 진행 사항을 확인해주세요.",
        "계약서 수정본 전달드립니다. 검토 후 회신 부탁드립니다.",
        "이번 달 팀 회식은 금요일 저녁 7시에 판교 맛집에서 진행합니다.",
        "출장 보고서를 제출합니다. 부산 현장 방문 결과를 정리했습니다.",
        "보고서 리뷰 감사합니다. 피드백 반영하여 수정본을 다시 보내드리겠습니다.",
        "신규 프로젝트 킥오프 미팅 일정을 조율하고 싶습니다. 가능한 시간을 알려주세요.",
        "지난 미팅에서 논의한 API 연동 건 진행 상황을 공유드립니다.",
        "인수인계 자료를 정리하여 공유 폴더에 업로드했습니다. 확인 부탁드립니다.",
        "고객사 미팅이 다음 주 화요일로 확정되었습니다. 발표 자료 준비 부탁드립니다.",
        "부서 이동 안내: 다음 달부로 김민수님이 우리 팀에 합류하십니다.",
        "팀 세미나 안내: 금주 금요일, 주제 '클라우드 아키텍처 설계'. 많은 참여 바랍니다.",
    ]
    samples = []
    for i in range(SAMPLES // 2):
        samples.append({"text": spam_texts[i % len(spam_texts)], "label": "스팸"})
    for i in range(SAMPLES // 2):
        samples.append({"text": ham_texts[i % len(ham_texts)], "label": "정상"})
    random.shuffle(samples)
    return samples

# ─── Prompt Builders ─────────────────────────────────────────────────────────
def build_ner_prompt(text, strategy):
    if strategy == "baseline":
        return None, f"""다음 한국어 문장에서 개인정보 및 개체명(인명, 장소, 기관, 날짜, 시간)을 추출하세요.

문장: {text}

응답 형식 (JSON 배열):
[{{"entity": "추출된_개체", "type": "인명|장소|기관|날짜|시간"}}]

개체명만 JSON으로 답하세요:"""

    system = SYSTEM_PROMPTS["ner"]

    if strategy == "system_prompt":
        return system, f"""문장: {text}

개체명을 JSON 배열로 추출하세요:"""

    # few_shot
    examples = ""
    for ex in FEW_SHOT_EXAMPLES["ner"]:
        examples += f"\n문장: {ex['input']}\n답: {ex['output']}\n"
    return system, f"""예시:{examples}
문장: {text}
답:"""

def build_doc_prompt(text, strategy):
    categories_list = "인사, 법무, 재무, IT, 마케팅, 고객지원"
    if strategy == "baseline":
        return None, f"""다음 업무 문서를 읽고 해당 부서를 분류하세요.

부서 목록: {categories_list}

문서: {text}

위 부서 중 하나만 답하세요:"""

    system = SYSTEM_PROMPTS["doc_routing"]

    if strategy == "system_prompt":
        return system, f"""부서 목록: {categories_list}

문서: {text}

부서:"""

    # few_shot
    examples = ""
    for ex in FEW_SHOT_EXAMPLES["doc_routing"]:
        examples += f"\n문서: {ex['input']}\n부서: {ex['output']}\n"
    return system, f"""부서 목록: {categories_list}

예시:{examples}
문서: {text}
부서:"""

def build_spam_prompt(text, strategy):
    if strategy == "baseline":
        return None, f"""다음 메시지가 스팸인지 정상인지 판별하세요.

메시지: {text}

"스팸" 또는 "정상" 중 하나만 답하세요:"""

    system = SYSTEM_PROMPTS["spam"]

    if strategy == "system_prompt":
        return system, f"""메시지: {text}

판별:"""

    # few_shot
    examples = ""
    for ex in FEW_SHOT_EXAMPLES["spam"]:
        examples += f"\n메시지: {ex['input']}\n판별: {ex['output']}\n"
    return system, f"""예시:{examples}
메시지: {text}
판별:"""

# ─── Evaluation Functions ────────────────────────────────────────────────────
def eval_ner(model_tag, samples, strategy, all_results):
    correct = 0
    total = len(samples)
    times = []
    for i, sample in enumerate(samples):
        system, prompt = build_ner_prompt(sample["text"], strategy)
        t0 = time.time()
        response = query_ollama(model_tag, prompt, system=system)
        elapsed = time.time() - t0
        times.append(elapsed)
        found = sum(1 for e in sample["entities"] if e["entity"] in response)
        if found >= len(sample["entities"]) * 0.5:
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    NER [{i+1}/{total}] acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) t={elapsed:.1f}s")
            update_progress(f"NER / {model_tag} / {strategy} [{i+1}/{total}]", all_results)
    return correct, total, sum(times) / len(times)

def eval_doc(model_tag, samples, strategy, all_results):
    categories_list = ["인사", "법무", "재무", "IT", "마케팅", "고객지원"]
    correct = 0
    total = len(samples)
    times = []
    for i, sample in enumerate(samples):
        system, prompt = build_doc_prompt(sample["text"], strategy)
        t0 = time.time()
        response = query_ollama(model_tag, prompt, system=system)
        elapsed = time.time() - t0
        times.append(elapsed)
        expected = sample["category"]
        is_correct = expected in response.strip()
        if not is_correct:
            for cat in categories_list:
                if cat in response.strip():
                    is_correct = (cat == expected)
                    break
        if is_correct:
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    Doc [{i+1}/{total}] acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) t={elapsed:.1f}s")
            update_progress(f"DocRoute / {model_tag} / {strategy} [{i+1}/{total}]", all_results)
    return correct, total, sum(times) / len(times)

def eval_spam(model_tag, samples, strategy, all_results):
    correct = 0
    total = len(samples)
    times = []
    for i, sample in enumerate(samples):
        system, prompt = build_spam_prompt(sample["text"], strategy)
        t0 = time.time()
        response = query_ollama(model_tag, prompt, system=system)
        elapsed = time.time() - t0
        times.append(elapsed)
        if sample["label"] in response.strip():
            correct += 1
        if (i + 1) % 10 == 0:
            log(f"    Spam [{i+1}/{total}] acc={correct}/{i+1} ({100*correct/(i+1):.1f}%) t={elapsed:.1f}s")
            update_progress(f"Spam / {model_tag} / {strategy} [{i+1}/{total}]", all_results)
    return correct, total, sum(times) / len(times)

# ─── Report ──────────────────────────────────────────────────────────────────
def generate_report(all_results, start_time):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    elapsed = time.time() - start_time

    lines = ["# 프롬프트 최적화 Before/After 비교 실험 결과\n"]
    lines.append(f"**실행 시간**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**총 소요**: {elapsed/3600:.1f}시간")
    lines.append(f"**환경**: Azure VM, 8 vCPU, 31GB RAM, CPU only, Ollama, think OFF")
    lines.append(f"**샘플 수**: 태스크당 {SAMPLES}개\n")

    lines.append("## 전략별 설명\n")
    lines.append("| 전략 | 설명 |")
    lines.append("|------|------|")
    lines.append("| baseline | 단순 지시 프롬프트 (설명 + 질문) |")
    lines.append("| system_prompt | 역할 부여 시스템 프롬프트 + 간결한 지시 + 출력 제약 |")
    lines.append("| few_shot | 시스템 프롬프트 + 2~3개 예시 + 출력 제약 |")

    # Per-model results
    for model in MODELS:
        mn = model["name"]
        lines.append(f"\n## {mn}\n")
        lines.append("| Task | baseline | +system_prompt | +few_shot | 개선폭 |")
        lines.append("|------|:---:|:---:|:---:|:---:|")

        tasks = ["PII/NER 탐지", "문서 라우팅", "스팸 탐지"]
        for task in tasks:
            base = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "baseline"), None)
            sys_r = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "system_prompt"), None)
            fs_r = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "few_shot"), None)

            base_acc = base["accuracy"] if base else 0
            sys_acc = sys_r["accuracy"] if sys_r else 0
            fs_acc = fs_r["accuracy"] if fs_r else 0
            best = max(base_acc, sys_acc, fs_acc)
            delta = best - base_acc

            lines.append(f"| {task} | {base_acc:.1f}% | {sys_acc:.1f}% | {fs_acc:.1f}% | **+{delta:.1f}%p** |")

        # Average
        base_avg = sum(r["accuracy"] for r in all_results if r["model"] == mn and r["strategy"] == "baseline") / 3
        sys_avg = sum(r["accuracy"] for r in all_results if r["model"] == mn and r["strategy"] == "system_prompt") / 3
        fs_avg = sum(r["accuracy"] for r in all_results if r["model"] == mn and r["strategy"] == "few_shot") / 3
        best_avg = max(base_avg, sys_avg, fs_avg)
        lines.append(f"| **평균** | **{base_avg:.1f}%** | **{sys_avg:.1f}%** | **{fs_avg:.1f}%** | **+{best_avg - base_avg:.1f}%p** |")

    # Speed comparison
    lines.append("\n## 응답 속도 비교 (avg sec/query)\n")
    lines.append("| Task | Model | baseline | +system_prompt | +few_shot |")
    lines.append("|------|-------|:---:|:---:|:---:|")
    for task in ["PII/NER 탐지", "문서 라우팅", "스팸 탐지"]:
        for model in MODELS:
            mn = model["name"]
            base = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "baseline"), None)
            sys_r = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "system_prompt"), None)
            fs_r = next((r for r in all_results if r["task"] == task and r["model"] == mn and r["strategy"] == "few_shot"), None)
            lines.append(f"| {task} | {mn} | {base['avg_time']:.2f}s | {sys_r['avg_time']:.2f}s | {fs_r['avg_time']:.2f}s |")

    # Conclusion
    lines.append("\n## 결론\n")

    # Find best strategy per task
    for task in ["PII/NER 탐지", "문서 라우팅", "스팸 탐지"]:
        task_results = [r for r in all_results if r["task"] == task]
        best = max(task_results, key=lambda x: x["accuracy"])
        base_best = max((r for r in task_results if r["strategy"] == "baseline"), key=lambda x: x["accuracy"])
        lines.append(f"- **{task}**: 최적 전략 = {STRATEGY_KR[best['strategy']]} ({best['model']}, {best['accuracy']:.1f}%), baseline 대비 +{best['accuracy'] - base_best['accuracy']:.1f}%p")

    report_text = "\n".join(lines) + "\n"

    with open(BASE_DIR / f"PROMPT_EXPERIMENT_{ts}.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(BASE_DIR / "PROMPT_EXPERIMENT_LATEST.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    # Raw JSON
    with open(BASE_DIR / f"prompt_experiment_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "timestamp": ts}, f, ensure_ascii=False, indent=2)

    log(f"Report saved: PROMPT_EXPERIMENT_{ts}.md")
    return report_text, ts

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log("=" * 60)
    log("프롬프트 최적화 Before/After 실험 시작")
    log(f"모델: {[m['name'] for m in MODELS]}")
    log(f"전략: {STRATEGIES}")
    log(f"샘플: 태스크당 {SAMPLES}개")
    log("=" * 60)

    # Load data
    log("데이터 준비...")
    ner_data = load_ner_data()
    doc_data = generate_doc_routing_data()
    spam_data = generate_spam_data()
    log(f"  NER: {len(ner_data)}, DocRoute: {len(doc_data)}, Spam: {len(spam_data)}")

    tasks_config = [
        ("PII/NER 탐지", ner_data, eval_ner),
        ("문서 라우팅", doc_data, eval_doc),
        ("스팸 탐지", spam_data, eval_spam),
    ]

    all_results = []

    for task_name, data, eval_fn in tasks_config:
        log(f"\n{'='*60}")
        log(f"Task: {task_name}")
        log(f"{'='*60}")

        for model in MODELS:
            for strategy in STRATEGIES:
                log(f"\n--- {model['name']} / {STRATEGY_KR[strategy]} ---")

                # Warmup
                log(f"  Warmup {model['tag']}...")
                query_ollama(model["tag"], "안녕")

                correct, total, avg_time = eval_fn(model["tag"], data, strategy, all_results)
                accuracy = 100 * correct / total if total > 0 else 0

                result = {
                    "task": task_name,
                    "model": model["name"],
                    "strategy": strategy,
                    "strategy_kr": STRATEGY_KR[strategy],
                    "accuracy": round(accuracy, 1),
                    "correct": correct,
                    "total": total,
                    "avg_time": round(avg_time, 2),
                }
                all_results.append(result)
                log(f"  ✅ {accuracy:.1f}% ({correct}/{total}) avg={avg_time:.2f}s")

                # Intermediate save
                with open(BASE_DIR / "experiment_intermediate.json", "w", encoding="utf-8") as f:
                    json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)

    log("\n" + "=" * 60)
    log("실험 완료! 리포트 생성...")
    report_text, ts = generate_report(all_results, start_time)
    log(f"총 소요: {(time.time()-start_time)/3600:.1f}시간")
    log("완료!")

if __name__ == "__main__":
    main()
