#!/usr/bin/env python3
"""
한국어 NLP 종합 벤치마크 - Gemma4 모델 크기별 비교
AwesomeKorean_Data 기반 다중 데이터셋 평가

사용법:
  python3 benchmark_korean_nlp.py              # 전체 (데이터셋당 100샘플)
  python3 benchmark_korean_nlp.py --samples 50 # 데이터셋당 50샘플
  python3 benchmark_korean_nlp.py --dataset nsmc,kornli  # 특정 데이터셋만
"""

import json
import csv
import time
import sys
import os
import random
import threading
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

OLLAMA_CHAT_API = "http://localhost:11434/api/chat"
PARALLEL_WORKERS = 4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "awesomekorean_data")
LOG_FILE = os.path.join(BASE_DIR, "benchmark_run.log")
PROGRESS_FILE = os.path.join(BASE_DIR, "benchmark_progress.md")

MODELS = [
    "gemma4:e2b",
    "gemma4:latest",
    "gemma4:26b",
    "gemma4:31b",
]

MODEL_LABELS = {
    "gemma4:e2b": "e2b (2.3B)",
    "gemma4:latest": "e4b (4.5B)",
    "gemma4:26b": "26b MoE",
    "gemma4:31b": "31b Dense",
}

_print_lock = threading.Lock()
_log_file = None


def log(msg: str):
    with _print_lock:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if _log_file:
            _log_file.write(line + "\n")
            _log_file.flush()


# ─────────────────────────────────────────────
# Dataset Loaders
# 각 로더는 [(input_text, expected_label), ...] 반환
# ─────────────────────────────────────────────

def load_nsmc(n):
    """NSMC 감성분석 (긍정/부정)"""
    path = os.path.join(DATA_DIR, "nsmc", "ratings_test.txt")
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[1].strip():
                label = "긍정" if parts[2] == "1" else "부정"
                items.append((parts[1].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_kornli(n):
    """KorNLI 자연어추론 (함의/중립/모순)"""
    path = os.path.join(DATA_DIR, "kor-nlu-datasets", "KorNLI", "xnli.test.ko.tsv")
    label_map = {"entailment": "함의", "neutral": "중립", "contradiction": "모순"}
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[2] in label_map:
                text = f"전제: {parts[0]}\n가설: {parts[1]}"
                items.append((text, label_map[parts[2]]))
    return random.sample(items, min(n, len(items)))


def load_korsts(n):
    """KorSTS 문장 유사도 (높음/중간/낮음)"""
    path = os.path.join(DATA_DIR, "kor-nlu-datasets", "KorSTS", "sts-test.tsv")
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                score = float(parts[4])
                if score >= 3.5:
                    label = "높음"
                elif score >= 1.5:
                    label = "중간"
                else:
                    label = "낮음"
                text = f"문장1: {parts[5]}\n문장2: {parts[6]}"
                items.append((text, label))
    return random.sample(items, min(n, len(items)))


def load_question_pair(n):
    """Question Pair 유사질문 판별 (유사/비유사)"""
    path = os.path.join(DATA_DIR, "Question_pair", "kor_Pair_test.csv")
    items = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                label = "유사" if row[3].strip() == "1" else "비유사"
                text = f"질문1: {row[1].strip()}\n질문2: {row[2].strip()}"
                items.append((text, label))
    return random.sample(items, min(n, len(items)))


def load_hate_speech(n):
    """Korean Hate Speech 혐오표현 (혐오/공격/일반)"""
    path = os.path.join(DATA_DIR, "korean-hate-speech", "labeled", "dev.tsv")
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                label_map = {"hate": "혐오", "offensive": "공격", "none": "일반"}
                label = label_map.get(parts[3], parts[3])
                items.append((parts[0].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_kmhas(n):
    """K-MHaS 다중라벨 혐오표현 (주요 카테고리)"""
    label_names = {
        "0": "출신차별", "1": "외모차별", "2": "정치성향차별",
        "3": "욕설", "4": "연령차별", "5": "성차별",
        "6": "인종차별", "7": "종교차별", "8": "해당없음"
    }
    path = os.path.join(DATA_DIR, "K-MHaS", "data", "kmhas_test.txt")
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0].strip():
                labels = parts[1].strip().split(",")
                primary = labels[0].strip()
                label = label_names.get(primary, primary)
                items.append((parts[0].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_dktc(n):
    """DKTC 위협대화 분류 (5종)"""
    path = os.path.join(DATA_DIR, "DKTC", "data", "test.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # data is dict with keys like "t_000", values like {"text": ..., "class": ...}
    # Actually need to check structure - DKTC train.csv has class column
    # test.json may not have labels
    # Use train.csv instead (has labels)
    path2 = os.path.join(DATA_DIR, "DKTC", "data", "train.csv")
    items = []
    with open(path2, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                cls = row[1].strip()
                text = row[2].strip()[:500]  # truncate long conversations
                items.append((text, cls))
    return random.sample(items, min(n, len(items)))


def load_sarcasm(n):
    """Korean Sarcasm 풍자탐지 (풍자/일반)"""
    path = os.path.join(DATA_DIR, "korean-sarcasm", "data", "jiwon", "test.csv")
    items = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                label = "풍자" if row[2].strip() == "1" else "일반"
                text = row[1].strip().replace("@user ", "")
                if text:
                    items.append((text, label))
    return random.sample(items, min(n, len(items)))


def load_apeach(n):
    """APEACH 혐오표현 판별 (혐오/일반)"""
    path = os.path.join(DATA_DIR, "APEACH", "APEACH", "test.csv")
    items = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 5:
                label = "혐오" if row[4].strip() != "Default" else "일반"
                items.append((row[0].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_3i4k(n):
    """3i4k 의도분류 (7종)"""
    label_names = {
        "0": "단편발화", "1": "평서문", "2": "질문",
        "3": "명령문", "4": "수사의문문", "5": "반어문", "6": "감탄문"
    }
    path = os.path.join(DATA_DIR, "3i4k", "data", "train_val_test", "fci_test.txt")
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                label = label_names.get(parts[0], parts[0])
                if parts[1].strip():
                    items.append((parts[1].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_chatbot(n):
    """Chatbot Data 감성분류 (긍정/부정/중립)"""
    label_map = {"0": "일상", "1": "부정", "2": "긍정"}
    path = os.path.join(DATA_DIR, "Chatbot_data", "ChatbotData.csv")
    items = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                label = label_map.get(row[2].strip(), row[2].strip())
                items.append((row[0].strip(), label))
    return random.sample(items, min(n, len(items)))


def load_unsmile(n):
    """Unsmile 혐오표현 (혐오/일반)"""
    path = os.path.join(DATA_DIR, "korean_unsmile_dataset", "unsmile_valid_v1.0.tsv")
    items = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 11:
                is_clean = parts[10].strip() == "1"
                label = "일반" if is_clean else "혐오"
                items.append((parts[0].strip(), label))
    return random.sample(items, min(n, len(items)))


# ─────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────

DATASETS = {
    "nsmc": {
        "name": "NSMC",
        "task": "감성분석",
        "labels": ["긍정", "부정"],
        "loader": load_nsmc,
        "metric": "Accuracy",
        "system_prompt": "당신은 한국어 감성분석기입니다. 주어진 영화 리뷰를 읽고 감성을 분류하세요.\n\n가능한 분류: 긍정, 부정\n\n반드시 분류명만 출력하세요.",
    },
    "kornli": {
        "name": "KorNLI",
        "task": "자연어추론",
        "labels": ["함의", "중립", "모순"],
        "loader": load_kornli,
        "metric": "Accuracy",
        "system_prompt": "당신은 자연어추론(NLI) 분류기입니다. 전제와 가설의 관계를 분류하세요.\n\n가능한 분류: 함의, 중립, 모순\n- 함의: 전제가 가설을 뒷받침\n- 모순: 전제와 가설이 모순\n- 중립: 판단 불가\n\n반드시 분류명만 출력하세요.",
    },
    "korsts": {
        "name": "KorSTS",
        "task": "문장유사도",
        "labels": ["높음", "중간", "낮음"],
        "loader": load_korsts,
        "metric": "Accuracy",
        "system_prompt": "당신은 문장 유사도 판별기입니다. 두 문장의 의미적 유사도를 분류하세요.\n\n가능한 분류: 높음, 중간, 낮음\n- 높음: 거의 같은 의미\n- 중간: 부분적으로 유사\n- 낮음: 관련 없음\n\n반드시 분류명만 출력하세요.",
    },
    "question_pair": {
        "name": "Question Pair",
        "task": "유사문장판별",
        "labels": ["유사", "비유사"],
        "loader": load_question_pair,
        "metric": "Accuracy",
        "system_prompt": "당신은 질문 유사도 판별기입니다. 두 질문이 같은 의미인지 판별하세요.\n\n가능한 분류: 유사, 비유사\n\n반드시 분류명만 출력하세요.",
    },
    "hate_speech": {
        "name": "Hate Speech",
        "task": "혐오표현탐지",
        "labels": ["혐오", "공격", "일반"],
        "loader": load_hate_speech,
        "metric": "Accuracy",
        "system_prompt": "당신은 혐오표현 탐지기입니다. 주어진 댓글을 분류하세요.\n\n가능한 분류: 혐오, 공격, 일반\n- 혐오: 특정 집단에 대한 혐오\n- 공격: 공격적이지만 혐오가 아닌 표현\n- 일반: 일반적인 표현\n\n반드시 분류명만 출력하세요.",
    },
    "kmhas": {
        "name": "K-MHaS",
        "task": "혐오유형분류",
        "labels": ["출신차별", "외모차별", "정치성향차별", "욕설", "연령차별", "성차별", "인종차별", "종교차별", "해당없음"],
        "loader": load_kmhas,
        "metric": "Accuracy",
        "system_prompt": "당신은 혐오표현 유형 분류기입니다. 주어진 텍스트의 혐오 유형을 분류하세요.\n\n가능한 분류: 출신차별, 외모차별, 정치성향차별, 욕설, 연령차별, 성차별, 인종차별, 종교차별, 해당없음\n\n반드시 분류명만 출력하세요.",
    },
    "dktc": {
        "name": "DKTC",
        "task": "위협대화분류",
        "labels": ["협박 대화", "갈취 대화", "직장 내 괴롭힘 대화", "기타 괴롭힘 대화"],
        "loader": load_dktc,
        "metric": "Accuracy",
        "system_prompt": "당신은 위협 대화 분류기입니다. 주어진 대화의 유형을 분류하세요.\n\n가능한 분류: 협박 대화, 갈취 대화, 직장 내 괴롭힘 대화, 기타 괴롭힘 대화\n\n반드시 분류명만 출력하세요.",
    },
    "sarcasm": {
        "name": "Kocasm",
        "task": "풍자탐지",
        "labels": ["풍자", "일반"],
        "loader": load_sarcasm,
        "metric": "Accuracy",
        "system_prompt": "당신은 풍자/비꼼 탐지기입니다. 주어진 텍스트가 풍자인지 판별하세요.\n\n가능한 분류: 풍자, 일반\n\n반드시 분류명만 출력하세요.",
    },
    "apeach": {
        "name": "APEACH",
        "task": "혐오표현판별",
        "labels": ["혐오", "일반"],
        "loader": load_apeach,
        "metric": "Accuracy",
        "system_prompt": "당신은 혐오표현 판별기입니다. 주어진 텍스트에 혐오표현이 포함되어 있는지 판별하세요.\n\n가능한 분류: 혐오, 일반\n\n반드시 분류명만 출력하세요.",
    },
    "3i4k": {
        "name": "3i4K",
        "task": "의도분류",
        "labels": ["단편발화", "평서문", "질문", "명령문", "수사의문문", "반어문", "감탄문"],
        "loader": load_3i4k,
        "metric": "Accuracy",
        "system_prompt": "당신은 한국어 발화 의도 분류기입니다. 주어진 발화의 의도를 분류하세요.\n\n가능한 분류: 단편발화, 평서문, 질문, 명령문, 수사의문문, 반어문, 감탄문\n\n반드시 분류명만 출력하세요.",
    },
    "chatbot": {
        "name": "Chatbot",
        "task": "감성분류",
        "labels": ["일상", "부정", "긍정"],
        "loader": load_chatbot,
        "metric": "Accuracy",
        "system_prompt": "당신은 한국어 감성 분류기입니다. 주어진 문장의 감성을 분류하세요.\n\n가능한 분류: 일상, 부정, 긍정\n- 일상: 특별한 감정 없는 일반 대화\n- 부정: 슬픔, 분노, 걱정 등 부정적 감정\n- 긍정: 기쁨, 감사, 만족 등 긍정적 감정\n\n반드시 분류명만 출력하세요.",
    },
    "unsmile": {
        "name": "Unsmile",
        "task": "혐오판별",
        "labels": ["혐오", "일반"],
        "loader": load_unsmile,
        "metric": "Accuracy",
        "system_prompt": "당신은 혐오표현 판별기입니다. 주어진 텍스트에 혐오표현이 포함되어 있는지 판별하세요.\n\n가능한 분류: 혐오, 일반\n\n반드시 분류명만 출력하세요.",
    },
}


# ─────────────────────────────────────────────
# Core Engine
# ─────────────────────────────────────────────

def query_model(model, user_message, system_prompt, num_predict=50):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_CHAT_API, json=payload, timeout=600)
        elapsed = time.time() - start
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        return {"answer": answer, "elapsed": elapsed, "error": None}
    except Exception as e:
        return {"answer": "", "elapsed": time.time() - start, "error": str(e)}


def normalize_label(raw, valid_labels):
    raw = raw.strip().strip('"').strip("'").strip()
    for label in valid_labels:
        if label in raw:
            return label
    return raw[:30]


def run_single(args):
    idx, text, expected, model, system_prompt, valid_labels = args
    result = query_model(model, text, system_prompt)
    predicted = normalize_label(result["answer"], valid_labels)
    return {
        "index": idx,
        "input": text[:200],
        "expected": expected,
        "predicted": predicted,
        "raw": result["answer"],
        "correct": predicted == expected,
        "time": result["elapsed"],
        "error": result["error"],
    }


def warmup_model(model):
    log(f"  ⏳ [{MODEL_LABELS[model]}] 모델 로딩 중...")
    t = time.time()
    query_model(model, "테스트", "테스트", num_predict=5)
    log(f"  ⏳ [{MODEL_LABELS[model]}] 로딩 완료 ({time.time()-t:.1f}s)")


def run_dataset_model(dataset_key, dataset_info, test_cases, model):
    """단일 데이터셋 + 단일 모델 벤치마크"""
    label = MODEL_LABELS[model]
    ds_name = dataset_info["name"]
    system_prompt = dataset_info["system_prompt"]
    valid_labels = dataset_info["labels"]

    log(f"  🔄 [{label}] × [{ds_name}] 시작 ({len(test_cases)}건)")

    tasks = [
        (i, text, exp, model, system_prompt, valid_labels)
        for i, (text, exp) in enumerate(test_cases)
    ]

    details = []
    completed = 0
    correct = 0
    errors = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(run_single, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            details.append(r)
            completed += 1
            if r["correct"]:
                correct += 1
            if r["error"]:
                errors += 1

            # raw 응답 로그
            status = "✓" if r["correct"] else "✗"
            log(f"    [{status}] {ds_name}/{label} #{r['index']} "
                f"exp={r['expected']} pred={r['predicted']} "
                f"raw=\"{r['raw'][:60]}\" {r['time']:.1f}s"
                f"{' ERR:'+r['error'][:30] if r['error'] else ''}")

    total_time = time.time() - start_time
    details.sort(key=lambda x: x["index"])
    accuracy = correct / len(test_cases) * 100 if test_cases else 0
    avg_time = sum(d["time"] for d in details) / len(details) if details else 0

    log(f"  ✅ [{label}] × [{ds_name}] 완료: {accuracy:.1f}% "
        f"({correct}/{len(test_cases)}), avg {avg_time:.2f}s, wall {total_time:.0f}s")

    return {
        "model": model,
        "model_label": label,
        "dataset": dataset_key,
        "dataset_name": ds_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "avg_time": avg_time,
        "wall_time": total_time,
        "errors": errors,
        "details": details,
    }


# ─────────────────────────────────────────────
# Progress & Reporting
# ─────────────────────────────────────────────

def update_progress(completed_results, current_info=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# 벤치마크 실시간 진행 현황", "", f"**마지막 업데이트**: {now}", ""]

    # Group by dataset
    ds_results = {}
    for r in completed_results:
        ds = r["dataset_name"]
        if ds not in ds_results:
            ds_results[ds] = {}
        ds_results[ds][r["model_label"]] = f"{r['accuracy']:.1f}%"

    if ds_results:
        lines.append("| Dataset | " + " | ".join(MODEL_LABELS.values()) + " |")
        lines.append("|---------|" + "|".join(["------"] * len(MODEL_LABELS)) + "|")
        for ds, models in ds_results.items():
            row = f"| {ds} |"
            for ml in MODEL_LABELS.values():
                row += f" {models.get(ml, '-')} |"
            lines.append(row)
        lines.append("")

    if current_info:
        lines.append(f"🔄 **진행 중**: {current_info}")

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_google_style_report(all_results, samples_per_ds):
    """Google 벤치마크 스타일 리포트"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []

    lines.append("# Korean NLP Benchmark Results")
    lines.append("")
    lines.append(f"**Date**: {now}")
    lines.append(f"**Samples per dataset**: {samples_per_ds}")
    lines.append(f"**Inference**: CPU only, think OFF, temperature 0")
    lines.append(f"**Workers**: {PARALLEL_WORKERS} parallel")
    lines.append("")

    # Model info
    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Parameters | Size | Active Params |")
    lines.append("|-------|-----------|------|---------------|")
    lines.append("| gemma4:e2b | 2.3B | 7.2 GB | 2.3B |")
    lines.append("| gemma4:e4b | 4.5B | 9.6 GB | 4.5B |")
    lines.append("| gemma4:26b | 26B (MoE) | 17 GB | 3.8B |")
    lines.append("| gemma4:31b | 31B (Dense) | 19 GB | 31B |")
    lines.append("")

    # Main results table
    lines.append("## Results (Accuracy %)")
    lines.append("")

    header = "| Benchmark | Task |"
    sep = "|-----------|------|"
    for ml in MODEL_LABELS.values():
        header += f" {ml} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # Organize by dataset
    ds_map = {}
    for r in all_results:
        key = r["dataset"]
        if key not in ds_map:
            ds_map[key] = {}
        ds_map[key][r["model"]] = r

    avg_scores = {m: [] for m in MODELS}

    for ds_key, ds_info in DATASETS.items():
        if ds_key not in ds_map:
            continue
        row = f"| {ds_info['name']} | {ds_info['task']} |"
        model_accs = {}
        for model in MODELS:
            if model in ds_map[ds_key]:
                acc = ds_map[ds_key][model]["accuracy"]
                model_accs[model] = acc
                avg_scores[model].append(acc)

        # Bold the best
        best = max(model_accs.values()) if model_accs else 0
        for model in MODELS:
            if model in model_accs:
                acc = model_accs[model]
                val = f"**{acc:.1f}**" if acc == best and best > 0 else f"{acc:.1f}"
                row += f" {val} |"
            else:
                row += " - |"
        lines.append(row)

    # Average row
    row = "| **Average** | **전체** |"
    avgs = {}
    for model in MODELS:
        if avg_scores[model]:
            avg = sum(avg_scores[model]) / len(avg_scores[model])
            avgs[model] = avg
    best_avg = max(avgs.values()) if avgs else 0
    for model in MODELS:
        if model in avgs:
            val = f"**{avgs[model]:.1f}**" if avgs[model] == best_avg else f"{avgs[model]:.1f}"
            row += f" {val} |"
        else:
            row += " - |"
    lines.append(row)
    lines.append("")

    # Speed comparison
    lines.append("## Inference Speed (avg seconds/query)")
    lines.append("")
    header2 = "| Benchmark |"
    sep2 = "|-----------|"
    for ml in MODEL_LABELS.values():
        header2 += f" {ml} |"
        sep2 += "------|"
    lines.append(header2)
    lines.append(sep2)

    for ds_key, ds_info in DATASETS.items():
        if ds_key not in ds_map:
            continue
        row = f"| {ds_info['name']} |"
        for model in MODELS:
            if model in ds_map[ds_key]:
                t = ds_map[ds_key][model]["avg_time"]
                row += f" {t:.2f}s |"
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # Error analysis summary
    lines.append("## Error Analysis (Top Confusions per Model)")
    lines.append("")
    for model in MODELS:
        ml = MODEL_LABELS[model]
        lines.append(f"### {ml}")
        lines.append("")
        all_errors = []
        for r in all_results:
            if r["model"] == model:
                for d in r["details"]:
                    if not d["correct"]:
                        all_errors.append((r["dataset_name"], d["expected"], d["predicted"]))
        if not all_errors:
            lines.append("오답 없음 🎉\n")
            continue
        confusion = Counter(all_errors)
        lines.append("| Dataset | Expected → Predicted | Count |")
        lines.append("|---------|---------------------|-------|")
        for (ds, exp, pred), cnt in confusion.most_common(10):
            lines.append(f"| {ds} | {exp} → {pred} | {cnt} |")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    global _log_file

    args = sys.argv[1:]
    samples = 100
    selected_datasets = None

    i = 0
    while i < len(args):
        if args[i] == "--samples" and i + 1 < len(args):
            samples = int(args[i + 1])
            i += 2
        elif args[i] == "--dataset" and i + 1 < len(args):
            selected_datasets = args[i + 1].split(",")
            i += 2
        else:
            i += 1

    datasets_to_run = {k: v for k, v in DATASETS.items()
                       if selected_datasets is None or k in selected_datasets}

    _log_file = open(LOG_FILE, "w", encoding="utf-8")

    log("=" * 70)
    log("  Korean NLP Comprehensive Benchmark - Gemma4")
    log(f"  데이터셋: {len(datasets_to_run)}개 | 샘플: {samples}/데이터셋")
    log(f"  모델: {len(MODELS)}개 | think OFF | workers: {PARALLEL_WORKERS}")
    log(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # Seed for reproducibility
    random.seed(42)

    # Load all datasets
    log("\n📂 데이터셋 로딩...")
    loaded_data = {}
    for ds_key, ds_info in datasets_to_run.items():
        try:
            data = ds_info["loader"](samples)
            loaded_data[ds_key] = data
            labels = Counter(l for _, l in data)
            log(f"  ✅ {ds_info['name']}: {len(data)}건 로드 | 라벨분포: {dict(labels)}")
        except Exception as e:
            log(f"  ❌ {ds_info['name']}: 로드 실패 - {e}")

    all_results = []

    for model in MODELS:
        log(f"\n{'='*70}")
        log(f"  🏁 모델: {MODEL_LABELS[model]}")
        log(f"{'='*70}")
        warmup_model(model)

        for ds_key, test_cases in loaded_data.items():
            ds_info = datasets_to_run[ds_key]
            update_progress(all_results,
                            f"[{MODEL_LABELS[model]}] × [{ds_info['name']}]")
            result = run_dataset_model(ds_key, ds_info, test_cases, model)
            all_results.append(result)

    # Final report
    log(f"\n{'='*70}")
    log("  📝 리포트 생성 중...")
    log(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw JSON
    raw_path = os.path.join(BASE_DIR, f"results_raw_{timestamp}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    log(f"💾 Raw JSON: {raw_path}")

    # Google-style markdown report
    report = generate_google_style_report(all_results, samples)
    md_path = os.path.join(BASE_DIR, f"RESULTS_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"📝 Report: {md_path}")

    # Symlink latest
    for name, src in [("RESULTS_LATEST.md", md_path), ("results_latest.json", raw_path)]:
        dst = os.path.join(BASE_DIR, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    update_progress(all_results)
    log(f"\n🔗 최신 결과: {md_path}")
    log("✅ 전체 벤치마크 완료!")

    _log_file.close()


if __name__ == "__main__":
    main()
