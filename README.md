# Gemma4 한국어 NLP 벤치마크

Gemma4 모델 패밀리(2.3B ~ 31B)의 한국어 자연어처리 성능을 종합 평가한 벤치마크입니다.

**환경**: Azure VM, 8 vCPU, 31GB RAM, CPU only, Ollama v0.20.4

## 모델 정보

| Model | Parameters | Size | Active Params | Context |
|-------|-----------|------|---------------|---------|
| gemma4:e2b | 2.3B | 7.2 GB | 2.3B | 128K |
| gemma4:e4b | 4.5B | 9.6 GB | 4.5B | 128K |
| gemma4:26b | 26B (MoE) | 17 GB | 3.8B | 256K |
| gemma4:31b | 31B (Dense) | 19 GB | 31B | 256K |

---

## 벤치마크 1: Intent 분류 (3,000건)

고객 서비스 도메인 10개 Intent × 300문장, think OFF, temperature 0

### 종합 결과

| Model | Accuracy | Correct | Avg Latency | Wall Time |
|-------|----------|---------|-------------|-----------|
| e2b (2.3B) | 81.0% | 2,429/3,000 | 2.90s | 36min |
| e4b (4.5B) | 91.4% | 2,743/3,000 | 5.37s | 67min |
| 26b MoE | 90.8% | 2,724/3,000 | 6.64s | 83min |
| **31b Dense** | **93.3%** | **2,799/3,000** | 42.05s | 8.8hr |

> 📂 상세 결과: [`benchmark_intermediate/`](benchmark_intermediate/)

---

## 벤치마크 2: 한국어 NLP 종합 (12개 데이터셋)

[AwesomeKorean_Data](https://github.com/hyogrin/AwesomeKorean_Data) 기반 12개 공개 데이터셋, 데이터셋당 100샘플, think OFF

### Results (Accuracy %)

| Benchmark | Task | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |
|-----------|------|:---:|:---:|:---:|:---:|
| NSMC | 감성분석 | 80.0 | 83.0 | 85.0 | **89.0** |
| KorNLI | 자연어추론 | 55.0 | 68.0 | 78.0 | **84.0** |
| KorSTS | 문장유사도 | 52.0 | 66.0 | 75.0 | **77.0** |
| Question Pair | 유사문장판별 | 40.0 | 40.0 | 40.0 | 40.0 |
| Hate Speech | 혐오표현탐지 | 57.0 | **64.0** | 63.0 | 63.0 |
| K-MHaS | 혐오유형분류 | 60.0 | 70.0 | 72.0 | **76.0** |
| DKTC | 위협대화분류 | 66.0 | 65.0 | 78.0 | **88.0** |
| Kocasm | 풍자탐지 | 53.0 | **63.0** | 52.0 | 59.0 |
| APEACH | 혐오표현판별 | 77.0 | 86.0 | **89.0** | 88.0 |
| 3i4K | 의도분류 | 74.0 | 76.0 | 77.0 | **80.0** |
| Chatbot | 감성분류 | 42.0 | 44.0 | **47.0** | 44.0 |
| Unsmile | 혐오판별 | 66.0 | 77.0 | 78.0 | **82.0** |
| **Average** | | **60.2** | **66.8** | **69.5** | **72.5** |

### Inference Speed (avg seconds/query)

| Benchmark | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |
|-----------|:---:|:---:|:---:|:---:|
| NSMC | 3.22s | 6.14s | 7.18s | 47.82s |
| KorNLI | 6.03s | 11.70s | 12.27s | 87.20s |
| KorSTS | 5.21s | 9.99s | 10.58s | 76.37s |
| Question Pair | 3.45s | 6.44s | 7.38s | 50.47s |
| Hate Speech | 3.75s | 7.17s | 7.95s | 55.63s |
| K-MHaS | 3.81s | 7.42s | 8.35s | 56.15s |
| DKTC | 14.42s | 28.73s | 27.57s | 204.72s |
| Kocasm | 3.52s | 6.65s | 7.43s | 51.28s |
| APEACH | 3.48s | 6.63s | 7.51s | 51.14s |
| 3i4K | 2.59s | 4.80s | 5.83s | 37.70s |
| Chatbot | 2.09s | 3.90s | 4.97s | 32.17s |
| Unsmile | 3.57s | 6.72s | 7.60s | 52.62s |

> 📂 상세 결과 및 오답분석: [`awd_benchmark_01/RESULTS_20260411_101216.md`](awd_benchmark_01/RESULTS_20260411_101216.md)

---

## 주요 발견

### 정확도
- **31b Dense가 전체 1위** — 두 벤치마크 모두에서 최고 성능 (Intent 93.3%, NLP 종합 72.5%)
- **모델 크기 ↑ = 정확도 ↑** 패턴이 대부분 태스크에서 일관
- **예외**: 풍자탐지(Kocasm)는 e4b(63%)가 31b(59%)보다 우수 — 단순 크기가 아닌 태스크 특성 영향
- **Question Pair 전 모델 40%** — 프롬프트 전략 개선 필요 (bias toward "유사" 판정)

### 속도
- **e2b는 31b 대비 ~15배 빠름** (CPU 환경)
- Intent 분류처럼 단순한 태스크에서는 e4b(91.4%, 5.37s)가 비용 대비 최적
- 입력이 긴 태스크(DKTC 대화)에서 속도 차이 극대화 (14s vs 205s)

### 권장 모델
| 사용 시나리오 | 권장 모델 | 이유 |
|-------------|----------|------|
| 실시간 서비스 | e4b (4.5B) | 속도-정확도 밸런스 최적 |
| 정확도 우선 | 31b Dense | 전체 최고 성능 |
| 엣지/모바일 | e2b (2.3B) | 7.2GB, 빠른 응답 |
| GPU 환경 | 31b Dense | GPU 시 ~0.2s/query 예상 |

---

## 재현 방법

```bash
# 데이터셋 다운로드 (47개 GitHub 레포, ~2.3GB)
bash download_datasets.sh

# Intent 분류 벤치마크 (3,000건)
python3 benchmark_intent.py --fair-only

# 한국어 NLP 종합 벤치마크 (12개 데이터셋)
cd awd_benchmark_01
python3 benchmark_korean_nlp.py --samples 100
```

## 데이터셋 출처

[AwesomeKorean_Data](https://github.com/hyogrin/AwesomeKorean_Data) — 한국어 NLP 오픈 데이터셋 모음

## License

MIT
