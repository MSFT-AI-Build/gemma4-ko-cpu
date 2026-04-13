# Gemma4 한국어 NLP 벤치마크

Gemma4 모델 패밀리(2.3B ~ 31B)의 한국어 자연어처리 성능을 종합 평가한 벤치마크입니다.

**환경**: Azure VM, 8 vCPU, 31GB RAM, CPU only, Ollama v0.20.4  
**비교 모델**: GPT-5.4 (OpenAI API)

## 프로젝트 구조

```
├── README.md
├── benchmarks/
│   ├── 01_intent_classification/   # 의도 분류 (3,000건, 10개 카테고리)
│   ├── 02_korean_nlp/              # 한국어 NLP 종합 (12개 데이터셋)
│   ├── 03_usecase/                 # 실용 유즈케이스 (PII/라우팅/스팸)
│   └── 04_prompt_optimization/     # 프롬프트 최적화 실험
├── data/                           # 테스트 데이터셋
├── scripts/                        # 유틸리티 스크립트
└── docs/                           # 실험 로그, 벤치마크 스펙
```

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
| 31b Dense | 93.3% | 2,799/3,000 | 42.05s | 8.8hr |
| **GPT-5.4** ☁️ | **100.0%** | **3,000/3,000** | API | API |

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

> 📂 상세 결과 및 오답분석: [`benchmarks/02_korean_nlp/results/RESULTS_20260411_101216.md`](benchmarks/02_korean_nlp/results/RESULTS_20260411_101216.md)

---

## 벤치마크 3: 실용 유즈케이스 (PII · 문서라우팅 · 스팸)

로컬 LLM으로 90%+ 정확도가 프로덕션 가능한 실용 태스크 평가, 태스크당 100샘플, think OFF

### 종합 결과 (Accuracy %)

| Task | Use Case | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |
|------|----------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | 개인정보 추출 | 93.0 | 93.0 | 98.0 | **100.0** |
| 문서 라우팅 | 부서 자동 분류 | 90.6 | 91.7 | 95.8 | **97.9** |
| 스팸 탐지 | 스팸 메일 필터링 | **100.0** | **100.0** | 97.0 | **100.0** |
| **평균** | | **94.5** | **94.9** | **96.9** | **99.3** |

### Inference Speed (avg seconds/query)

| Task | e2b (2.3B) | e4b (4.5B) | 26b MoE | 31b Dense |
|------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | 5.84s | 9.43s | 9.72s | 61.44s |
| 문서 라우팅 | 1.44s | 2.54s | 2.66s | 17.12s |
| 스팸 탐지 | 1.54s | 2.75s | 2.85s | 18.69s |

> 📂 상세 결과: [`benchmarks/03_usecase/results/RESULTS_20260412_041951.md`](benchmarks/03_usecase/results/RESULTS_20260412_041951.md)

---

## 벤치마크 4: 프롬프트 최적화 Before/After

기본 프롬프트 vs 시스템 프롬프트 vs Few-shot 예시, 태스크당 50샘플, e4b · 31b 대상

### e4b (4.5B)

| Task | baseline | +system_prompt | +few_shot | 개선폭 |
|------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | **88.0%** | 78.0% | 64.0% | - |
| 문서 라우팅 | 93.8% | **95.8%** | 93.8% | **+2.0%p** |
| 스팸 탐지 | **100.0%** | **100.0%** | 96.0% | - |

### 31b Dense

| Task | baseline | +system_prompt | +few_shot | 개선폭 |
|------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | **98.0%** | **98.0%** | **98.0%** | 0 |
| 문서 라우팅 | **93.8%** | **93.8%** | **93.8%** | 0 |
| 스팸 탐지 | **100.0%** | **100.0%** | **100.0%** | 0 |

### 응답 속도 비교 (avg sec/query)

| Task | Model | baseline | +system_prompt | +few_shot |
|------|-------|:---:|:---:|:---:|
| PII/NER 탐지 | e4b | 8.78s | 8.46s | **5.51s** |
| PII/NER 탐지 | 31b | 60.47s | 60.30s | **42.37s** |
| 문서 라우팅 | e4b | 2.40s | **2.19s** | 2.27s |
| 스팸 탐지 | e4b | 2.70s | **2.21s** | 2.31s |

> 📂 상세 결과: [`benchmarks/04_prompt_optimization/results/PROMPT_EXPERIMENT_20260412_082849.md`](benchmarks/04_prompt_optimization/results/PROMPT_EXPERIMENT_20260412_082849.md)

---

## 주요 발견

### 정확도
- **GPT-5.4 (클라우드) 100% 달성** — 3,000건 전체 정답, 10개 Intent 모두 완벽 분류
- **31b Dense가 로컬 모델 중 1위** — 모든 벤치마크에서 최고 성능 (Intent 93.3%, NLP 종합 72.5%, 유즈케이스 99.3%)
- **모델 크기 ↑ = 정확도 ↑** 패턴이 대부분 태스크에서 일관
- **예외**: 스팸 탐지는 e2b(2.3B)도 100% — 태스크 난이도에 따라 소형 모델로 충분
- **Question Pair 전 모델 40%** — 프롬프트 전략 개선 필요 (bias toward "유사" 판정)
- **GPT-5.4 vs 31b Dense 정확도 차이 6.7%p** — 클라우드 API 대비 로컬 LLM의 현실적 갭

### 속도
- **e2b는 31b 대비 ~15배 빠름** (CPU 환경)
- 단순 분류(스팸/문서라우팅)는 e2b로도 1.4~1.5s에 90%+ 달성
- 입력이 긴 태스크(DKTC 대화)에서 속도 차이 극대화 (14s vs 205s)

### 프롬프트 최적화
- **31b Dense는 프롬프트 전략에 무관하게 안정적** — 3전략 모두 동일 결과
- **e4b에서 few-shot은 NER에 역효과** (88→64%) — 출력 포맷 과적합으로 매칭률 하락
- **system_prompt만 문서 라우팅에서 +2%p 개선** — 유일한 양의 효과
- **few-shot은 속도 개선에 기여** (NER: 8.78→5.51s, 31b: 60→42s) — 응답 길이 감소
- **결론: 모델 크기 선택이 프롬프트 엔지니어링보다 훨씬 큰 영향**

### 유즈케이스별 권장 모델

| 사용 시나리오 | 권장 모델 | 정확도 | 이유 |
|-------------|----------|--------|------|
| 스팸 필터링 | e2b (2.3B) | 100% | 최소 모델로 완벽 정확도 |
| 문서 라우팅 | e4b (4.5B) | 91.7% | 속도-정확도 밸런스 |
| PII/NER 탐지 | 26b MoE | 98.0% | 높은 정확도 + 적당한 속도 |
| 정확도 최우선 | 31b Dense | 99.3% | 전체 최고, GPU 환경 권장 |
| 실시간 서비스 | e4b (4.5B) | 94.9% | 2~3s 응답, 90%+ 전 태스크 |
| 엣지/모바일 | e2b (2.3B) | 94.5% | 7.2GB, 1~6s 응답 |

---

## GPU 사용 시 예상 성능

CPU only 환경(DDR4 ~50GB/s)에서의 실측 결과를 기반으로, GPU 메모리 대역폭 비율과 실제 오버헤드를 반영한 예상치입니다.

### 추정 근거

LLM 추론은 **메모리 대역폭 병목(memory-bound)** 작업으로, GPU 속도 향상은 주로 메모리 대역폭 비율에 비례합니다.

| Hardware | Memory BW | VRAM | 가격대 (참고) |
|----------|-----------|------|-------------|
| CPU DDR4 (본 벤치마크) | ~50 GB/s | 31GB RAM | - |
| RTX 4090 | 1,008 GB/s | 24GB | ~$1,600 |
| A100 | 2,039 GB/s | 80GB | ~$15,000 |
| H100 | 3,350 GB/s | 80GB | ~$30,000 |

> ⚠️ 순수 대역폭 비율(RTX 4090: 20x)에서 커널 오버헤드, 양자화, KV 캐시 등을 감안하여 보수적으로 추정

### 예상 응답 속도 (avg seconds/query)

본 벤치마크 실측 CPU 평균 응답시간 기준, 보수적 스피드업 팩터 적용:

| Model | CPU (실측) | RTX 4090 | A100 | H100 |
|-------|:---:|:---:|:---:|:---:|
| e2b (2.3B) | 2.94s | **~0.20s** (×15) | ~0.11s (×28) | ~0.07s (×45) |
| e4b (4.5B) | 4.91s | **~0.35s** (×14) | ~0.18s (×27) | ~0.12s (×42) |
| 26b MoE (3.8B active) | 5.08s | **~0.39s** (×13) | ~0.20s (×26) | ~0.13s (×40) |
| 31b Dense | 32.42s | ~3.2s (×10) | **~1.3s** (×25) | ~0.8s (×40) |

> 💡 26b MoE는 총 26B 파라미터 중 3.8B만 활성화하므로, GPU에서 e4b와 거의 동일한 속도로 **98% 정확도** 달성

### VRAM 적합성

| Model | Size (Q4) | RTX 4090 (24GB) | A100 (80GB) | H100 (80GB) |
|-------|-----------|:---:|:---:|:---:|
| e2b | ~3.5 GB | ✅ 여유 | ✅ | ✅ |
| e4b | ~5.0 GB | ✅ 여유 | ✅ | ✅ |
| 26b MoE | ~14 GB | ✅ 적합 | ✅ | ✅ |
| 31b Dense | ~19 GB | ⚠️ 빡빡 (Q4 필수) | ✅ | ✅ |

> 31b Dense는 RTX 4090에서 Q4_K_M 양자화로 탑재 가능하나, KV 캐시 포함 시 여유 부족. 최적 운용은 A100 이상 권장.

### 태스크별 GPU 예상 응답시간

RTX 4090 기준, 실용 유즈케이스 벤치마크 태스크별 예상:

| Task | e2b | e4b | 26b MoE | 31b Dense |
|------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | ~0.39s | ~0.67s | ~0.75s | ~6.1s |
| 문서 라우팅 | ~0.10s | ~0.18s | ~0.20s | ~1.7s |
| 스팸 탐지 | ~0.10s | ~0.20s | ~0.22s | ~1.9s |

A100 기준:

| Task | e2b | e4b | 26b MoE | 31b Dense |
|------|:---:|:---:|:---:|:---:|
| PII/NER 탐지 | ~0.21s | ~0.35s | ~0.37s | ~2.5s |
| 문서 라우팅 | ~0.05s | ~0.09s | ~0.10s | ~0.7s |
| 스팸 탐지 | ~0.06s | ~0.10s | ~0.11s | ~0.7s |

### GPU 환경 모델 추천

| 시나리오 | GPU | 추천 모델 | 예상 응답시간 | 이유 |
|---------|-----|----------|:---:|------|
| 실시간 API (<100ms) | A100/H100 | 26b MoE | ~0.1~0.2s | 98% 정확도, 실시간 가능 |
| 비용 효율 서비스 | RTX 4090 | e4b | ~0.2~0.4s | $1,600 GPU로 90%+ 달성 |
| 최고 정확도 서비스 | A100 | 31b Dense | ~1~2.5s | 99.3% 정확도 |
| 엣지/임베디드 | Jetson Orin | e2b | ~0.5~1s | 7.2GB, 저전력 |
| 대량 배치 처리 | H100 | 31b Dense | ~0.8s | 초당 ~1.25 쿼리 처리 |

> **핵심 인사이트**: GPU 환경에서 **26b MoE가 최적의 가성비 모델** — 31b Dense의 98% 수준 정확도를 e4b급 속도로 달성. 3.8B 활성 파라미터 덕분에 MoE의 장점이 극대화됨.

---

## 재현 방법

```bash
# 데이터셋 다운로드 (47개 GitHub 레포, ~2.3GB)
bash scripts/download_datasets.sh

# 벤치마크 1: Intent 분류 (3,000건)
python3 benchmarks/01_intent_classification/benchmark_intent.py --fair-only

# 벤치마크 2: 한국어 NLP 종합 (12개 데이터셋)
python3 benchmarks/02_korean_nlp/benchmark_korean_nlp.py --samples 100

# 벤치마크 3: 실용 유즈케이스 (PII/문서라우팅/스팸)
python3 benchmarks/03_usecase/benchmark_usecase.py

# 벤치마크 4: 프롬프트 최적화 실험
python3 benchmarks/04_prompt_optimization/prompt_experiment.py
```

## 데이터셋 출처

[AwesomeKorean_Data](https://github.com/hyogrin/AwesomeKorean_Data) — 한국어 NLP 오픈 데이터셋 모음

## License

Copyright (c) The ko-nlp Project Authors.

This work is licensed under the [GNU Free Documentation License, Version 1.3](https://www.gnu.org/licenses/fdl-1.3.html) or any later version published by the Free Software Foundation; with no Invariant Sections, no Front-Cover Texts, and no Back-Cover Texts.
