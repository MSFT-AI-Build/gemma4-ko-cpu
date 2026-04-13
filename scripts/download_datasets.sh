#!/bin/bash
# AwesomeKorean_Data (https://github.com/hyogrin/AwesomeKorean_Data)에서
# GitHub로 공개된 47개 한국어 NLP 데이터셋을 shallow clone합니다.
# 로그인 필요 사이트(KAIST, AIHub, 모두의말뭉치)는 제외.
# 총 약 2.3GB, 소요시간 ~2분.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$SCRIPT_DIR/awesomekorean_data"
LOG="$BASE/download_log.txt"
mkdir -p "$BASE"
echo "=== 다운로드 시작: $(date) ===" | tee "$LOG"

# 메타/문서 레포 제외, 실제 데이터셋 레포만 shallow clone
REPOS=(
  "https://github.com/kmounlp/NER"
  "https://github.com/korean-named-entity/konne"
  "https://github.com/songys/Question_pair"
  "https://github.com/kakaobrain/kor-nlu-datasets"
  "https://github.com/warnikchow/ParaKQC"
  "https://github.com/smilegate-ai/korean_smile_style_dataset"
  "https://github.com/e9t/nsmc"
  "https://github.com/SpellOnYou/korean-sarcasm"
  "https://github.com/kocohub/korean-hate-speech"
  "https://github.com/jason9693/APEACH"
  "https://github.com/smilegate-ai/korean_unsmile_dataset"
  "https://github.com/sgunderscore/hatescore-korean-hate-speech"
  "https://github.com/boychaboy/KOLD"
  "https://github.com/tunib-ai/DKTC"
  "https://github.com/adlnlp/K-MHaS"
  "https://github.com/warnikchow/3i4k"
  "https://github.com/smilegate-ai/HuLiC"
  "https://github.com/smilegate-ai/OPELA"
  "https://github.com/theeluwin/sci-news-sum-kr-50"
  "https://github.com/warnikchow/sae4k"
  "https://github.com/jungyeul/korean-parallel-corpora"
  "https://github.com/muik/transliteration"
  "https://github.com/google-research-datasets/paws"
  "https://github.com/google-research-datasets/tydiqa"
  "https://github.com/HLTCHKUST/Xpersona"
  "https://github.com/naver-ai/carecall-memory"
  "https://github.com/kevinduh/iwslt22-dialect"
  "https://github.com/Kyubyong/kss"
  "https://github.com/goodatlas/zeroth"
  "https://github.com/clovaai/ClovaCall"
  "https://github.com/yc9701/pansori-tedxkr-corpus"
  "https://github.com/warnikchow/prosem"
  "https://github.com/warnikchow/kosp2e"
  "https://github.com/Gyeongmin47/KoCHET-A-Korean-Cultural-Heritage-corpus-for-Entity-related-Tasks"
  "https://github.com/nlpai-lab/KommonGen"
  "https://github.com/lbox-kr/lbox-open"
  "https://github.com/machinereading/K2NLG-Dataset"
  "https://github.com/bareun-nlp/korean-ambiguity-data"
  "https://github.com/kakaobrain/jejueo"
  "https://github.com/soyoung97/Standard_Korean_GEC"
  "https://github.com/emorynlp/ud-korean"
  "https://github.com/openkorpos/openkorpos"
  "https://github.com/akngs/petitions"
  "https://github.com/songys/Chatbot_data"
  "https://github.com/lovit/kmrd"
  "https://github.com/choe-hyonsu-gabrielle/korean-amr-corpus"
  "https://github.com/bab2min/corpus"
)

TOTAL=${#REPOS[@]}
SUCCESS=0
FAIL=0

for i in "${!REPOS[@]}"; do
  URL="${REPOS[$i]}"
  NAME=$(basename "$URL" .git)
  IDX=$((i+1))
  
  if [ -d "$BASE/$NAME" ]; then
    echo "[$IDX/$TOTAL] ⏭ SKIP (exists): $NAME" | tee -a "$LOG"
    SUCCESS=$((SUCCESS+1))
    continue
  fi
  
  echo "[$IDX/$TOTAL] 📥 Cloning: $NAME ..." | tee -a "$LOG"
  if git clone --depth 1 --quiet "$URL" "$BASE/$NAME" 2>>"$LOG"; then
    SIZE=$(du -sh "$BASE/$NAME" | cut -f1)
    echo "[$IDX/$TOTAL] ✅ OK: $NAME ($SIZE)" | tee -a "$LOG"
    SUCCESS=$((SUCCESS+1))
  else
    echo "[$IDX/$TOTAL] ❌ FAIL: $NAME" | tee -a "$LOG"
    FAIL=$((FAIL+1))
  fi
done

echo "" | tee -a "$LOG"
echo "=== 완료: $(date) ===" | tee -a "$LOG"
echo "성공: $SUCCESS / $TOTAL, 실패: $FAIL" | tee -a "$LOG"
TOTAL_SIZE=$(du -sh "$BASE" | cut -f1)
echo "총 크기: $TOTAL_SIZE" | tee -a "$LOG"
