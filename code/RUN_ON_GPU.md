# tideguard — GPU 재실행 안내 (EM&S 리비전)

두 스크립트를 **tideguard 프로젝트 루트**(`wbe_forecast.py`, `chronos_run.py`, `finetune_timesfm.py`가 있는 폴더)에 복사한 뒤,
기존에 TimesFM/Chronos를 돌리던 그 conda 환경(예: `tideguard`)에서 실행하세요.

```bash
# 0) 위치 확인 — 아래 파일들이 같은 폴더에 있어야 함
ls wbe_forecast.py chronos_run.py finetune_timesfm.py pathogen_extension.py lora_sweep.py

conda activate tideguard      # TimesFM/Chronos 돌리던 그 환경
```

---

## 1. 독립 병원체 확장 (리뷰어 2-2 대응) — `pathogen_extension.py`

Swiss 데이터에 이미 있는 **Influenza B (10개 도시)·RSV (8개 도시)·Influenza A (4개 도시, 탐색용)**를
SARS-CoV-2와 **똑같은** rolling-origin 프로토콜·같은 11개 모델 집합으로 예보합니다.

```bash
# (a) 스모크 테스트 — 5분 내, 파이프라인이 도는지 확인
python pathogen_extension.py --smoke

# (b) 정식 실행 — 병원체 3종 x 이력 x 결측 x 시드 (GPU로 대략 수십 분~1시간)
python pathogen_extension.py
```

**생성물 (data/):**
- `raw_results_pathogens.csv` — (pathogen, city, model, horizon, history, missing, seed, mae)
- `alert_perorigin_pathogens.csv` — 병원체별 surge 조기경보용 per-origin 예측
- `pathogens_summary.json` — 병원체별 평균 MAE + site-clustered TimesFM-vs-baseline 통계

---

## 2. LoRA 스윕 (리뷰어 1-4 대응) — `lora_sweep.py`

제출본은 LoRA 설정 1개(rank4/α8/lr1e-4/10ep)만 시험 → "fine-tuning 전반이 무용"이라 말할 수 없다는 지적.
여러 설정을 같은 백엔드로 재학습·평가해 **어떤 설정이라도 zero-shot을 이기는지**를 봅니다.

```bash
# (a) 스모크 — 2개 설정, 2개 사이트, 짧은 epoch
python lora_sweep.py --smoke

# (b) 기본 그리드 — 6개 설정 (rank 4/8/16 x lr 5e-5/1e-4/3e-4 일부)
python lora_sweep.py

# (c) 전체 그리드 — rank{2,4,8,16} x lr{5e-5,1e-4,3e-4} x epoch{5,10} = 24개 (오래 걸림)
python lora_sweep.py --full
```
> peft 필요 시: `pip install peft accelerate transformers`

**생성물 (data/):**
- `lora_sweep.csv` — 설정별 `mae_zs, mae_ft, delta, better(=ft가 더 좋은지)`
- `raw_results_ft_<config>.csv` — 설정별 상세

---

## 끝나면 나에게 보내줄 파일

아래만 이 폴더(또는 채팅으로 업로드)에 돌려주시면 제가 그림·표·본문에 반영합니다:

1. `data/raw_results_pathogens.csv`
2. `data/alert_perorigin_pathogens.csv`
3. `data/pathogens_summary.json`
4. `data/lora_sweep.csv`

문제가 나면 콘솔 로그(마지막 30줄)도 같이 주세요. 스모크가 통과하면 정식 실행은 대개 문제없이 돕니다.
