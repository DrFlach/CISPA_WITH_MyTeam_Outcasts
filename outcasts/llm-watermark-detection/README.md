# LLM Watermark Detection

Бінарна детекція AI-вотермарків (Kirchenbauer / Aaronson / SynthID-стиль) у текстах.

## Метрика конкурсу

**TPR @ FPR = 1%** — True Positive Rate коли False Positive Rate ≤ 0.01.

Це **значно суворіша** метрика за accuracy. Вона вимагає, щоб модель була дуже впевнена у вотермаркованих текстах і ставила їм найвищі скори (топ ~1% за score). Ranking важить більше за калібровку.

Сервер ділить test на 70% public + 30% held-out за MD5-хешем id і рахує метрику окремо.

## Формат сабмішини

```
id,score
1,0.823
2,0.154
...
2250,0.412
```

- header: рівно `id,score`
- 2250 рядків, ids 1..2250 (кожен один раз)
- `score`: float у [0, 1], finite, **що вищий — тим більш вотермарковано**
- Не бінаризуй! TPR@FPR=0.01 рахується через ROC, бінарні значення дають скор 0.

## Структура проекту

```
llm-watermark/
├── README.md
├── requirements.txt
├── main.py                       # CLI: тренування + опційно upload
├── pipeline.py                   # оркестратор
├── upload.py                     # окремий скрипт відправки на сервер
├── dataset/                      # 5 .jsonl файлів
└── src/
    ├── config.py                 # центральні hyperparams
    ├── data.py                   # завантаження jsonl
    ├── cv.py                     # 5-fold stratified, seed=42
    ├── metric.py                 # TPR @ FPR = 1%
    ├── submit.py                 # CSV builder + валідатор
    ├── features/
    │   ├── statistical.py        # 15 CPU фічей
    │   └── llm_logprobs.py       # 24 GPU фічі — головна зброя
    └── models/
        └── lgbm.py               # LightGBM 5-fold + OOF + importance
```

## Швидкий старт

```bash
cd llm-watermark
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1) baseline (CPU, ~1 хв) — переконатись, що pipeline працює
python main.py --mode stat

# 2) головний прогон — LLM logprobs з Llama-3-8B (GPU)
python main.py --mode logprobs

# 3) усе разом + одразу відправити на сервер
python main.py --mode all --upload
```

## Окрема відправка

`main.py --upload` робить тренування + відправку. Якщо хочеш контролювати окремо:

```bash
python upload.py                                       # відправляє останній CSV
python upload.py output/submissions/submission_X.csv   # конкретний файл
python upload.py --dry-run                             # тільки валідація
```

API key і URL хардкоднуті в `upload.py` (взяті з `submission_template.py`). Можна перевизначити через env vars:

```bash
export WM_API_KEY=...
export WM_BASE_URL=...
```

## Що очікувати від кожного режиму

| Режим | OOF AUC | OOF TPR@FPR=1% | Час на твоїй GPU |
|---|---|---|---|
| `stat` | ~0.60 | ~0.00 | <1 хв |
| `logprobs` (Llama-3-8B) | 0.85-0.95 | 0.30-0.70 | 5-15 хв |
| `all` | трохи вище | трохи вище | ~15 хв |

Реальні числа залежать від того, наскільки base LM збігається з моделлю-генератором.

## Кешування

Усі фічі кешуються у `output/features/*.parquet` — повторні запуски пропускають їх обчислення.

## Зміна базової моделі

У `src/config.py` редагуй `BASE_LM_NAME`:
- `meta-llama/Meta-Llama-3-8B` — сильний дефолт
- `mistralai/Mistral-7B-v0.3`
- `Qwen/Qwen2-7B`
- `microsoft/Phi-3-mini-4k-instruct` — швидкий, 3.8B

Що ближче base LM до моделі-генератора текстів — тим сильніший сигнал.

## Виходи

- `output/models/lgbm_<mode>/fold_*.txt` — per-fold LightGBM boosters
- `output/models/lgbm_<mode>/oof.npy` — OOF predictions для майбутнього ансамблювання
- `output/models/lgbm_<mode>/feature_importance.csv` — топ фічі
- `output/submissions/submission_<timestamp>_<mode>_oofTPR<value>.csv` — готова сабмішина
