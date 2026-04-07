# NKTab

## Структура
- `docs/` — сайт для GitHub Pages
- `model_train/` — код твоей нейросети
- `data/train.txt` — текст для обучения
- `checkpoints/` — сюда сохранятся веса модели

## Установка
```bash
pip install -r requirements.txt
```

## Обучение
```bash
python model_train/nktab_model_from_scratch.py train
```

## Генерация
```bash
python model_train/nktab_model_from_scratch.py generate "привет"
```

## Запуск сервера модели
```bash
python model_train/nktab_model_from_scratch.py serve
```

После этого сайт из `docs/` будет отправлять запросы на `http://127.0.0.1:8000/generate`.
