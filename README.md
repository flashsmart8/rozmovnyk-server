# Rozmovnyk Server

Проксі-сервер для AI-діалогів додатку `rozmovnyk-english`.

## Що це
- Приймає запити з `https://flashsmart8.github.io/rozmovnyk-english/`
- Викликає Anthropic Claude API
- Захищений паролем для родинного використання

## Розгортання на Railway

### Крок 1: Створити репо на GitHub
1. Створи новий приватний репо: `rozmovnyk-server`
2. Завантаж усі файли через Upload files: `main.py`, `requirements.txt`, `runtime.txt`, `Procfile`, `README.md`

### Крок 2: Створити проєкт на Railway
1. Зайди на railway.app → New Project → Deploy from GitHub
2. Вибери репозиторій `rozmovnyk-server`
3. Railway автоматично визначить Python і почне будувати

### Крок 3: Додати змінні середовища
В розділі **Variables** додай:

```
ANTHROPIC_API_KEY = sk-ant-... (твій API-ключ)
ACCESS_PASSWORD = твій_пароль (наприклад rozmovnyk2026)
ALLOWED_ORIGINS = https://flashsmart8.github.io
```

### Крок 4: Згенерувати публічний URL
1. Settings → Networking → Generate Domain
2. Скопіюй URL — буде щось типу `rozmovnyk-server-production.up.railway.app`
3. Передай цей URL Claude — він пропише його в `index.html`

### Крок 5: Перевірити що працює
Відкрий у браузері: `https://[твій-url]/`
Має показати: `{"status":"ok","service":"rozmovnyk-server"}`
