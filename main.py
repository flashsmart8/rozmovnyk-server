"""
Rozmovnyk Server — проксі для AI-діалогів з Claude
Деплоїться на Railway. Приймає запити з flashsmart8.github.io,
викликає Anthropic API, повертає відповідь.
"""
import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# --- Налаштування логування ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Змінні середовища ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://flashsmart8.github.io").split(",")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY не встановлено в змінних середовища")
if not ACCESS_PASSWORD:
    raise RuntimeError("ACCESS_PASSWORD не встановлено в змінних середовища")

# --- Модель Claude ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Sonnet
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 600

# --- FastAPI ---
app = FastAPI(title="Rozmovnyk Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# --- Системні промпти для ролей ---
# Базові інструкції — спільні для всіх ролей
BASE_INSTRUCTIONS = """You are an English conversation partner helping a Ukrainian-speaking learner improve their British English. The learner's goal is to speak like a native British speaker.

CRITICAL RULES:
1. ALWAYS respond in BRITISH English (use 'lift' not 'elevator', 'flat' not 'apartment', 'lovely' instead of 'great', etc.). Use British spelling: colour, centre, realise, organisation.
2. If the learner writes in Ukrainian (because they don't know the English word), understand them but ALWAYS reply in English. Don't translate to Ukrainian. Gently weave the English word into your reply so they learn it from context.
3. If the learner makes a mistake (grammar, word choice, awkward phrasing), gently correct them. Show the corrected version in *italics* like this: *Did you mean: "I'd like to book a table"?*  Then continue the conversation naturally.
4. Keep your replies SHORT (1-3 sentences). This is a conversation, not a lecture. Long replies overwhelm learners.
5. Speak naturally — use contractions (I'm, don't, can't, won't, that's). Use everyday British phrases (cheers, lovely, brilliant, sorted, no worries, sounds good).
6. Stay IN CHARACTER throughout. Don't break the role-play to give grammar lectures unless correcting a specific mistake.
7. Ask questions to keep the conversation going. Don't just answer — engage.
8. NEVER use emojis. NEVER use markdown formatting like bold, headers, or bullet points (except *italics* for corrections).
9. If the learner clearly doesn't understand and asks in Ukrainian what something means, give a very brief English explanation with a simple example. Don't translate to Ukrainian.

EXAMPLE of a correction:
Learner: "I want order pizza"
You: *Did you mean: "I'd like to order a pizza"?* Of course, what kind would you like? We've got margherita, pepperoni, and a lovely four cheese.
"""

ROLE_PROMPTS = {
    "waiter": """You are a friendly waiter at a traditional British pub-restaurant in London. The learner is your customer. You should:
- Greet them, offer a table, present the menu
- Recommend dishes (fish and chips, Sunday roast, shepherd's pie, sticky toffee pudding)
- Take their order, ask about drinks
- Be warm, polite, slightly chatty — typical British hospitality
""",
    "hotel": """You are a hotel receptionist at a mid-range hotel in Bath, England. The learner is a guest. You should:
- Welcome them, handle check-in or enquiries
- Discuss room types, breakfast, Wi-Fi, local attractions
- Help with requests (extra towels, taxi, late check-out)
- Be professional, warm, helpful — classic British hospitality
""",
    "airport": """You are airport staff at Heathrow — could be check-in agent, security, or information desk. The learner is a passenger. You should:
- Help with check-in, baggage, gate information, delays
- Answer questions about flights, terminals, transport
- Be efficient but polite — British professionalism
""",
    "shop": """You are a friendly shop assistant in a clothing shop on a British high street. The learner is a customer. You should:
- Greet them, offer help, suggest items, sizes, colours
- Discuss prices, fittings, returns policy
- Be helpful and chatty — typical British shop assistant
""",
    "colleague": """You are a friendly British colleague at work — let's say you work in an office in Manchester. The learner is your workmate. You should:
- Make small talk: weekend plans, weather, lunch, work projects, holidays
- Use casual British expressions (cheers, no worries, sorted, fancy a coffee?)
- Keep it light and friendly — typical office banter
""",
    "friend": """You are a British friend of the learner. You're chatting casually, like mates. You should:
- Talk about anything: hobbies, weekend, films, music, food, holidays, life
- Be relaxed, use casual British slang (mate, brilliant, knackered, gutted, chuffed)
- Ask about their life, share your own, joke around
- This is the most informal of all the roles — like texting a friend
"""
}

WELCOME_MESSAGES = {
    "waiter": "Good evening! Welcome to The Crown. Table for one, is it? Right this way. Here's the menu — can I get you a drink to start?",
    "hotel": "Good afternoon! Welcome to The Royal Crescent Hotel. Do you have a reservation with us today?",
    "airport": "Hello there, how can I help you?",
    "shop": "Hiya, welcome in! Let me know if you need any help finding anything. Just having a browse, are you?",
    "colleague": "Morning! How was your weekend then? Fancy a coffee before the meeting?",
    "friend": "Hiya! How's it going? What've you been up to?"
}


# --- Pydantic моделі ---
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    role: str  # "waiter", "hotel", etc.
    history: List[Message]
    password: str

class ChatResponse(BaseModel):
    reply: str

class WelcomeRequest(BaseModel):
    role: str
    password: str


# --- Endpoints ---

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "rozmovnyk-server"}


@app.post("/welcome")
async def get_welcome(req: WelcomeRequest):
    """Повертає привітальне повідомлення для обраної ролі."""
    if req.password != ACCESS_PASSWORD:
        raise HTTPException(status_code=401, detail="Невірний пароль")

    if req.role not in WELCOME_MESSAGES:
        raise HTTPException(status_code=400, detail="Невідома роль")

    return {"reply": WELCOME_MESSAGES[req.role]}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Основний endpoint — приймає історію розмови, повертає відповідь AI."""

    # 1. Перевірка пароля
    if req.password != ACCESS_PASSWORD:
        logger.warning(f"Невірний пароль: {req.password[:3]}...")
        raise HTTPException(status_code=401, detail="Невірний пароль")

    # 2. Перевірка ролі
    if req.role not in ROLE_PROMPTS:
        raise HTTPException(status_code=400, detail="Невідома роль")

    # 3. Перевірка історії
    if not req.history:
        raise HTTPException(status_code=400, detail="Порожня історія")

    if len(req.history) > 50:
        # Обмеження довжини розмови — щоб не перевищити токени
        req.history = req.history[-50:]

    # 4. Будуємо системний промпт
    system_prompt = BASE_INSTRUCTIONS + "\n\n" + "ROLE-SPECIFIC INSTRUCTIONS:\n" + ROLE_PROMPTS[req.role]

    # 5. Будуємо повідомлення для Anthropic
    messages = []
    for msg in req.history:
        if msg.role not in ("user", "assistant"):
            continue
        messages.append({"role": msg.role, "content": msg.content})

    # 6. Запит до Anthropic
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt,
        "messages": messages
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(ANTHROPIC_URL, headers=headers, json=body)

        if response.status_code != 200:
            logger.error(f"Anthropic API error {response.status_code}: {response.text[:300]}")
            raise HTTPException(
                status_code=502,
                detail=f"Помилка AI-сервісу: {response.status_code}"
            )

        data = response.json()
        content = data.get("content", [])

        # Витягуємо текст з відповіді
        reply_text = ""
        for block in content:
            if block.get("type") == "text":
                reply_text += block.get("text", "")

        if not reply_text:
            raise HTTPException(status_code=502, detail="Порожня відповідь від AI")

        logger.info(f"Role: {req.role}, msgs: {len(messages)}, reply len: {len(reply_text)}")
        return ChatResponse(reply=reply_text.strip())

    except httpx.TimeoutException:
        logger.error("Timeout при запиті до Anthropic")
        raise HTTPException(status_code=504, detail="Таймаут — спробуйте ще раз")
    except httpx.RequestError as e:
        logger.error(f"Помилка запиту: {e}")
        raise HTTPException(status_code=502, detail="Помилка звʼязку з AI")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
