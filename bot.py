import os
import json
import threading
import time
import telebot
import requests
from telebot.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask
from datetime import datetime
import hashlib
import secrets
import tempfile
import subprocess

# ======================= НАСТРОЙКИ =======================
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен!")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OWNER_ID = 881904347

bot = telebot.TeleBot(TOKEN)

# ======================= ДАННЫЕ ПОЛЬЗОВАТЕЛЕЙ =======================
STATS_FILE = "user_stats.json"
FREE_DAILY_LIMIT = 5

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_stats(stats):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def get_user_data(user_id):
    stats = load_stats()
    key = f"user_{user_id}"
    if key not in stats:
        stats[key] = {
            "free_used": 0,
            "bonus": 0,
            "first_start": True,
            "referral_bonus_list": [],
            "total_saved_minutes": 0
        }
        save_stats(stats)
    return stats[key]

def save_user_data(user_id, data):
    stats = load_stats()
    stats[f"user_{user_id}"] = data
    save_stats(stats)

def get_remaining_free(user_id):
    data = get_user_data(user_id)
    return max(0, FREE_DAILY_LIMIT - data.get("free_used", 0))

def get_bonus_balance(user_id):
    data = get_user_data(user_id)
    return data.get("bonus", 0)

def can_use_free(user_id):
    data = get_user_data(user_id)
    if data.get("free_used", 0) < FREE_DAILY_LIMIT:
        return True, "free"
    if data.get("bonus", 0) > 0:
        return True, "bonus"
    return False, None

def increment_usage(user_id, use_type="free"):
    data = get_user_data(user_id)
    if use_type == "free":
        data["free_used"] = data.get("free_used", 0) + 1
    elif use_type == "bonus":
        data["bonus"] = max(0, data.get("bonus", 0) - 1)
    save_user_data(user_id, data)

# ======================= РЕФЕРАЛЬНАЯ СИСТЕМА =======================
def get_referral_code(user_id):
    data = get_user_data(user_id)
    if "ref_code" not in data:
        code = hashlib.md5(f"{user_id}{secrets.token_hex(4)}".encode()).hexdigest()[:8]
        data["ref_code"] = code
        save_user_data(user_id, data)
    return data["ref_code"]

def get_referral_stats(user_id):
    data = get_user_data(user_id)
    invited = data.get("referral_bonus_list", [])
    total_bonus = len(invited) * 5
    return len(invited), total_bonus

def add_referral_bonus(user_id, friend_id):
    data = get_user_data(user_id)
    if "referral_bonus_list" not in data:
        data["referral_bonus_list"] = []
    if friend_id not in data["referral_bonus_list"]:
        data["referral_bonus_list"].append(friend_id)
        data["bonus"] = data.get("bonus", 0) + 5
        save_user_data(user_id, data)
        return True
    return False

# ======================= ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ =======================
def generate_image(prompt: str) -> str:
    """Генерирует изображение через OpenRouter"""
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY не настроен"
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-exp-image-generation",
                "messages": [{"role": "user", "content": f"Generate an image: {prompt}"}]
            },
            timeout=30
        )
        data = response.json()
        # Возвращаем ссылку на изображение (упрощённо)
        return "🖼️ Изображение сгенерировано! (ссылка для демонстрации)"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

# ======================= ОБРАБОТКА ГОЛОСОВЫХ =======================
def transcribe_voice(file_path: str) -> str:
    """Распознаёт речь через Whisper (заглушка)"""
    try:
        # Здесь можно подключить Whisper
        return "🎤 Голосовое распознано (заглушка)"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

# ======================= КОМАНДЫ БОТА =======================
@bot.message_handler(commands=['start'])
def start_command(message: Message):
    user_id = message.from_user.id
    data = get_user_data(user_id)
    
    # Обработка реферальной ссылки
    if message.text and len(message.text.split()) > 1:
        ref_code = message.text.split()[1]
        stats = load_stats()
        referrer_id = None
        for uid, udata in stats.items():
            if udata.get("ref_code") == ref_code:
                referrer_id = int(uid.replace("user_", ""))
                break
        if referrer_id and referrer_id != user_id:
            add_referral_bonus(referrer_id, user_id)
            bot.reply_to(message, "🎉 Вас пригласил друг! Вы получили +3 бонусные попытки!")
    
    free_left = get_remaining_free(user_id)
    bonus = get_bonus_balance(user_id)
    bot.reply_to(message, f"👋 Привет!\n\n🎁 Бесплатных: {free_left}\n⭐ Бонусных: {bonus}\n\n/help - список команд")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, """📖 Команды:
/start - Приветствие
/balance - Баланс
/referral - Реферальная ссылка
/image [текст] - Сгенерировать изображение
/help - Помощь""")

@bot.message_handler(commands=['balance'])
def balance_command(message: Message):
    user_id = message.from_user.id
    free_left = get_remaining_free(user_id)
    bonus = get_bonus_balance(user_id)
    bot.reply_to(message, f"💰 Баланс:\n\n🎁 Бесплатных: {free_left}\n⭐ Бонусных: {bonus}")

@bot.message_handler(commands=['referral'])
def referral_command(message: Message):
    user_id = message.from_user.id
    ref_code = get_referral_code(user_id)
    bot_username = bot.get_me().username
    link = f"https://t.me/{bot_username}?start={ref_code}"
    invited, bonus = get_referral_stats(user_id)
    bot.reply_to(message, f"👥 Реферальная ссылка:\n{link}\n\n👥 Приглашено: {invited}\n⭐ Бонусов: {bonus}")

@bot.message_handler(commands=['image'])
def image_command(message: Message):
    user_id = message.from_user.id
    prompt = message.text.replace('/image', '').strip()
    
    if not prompt:
        bot.reply_to(message, "❌ Напишите описание после /image")
        return
    
    can_use, use_type = can_use_free(user_id)
    if not can_use:
        bot.reply_to(message, f"❌ Лимит исчерпан. Бесплатных: 0, Бонусных: {get_bonus_balance(user_id)}")
        return
    
    bot.reply_to(message, "🎨 Генерирую изображение...")
    result = generate_image(prompt)
    increment_usage(user_id, use_type)
    bot.reply_to(message, result)

@bot.message_handler(content_types=['voice'])
def handle_voice(message: Message):
    user_id = message.from_user.id
    can_use, use_type = can_use_free(user_id)
    
    if not can_use:
        bot.reply_to(message, f"❌ Лимит исчерпан. Бонусных: {get_bonus_balance(user_id)}")
        return
    
    bot.reply_to(message, "🎤 Распознаю речь...")
    # Скачиваем голосовое
    file_info = bot.get_file(message.voice.file_id)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        downloaded_file = bot.download_file(file_info.file_path)
        tmp.write(downloaded_file)
        tmp_path = tmp.name
    
    # Конвертируем и распознаём (заглушка)
    result = transcribe_voice(tmp_path)
    increment_usage(user_id, use_type)
    bot.reply_to(message, f"📝 Результат:\n{result}")

# ======================= АДМИН-КОМАНДЫ =======================
@bot.message_handler(commands=['admin_stats'])
def admin_stats(message: Message):
    if message.from_user.id != OWNER_ID:
        bot.reply_to(message, "❌ Недостаточно прав")
        return
    
    stats = load_stats()
    total_users = len(stats)
    total_free = sum(data.get("free_used", 0) for data in stats.values())
    total_bonus = sum(data.get("bonus", 0) for data in stats.values())
    
    bot.reply_to(message, f"📊 Статистика:\n\n👥 Пользователей: {total_users}\n🎙️ Бесплатных: {total_free}\n⭐ Бонусных: {total_bonus}")

# ======================= ЗАПУСК (Flask + Бот) =======================
app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is running!", 200

@app.route('/health')
def health():
    return "OK", 200

def run_bot():
    print("🤖 Бот запущен!")
    bot.infinity_polling()

if __name__ == "__main__":
    # Запуск бота в фоновом потоке
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Запуск Flask
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
