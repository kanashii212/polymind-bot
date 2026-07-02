import os
import json
import threading
import time
import telebot
import requests
from telebot.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import hashlib
import secrets
import tempfile
import subprocess

# ======================= НАСТРОЙКИ =======================
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OWNER_ID = 881904347

if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен!")

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
            "total_generated": 0
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
    data["total_generated"] = data.get("total_generated", 0) + 1
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
        return "❌ OPENROUTER_API_KEY не настроен. Добавьте его в переменные окружения Render."
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://t.me/MAB_GatewayBot",
                "X-Title": "MAB Gateway Bot"
            },
            json={
                "model": "google/gemini-2.0-flash-exp-image-generation",
                "messages": [
                    {"role": "user", "content": f"Generate an image: {prompt}"}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                return f"🖼️ Изображение по запросу: '{prompt}'\n\n{content}"
            else:
                return f"❌ Неожиданный ответ от API: {data}"
        else:
            return f"❌ Ошибка API: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "❌ Превышено время ожидания API. Попробуйте позже."
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
    bot.reply_to(message, 
        f"👋 Привет! Я MAB Gateway — AI-помощник.\n\n"
        f"🎁 Бесплатных попыток: {free_left}\n"
        f"⭐ Бонусных попыток: {bonus}\n\n"
        f"📖 Команды:\n"
        f"/help - список всех команд\n"
        f"/image [описание] - сгенерировать изображение\n"
        f"/balance - баланс\n"
        f"/referral - реферальная ссылка")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, 
        "📖 Список команд:\n\n"
        "/start - Главное меню\n"
        "/help - Эта справка\n"
        "/balance - Баланс попыток\n"
        "/referral - Реферальная ссылка\n"
        "/image [описание] - Генерация изображения\n"
        "/image - Покажет остаток попыток")

@bot.message_handler(commands=['balance'])
def balance_command(message: Message):
    user_id = message.from_user.id
    free_left = get_remaining_free(user_id)
    bonus = get_bonus_balance(user_id)
    total = get_user_data(user_id).get("total_generated", 0)
    bot.reply_to(message, 
        f"💰 Баланс:\n\n"
        f"🎁 Бесплатных: {free_left} из {FREE_DAILY_LIMIT}\n"
        f"⭐ Бонусных: {bonus}\n"
        f"📊 Всего генераций: {total}")

@bot.message_handler(commands=['referral'])
def referral_command(message: Message):
    user_id = message.from_user.id
    ref_code = get_referral_code(user_id)
    bot_username = bot.get_me().username
    link = f"https://t.me/{bot_username}?start={ref_code}"
    invited, bonus = get_referral_stats(user_id)
    bot.reply_to(message, 
        f"👥 Реферальная программа\n\n"
        f"🔗 Ваша ссылка:\n{link}\n\n"
        f"👥 Приглашено друзей: {invited}\n"
        f"⭐ Получено бонусов: {bonus}\n\n"
        f"🎁 За каждого друга вы получаете +5 бонусных попыток!")

@bot.message_handler(commands=['image'])
def image_command(message: Message):
    user_id = message.from_user.id
    prompt = message.text.replace('/image', '', 1).strip()
    
    if not prompt:
        free_left = get_remaining_free(user_id)
        bonus = get_bonus_balance(user_id)
        bot.reply_to(message, 
            f"❌ Напишите описание после /image\n\n"
            f"Пример: /image красивый закат\n\n"
            f"🎁 Осталось попыток: {free_left}\n"
            f"⭐ Бонусных: {bonus}")
        return
    
    # Проверка лимитов
    can_use, use_type = can_use_free(user_id)
    if not can_use:
        bot.reply_to(message, 
            f"❌ Лимит попыток исчерпан!\n\n"
            f"Бесплатных: 0\n"
            f"Бонусных: {get_bonus_balance(user_id)}\n\n"
            f"💡 Пригласите друга: /referral")
        return
    
    # Сразу отвечаем, что начали генерацию
    status_msg = bot.reply_to(message, "🔄 Начинаю генерацию изображения... Подождите немного ⏳")
    
    # Генерируем
    result = generate_image(prompt)
    
    # Увеличиваем счётчик
    increment_usage(user_id, use_type)
    
    # Обновляем сообщение с результатом
    bot.edit_message_text(
        chat_id=message.chat.id,
        message_id=status_msg.message_id,
        text=result
    )

@bot.message_handler(content_types=['voice'])
def handle_voice(message: Message):
    user_id = message.from_user.id
    can_use, use_type = can_use_free(user_id)
    
    if not can_use:
        bot.reply_to(message, f"❌ Лимит исчерпан. Бонусных: {get_bonus_balance(user_id)}")
        return
    
    status_msg = bot.reply_to(message, "🎤 Распознаю голосовое... Подождите ⏳")
    
    try:
        # Скачиваем голосовое
        file_info = bot.get_file(message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            downloaded_file = bot.download_file(file_info.file_path)
            tmp.write(downloaded_file)
            tmp_path = tmp.name
        
        # Здесь можно добавить распознавание через Whisper
        # Пока заглушка
        result = "🎤 Голосовое распознано (функция в разработке)"
        
        increment_usage(user_id, use_type)
        
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text=f"📝 Результат:\n{result}"
        )
        
        os.unlink(tmp_path)
        
    except Exception as e:
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text=f"❌ Ошибка обработки: {str(e)}"
        )

@bot.message_handler(commands=['admin_stats'])
def admin_stats(message: Message):
    if message.from_user.id != OWNER_ID:
        bot.reply_to(message, "❌ Недостаточно прав")
        return
    
    stats = load_stats()
    total_users = len(stats)
    total_free = sum(data.get("free_used", 0) for data in stats.values())
    total_bonus = sum(data.get("bonus", 0) for data in stats.values())
    total_gen = sum(data.get("total_generated", 0) for data in stats.values())
    
    bot.reply_to(message, 
        f"📊 Статистика бота:\n\n"
        f"👥 Пользователей: {total_users}\n"
        f"🎙️ Бесплатных: {total_free}\n"
        f"⭐ Бонусных: {total_bonus}\n"
        f"🖼️ Всего генераций: {total_gen}")

# ======================= HTTP-СЕРВЕР ДЛЯ RENDER =======================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    print(f"🌐 HTTP-сервер запущен на порту {port}")
    server.serve_forever()

# ======================= ЗАПУСК =======================
if __name__ == "__main__":
    http_thread = threading.Thread(target=run_http_server)
    http_thread.daemon = True
    http_thread.start()
    
    print("🤖 MAB Gateway Bot запущен!")
    bot.infinity_polling()
