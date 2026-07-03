import os
import json
import threading
import telebot
import requests
from telebot.types import Message
from http.server import HTTPServer, BaseHTTPRequestHandler
import hashlib
import secrets
import tempfile
import re

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

# ======================= ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ (С ВОЗВРАТОМ URL) =======================
def generate_image_url(prompt: str) -> str:
    """Генерирует изображение и возвращает URL картинки"""
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY не настроен."
    
    try:
        # Пробуем Gemini 2.5 Flash Image
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://t.me/MAB_GatewayBot",
                "X-Title": "MAB Gateway Bot"
            },
            json={
                "model": "google/gemini-2.5-flash-image",
                "messages": [
                    {"role": "user", "content": f"Generate an image: {prompt}. Return ONLY the image URL."}
                ],
                "modalities": ["image", "text"]
            },
            timeout=60
        )
        
        # Если Gemini не сработала, пробуем DALL-E 3
        if response.status_code != 200:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://t.me/MAB_GatewayBot",
                    "X-Title": "MAB Gateway Bot"
                },
                json={
                    "model": "openai/dall-e-3",
                    "messages": [
                        {"role": "user", "content": f"Generate an image: {prompt}"}
                    ]
                },
                timeout=60
            )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                
                # Ищем ссылку на изображение
                url_pattern = r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|webp|svg)'
                urls = re.findall(url_pattern, content, re.IGNORECASE)
                
                # Если не нашли по расширению, ищем любую ссылку
                if not urls:
                    url_pattern = r'https?://[^\s<>"]+'
                    urls = re.findall(url_pattern, content, re.IGNORECASE)
                
                if urls:
                    return urls[0]  # Первая найденная ссылка
                else:
                    return f"❌ Ссылка не найдена. Ответ: {content[:200]}..."
            else:
                return f"❌ Неожиданный ответ: {data}"
        else:
            return f"❌ Ошибка API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

# ======================= КОМАНДЫ БОТА =======================
@bot.message_handler(commands=['start'])
def start_command(message: Message):
    user_id = message.from_user.id
    
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
        f"👋 Привет! Я MAB Gateway.\n\n"
        f"🎁 Бесплатных: {free_left}\n"
        f"⭐ Бонусных: {bonus}\n\n"
        f"📖 Команды:\n"
        f"/image [описание] - генерация\n"
        f"/balance - баланс\n"
        f"/referral - рефералка")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, 
        "📖 Команды:\n"
        "/start - Главное меню\n"
        "/help - Справка\n"
        "/balance - Баланс\n"
        "/referral - Реферальная ссылка\n"
        "/image [описание] - Генерация изображения")

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
        f"📊 Всего: {total}")

@bot.message_handler(commands=['referral'])
def referral_command(message: Message):
    user_id = message.from_user.id
    ref_code = get_referral_code(user_id)
    bot_username = bot.get_me().username
    link = f"https://t.me/{bot_username}?start={ref_code}"
    invited, bonus = get_referral_stats(user_id)
    bot.reply_to(message, 
        f"👥 Реферальная ссылка:\n{link}\n\n"
        f"👥 Приглашено: {invited}\n"
        f"⭐ Бонусов: {bonus}")

@bot.message_handler(commands=['image'])
def image_command(message: Message):
    user_id = message.from_user.id
    prompt = message.text.replace('/image', '', 1).strip()
    
    if not prompt:
        free_left = get_remaining_free(user_id)
        bonus = get_bonus_balance(user_id)
        bot.reply_to(message, 
            f"❌ Напишите описание после /image\n"
            f"Пример: /image красивый закат\n\n"
            f"🎁 Осталось: {free_left}\n"
            f"⭐ Бонусных: {bonus}")
        return
    
    can_use, use_type = can_use_free(user_id)
    if not can_use:
        bot.reply_to(message, 
            f"❌ Лимит исчерпан!\n"
            f"Бесплатных: 0\n"
            f"Бонусных: {get_bonus_balance(user_id)}\n\n"
            f"💡 /referral - пригласите друга")
        return
    
    status_msg = bot.reply_to(message, "🔄 Генерация изображения... ⏳")
    
    image_url = generate_image_url(prompt)
    
    if image_url.startswith('http'):
        # Отправляем изображение по ссылке
        bot.send_photo(
            chat_id=message.chat.id,
            photo=image_url,
            caption=f"🖼️ По запросу: '{prompt}'"
        )
        # Удаляем сообщение с "Генерация..."
        bot.delete_message(chat_id=message.chat.id, message_id=status_msg.message_id)
    else:
        # Если ошибка, обновляем сообщение с ошибкой
        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            text=image_url
        )
    
    increment_usage(user_id, use_type)

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
        f"📊 Статистика:\n\n"
        f"👥 Пользователей: {total_users}\n"
        f"🎙️ Бесплатных: {total_free}\n"
        f"⭐ Бонусных: {total_bonus}\n"
        f"🖼️ Всего генераций: {total_gen}")

@bot.message_handler(content_types=['voice'])
def handle_voice(message: Message):
    user_id = message.from_user.id
    can_use, use_type = can_use_free(user_id)
    
    if not can_use:
        bot.reply_to(message, f"❌ Лимит исчерпан")
        return
    
    status_msg = bot.reply_to(message, "🎤 Распознаю голосовое... ⏳")
    
    try:
        file_info = bot.get_file(message.voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            downloaded_file = bot.download_file(file_info.file_path)
            tmp.write(downloaded_file)
            tmp_path = tmp.name
        
        # Заглушка для распознавания
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
            text=f"❌ Ошибка: {str(e)}"
        )

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
    
    try:
        bot.remove_webhook()
        print("✅ Webhook удалён")
    except Exception as e:
        print(f"⚠️ Webhook: {e}")
    
    print("🤖 MAB Gateway Bot запущен!")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
