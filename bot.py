import os
import json
import telebot
from telebot.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from datetime import datetime

# ======================= НАСТРОЙКИ =======================
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен!")

bot = telebot.TeleBot(TOKEN)

# ======================= ДАННЫЕ ПОЛЬЗОВАТЕЛЕЙ =======================
STATS_FILE = "user_stats.json"
FREE_DAILY_LIMIT = 5
OWNER_ID = 881904347

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
            "legal_accepted": False,
            "referral_bonus_list": [],
            "total_saved_minutes": 0,
            "preferred_format": None
        }
        save_stats(stats)
    return stats[key]

def save_user_data(user_id, data):
    stats = load_stats()
    stats[f"user_{user_id}"] = data
    save_stats(stats)

def get_remaining_free(user_id):
    data = get_user_data(user_id)
    return max(0, FREE_DAILY_LIMIT - data["free_used"])

def get_bonus_balance(user_id):
    data = get_user_data(user_id)
    return data.get("bonus", 0)

# ======================= РЕФЕРАЛЬНАЯ СИСТЕМА =======================
def get_referral_code(user_id):
    import hashlib, secrets
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

# ======================= КОМАНДЫ =======================
@bot.message_handler(commands=['start'])
def start_command(message: Message):
    user_id = message.from_user.id
    data = get_user_data(user_id)
    
    # Приветствие для новых пользователей
    if data.get("first_start", True):
        data["first_start"] = False
        save_user_data(user_id, data)
        bot.reply_to(message, f"🎉 Добро пожаловать!\n\nБот дарит вам {FREE_DAILY_LIMIT} бесплатных попыток.\n\nОтправьте голосовое или аудио, чтобы начать!")
    else:
        free_left = get_remaining_free(user_id)
        bonus = get_bonus_balance(user_id)
        bot.reply_to(message, f"🎙️ С возвращением!\n\n🎁 Бесплатных осталось: {free_left}\n⭐ Бонусных: {bonus}\n\nОтправьте голосовое или аудио!")

@bot.message_handler(commands=['balance'])
def balance_command(message: Message):
    user_id = message.from_user.id
    free_left = get_remaining_free(user_id)
    bonus = get_bonus_balance(user_id)
    bot.reply_to(message, f"💰 Ваш баланс:\n\n🎁 Бесплатных: {free_left}\n⭐ Бонусных: {bonus}")

@bot.message_handler(commands=['referral'])
def referral_command(message: Message):
    user_id = message.from_user.id
    ref_code = get_referral_code(user_id)
    bot_username = bot.get_me().username
    link = f"https://t.me/{bot_username}?start=ref_{ref_code}"
    invited, bonus = get_referral_stats(user_id)
    bot.reply_to(message, f"👥 Реферальная программа\n\n🔗 Ваша ссылка:\n{link}\n\n👥 Приглашено: {invited}\n⭐ Получено бонусов: {bonus}")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, "📖 Команды:\n/start - Начать\n/balance - Баланс\n/referral - Пригласить друга\n/help - Помощь")

# ======================= ОБРАБОТЧИК РЕФЕРАЛЬНЫХ ССЫЛОК =======================
@bot.message_handler(func=lambda message: message.text and message.text.startswith('/start ref_'))
def handle_ref_start(message: Message):
    user_id = message.from_user.id
    ref_code = message.text.replace('/start ref_', '').strip()
    
    # Ищем пригласившего
    stats = load_stats()
    referrer_id = None
    for uid, data in stats.items():
        if data.get("ref_code") == ref_code:
            referrer_id = int(uid.replace("user_", ""))
            break
    
    if referrer_id and referrer_id != user_id:
        add_referral_bonus(referrer_id, user_id)
        bot.reply_to(message, "🎉 Вас пригласил друг! Вы получили +3 бонусные попытки!")
    else:
        bot.reply_to(message, "👋 Добро пожаловать!")

# ======================= ЗАПУСК =======================
if __name__ == "__main__":
    print("🤖 Бот с полной логикой запущен!")
    bot.infinity_polling()
