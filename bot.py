import os
import json
import threading
import time
import telebot
from telebot.types import Message
from flask import Flask

# Конфигурация
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен!")

# Инициализация бота
bot = telebot.TeleBot(TOKEN)

# Ваши обработчики команд
@bot.message_handler(commands=['start'])
def start_command(message: Message):
    bot.reply_to(message, "👋 Бот работает!")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, "📖 Доступные команды: /start, /help")

# Запуск бота в отдельном потоке
def run_bot():
    bot.infinity_polling()

# Запуск Flask для поддержания порта
app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is running!", 200

@app.route('/health')
def health():
    return "OK", 200

if __name__ == "__main__":
    # Запуск бота в фоновом потоке
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    # Запуск веб-сервера Flask на порту 10000 (по умолчанию для Render)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
