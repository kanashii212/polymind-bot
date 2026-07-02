import os
import telebot
from telebot.types import Message

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message: Message):
    bot.reply_to(message, "👋 Привет! Я бот MAB Gateway. Я работаю!")

@bot.message_handler(func=lambda message: True)
def echo_all(message: Message):
    bot.reply_to(message, f"Вы написали: {message.text}")

if __name__ == "__main__":
    print("🤖 Бот запущен и работает!")
    bot.infinity_polling()
