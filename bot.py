import os
import json
import threading
import time
import telebot
from telebot.types import Message
from http.server import HTTPServer, BaseHTTPRequestHandler

# ======================= НАСТРОЙКИ =======================
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не установлен!")

bot = telebot.TeleBot(TOKEN)

# ======================= КОМАНДЫ БОТА =======================
@bot.message_handler(commands=['start'])
def start_command(message: Message):
    bot.reply_to(message, "👋 Привет! Бот работает на Render!")

@bot.message_handler(commands=['help'])
def help_command(message: Message):
    bot.reply_to(message, "📖 Команды: /start, /help")

# ======================= ПРОСТОЙ HTTP-СЕРВЕР ДЛЯ RENDER =======================
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
    # Запускаем HTTP-сервер в отдельном потоке
    http_thread = threading.Thread(target=run_http_server)
    http_thread.daemon = True
    http_thread.start()
    
    print("🤖 Бот запущен!")
    bot.infinity_polling()
