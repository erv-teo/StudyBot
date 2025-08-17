import telebot
from dotenv import load_dotenv
import os

load_dotenv()

TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_KEY")
if TELEGRAM_BOT_KEY:
    print(TELEGRAM_BOT_KEY)
    bot = telebot.TeleBot(TELEGRAM_BOT_KEY)

    @bot.message_handler(commands=['start', 'hello'])
    def hello_world(message):
        bot.reply_to(message, "Hello World")
    
    bot.infinity_polling()

else:
    print("Telegram API Key not found")
