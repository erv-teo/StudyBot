from dotenv import load_dotenv
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

ALLOWED_USERS = {}

def restricted(func):
    async def wrapper(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_USERS:
            await update.message.reply_text("You are not authorized to use this bot.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

@restricted
async def start(update: Update, context:ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Hello World')

if __name__ == "__main__":
    load_dotenv()
    TELEGRAM_BOT_KEY = os.getenv("TELEGRAM_BOT_KEY")
    if TELEGRAM_BOT_KEY:
        application = ApplicationBuilder().token(TELEGRAM_BOT_KEY).build()
        start_handler = CommandHandler('start', start)
        application.add_handler(start_handler)


        application.run_polling()


    else:
        print("Telegram API Key not found")
