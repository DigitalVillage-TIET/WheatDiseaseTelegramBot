import os
import Inference
import logging
from time import sleep
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    Filters,
    MessageHandler,
    CallbackContext
)



TOKEN = os.environ.get("TELEGRAM_ID")
format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format = format, level=logging.INFO, filename = "WheatDiseaseBotLogFile.txt", )
logger = logging.getLogger(__name__)


if os.path.isdir(os.getcwd() + "/Images"):
    pass
else:
    os.mkdir(os.getcwd() + "/Images")
logger.info(f"A directory name Images has be created")

def start(update: Update, context: CallbackContext):
    yourname = update.message.from_user.first_name
    chat_id = update.message.chat.id
    logger.info("User %s started the conversation.",yourname)
    msg = f"ਸਤ ਸ੍ਰੀ ਅਕਾਲ {yourname}! Wheat Crop Disease Identification Bot ਵਿੱਚ ਤੁਹਾਡਾ ਸੁਆਗਤ ਹੈ \nनमस्ते {yourname}! गेहूं फसल रोग पहचान बॉट में आपका स्वागत है \nHello {yourname}! Welcome to Wheat Crop Disease Identification Bot"
    context.bot.sendChatAction(chat_id = chat_id, action = "typing") 
    sleep(2) 
    context.bot.send_message(update.message.chat.id, msg)

def cancel(update: Update, context: CallbackContext):
    """Cancels and ends the conversation."""
    username = update.message.from_user.first_name
    chat_id = update.message.chat.id
    logger.info("User %s canceled the conversation.", username)
    msg = f"ਬਾਈ {username}! ਤੁਹਾਡਾ ਦਿਨ ਚੰਗਾ ਬੀਤੇ \nअलविदा {username}! आप का दिन मंगलमय हो \nBye {username}!, Have a good day."
    context.bot.sendChatAction(chat_id = chat_id, action = "typing") 
    sleep(2) 
    context.bot.send_message(update.message.chat.id, msg)

def help(update: Update, context: CallbackContext):
    """Help Command get the information about bot"""
    user = update["message"]["from_user"]
    chat_id = update.message.chat.id
    logger.info("User %s has chosen help command" % user)
    msg = """
            ਮੈਂ ਇੱਕ ਚਿੱਤਰ ਪਛਾਣ ਟੈਲੀਗ੍ਰਾਮ ਬੋਟ ਹਾਂ ਜੋ ਕਣਕ ਦੀਆਂ ਫਸਲਾਂ ਵਿੱਚ ਬਿਮਾਰੀਆਂ ਦੀ ਪਛਾਣ ਕਰ ਸਕਦਾ ਹੈ।
            ਬੋਟ ਨੂੰ ਸ਼ੁਰੂ ਕਰਨ ਲਈ '/start' ਟਾਈਪ ਕਰੋ ਅਤੇ '/cancel' ਨੂੰ ਰੱਦ ਕਰਨ ਜਾਂ ਬਾਹਰ ਆਉਣ ਲਈ।
            ਇਹ ਬੋਟ 3 ਭਾਸ਼ਾਵਾਂ ਵਿੱਚ ਆਉਟਪੁੱਟ ਹੈ ਜੋ ਕਿ ਪੰਜਾਬੀ, ਹਿੰਦੀ, ਅੰਗਰੇਜ਼ੀ ਹਨ।
            ਬਿਮਾਰੀ ਬਾਰੇ ਜਾਣਨ ਲਈ ਮੈਨੂੰ ਕਣਕ ਦੀ ਫਸਲ ਦੀ ਤਸਵੀਰ ਭੇਜੋ।
            ----------------------------------------------------------------------------------
            मैं एक इमेज रिकग्निशन टेलीग्राम बॉट हूं जो गेहूं की फसलों में बीमारियों की पहचान कर सकता है।
            बॉट प्रकार शुरू करने के लिए '/start' और रद्द करने या '/cancel' से बाहर निकलने के लिए।
            यह बॉट 3 भाषाओं में आउटपुट करता है जो पंजाबी, हिंदी, अंग्रेजी हैं।
            बीमारी के बारे में जानने के लिए मुझे गेहूं की फसल की तस्वीर भेजें।
            ----------------------------------------------------------------------------------
            I am an image recognition telegram bot that can identify diseases in Wheat crops.
            To start the bot type '/start' and to cancel or exit '/cancel'.
            This bot output in 3 languages which are Punjabi, Hindi, English.
            Send me a wheat crop image to know about the disease.
          """
    context.bot.sendChatAction(chat_id = chat_id, action = "typing") 
    sleep(2)     
    context.bot.send_message(update.message.chat.id, msg)

def DiseasePrediction(update:Update,context:CallbackContext):
    """Disease Prediction Function"""
    username = update.message.from_user.first_name
    chat_id = str(update.message.chat.id)
    msg_id = update.message.message_id
    logger.info("User %s has tested one image" % username)
    image_path = "%s_%s_%s.jpg" % (username,chat_id, msg_id)
    customPath = os.getcwd() + "/Images/" + image_path
    file = update.message.photo[-1].file_id
    obj = context.bot.get_file(file)
    obj.download(customPath)

    DC = Inference.DiseaseClassification(filename = customPath)
    English, Hindi, Punjabi = DC.prediction()

    TitleList = ["I processed the image and it has the following diseass:-",
                 "मैंने छवि को संसाधित किया है और इसे निम्न रोग है:-",
                 "ਮੈਂ ਚਿੱਤਰ ਨੂੰ ਸੰਸਾਧਿਤ ਕੀਤਾ ਅਤੇ ਇਸ ਵਿੱਚ ਹੇਠ ਲਿਖੀ ਬਿਮਾਰੀ ਹੈ:-"]
    
    responseEnglish = f"{TitleList[0]} \n{English[0][0]} (Confidence = {English[0][1]:.3}) \n{English[1][0]} (Confidence = {English[1][1]:.3}) \n{English[2][0]} (Confidence = {English[2][1]:.3}) \n{English[3][0]} (Confidence = {English[3][1]:.3}) \n{English[4][0]} (Confidence = {English[4][1]:.3}) \n{English[5][0]} (Confidence = {English[5][1]:.3}) \n{English[6][0]} (Confidence = {English[6][1]:.3})"
    responseHindi = f"{TitleList[1]} \n{Hindi[0][0]} (आत्मविश्वास = {Hindi[0][1]:.3}) \n{Hindi[1][0]} (आत्मविश्वास = {Hindi[1][1]:.3}) \n{Hindi[2][0]} (आत्मविश्वास = {Hindi[2][1]:.3}) \n{Hindi[3][0]} (आत्मविश्वास = {Hindi[3][1]:.3}) \n{Hindi[4][0]} (आत्मविश्वास = {Hindi[4][1]:.3}) \n{Hindi[5][0]} (आत्मविश्वास = {Hindi[5][1]:.3}) \n{Hindi[6][0]} (आत्मविश्वास = {Hindi[6][1]:.3})"
    responsePunjabi = f"{TitleList[2]} \n{Punjabi[0][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[0][1]:.3}) \n{Punjabi[1][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[1][1]:.3}) \n{Punjabi[2][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[2][1]:.3}) \n{Punjabi[3][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[3][1]:.3}) \n{Punjabi[4][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[4][1]:.3}) \n{Punjabi[5][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[5][1]:.3}) \n{Punjabi[6][0]} (ਦਾ ਭਰੋਸਾ = {Punjabi[6][1]:.3})"
    
    response = responsePunjabi + "\n" + "-" * 45 + "\n" + responseHindi + "\n" + "-" * 45 + "\n" + responseEnglish
    context.bot.sendChatAction(chat_id = update.message.chat_id, action = "typing") 
    sleep(7) 
    context.bot.send_message(chat_id = update.message.chat_id, text = response)
    logger.info(f"Image Name {username}_{chat_id}_{msg_id}.jpg has Response {responseEnglish}")

def main():
    updater = Updater(token = TOKEN, use_context = True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("cancel", cancel))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(MessageHandler(Filters.photo, DiseasePrediction))

    updater.start_webhook(listen = "0.0.0.0", port = os.environ.get("PORT",443), url_path = TOKEN, webhook_url = "https://wheatdiseasetelegram-app.herokuapp.com/" + TOKEN)
    #updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()