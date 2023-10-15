with open('test_token.txt') as file:
    token = file.readline()

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
import shap
import telebot
from pathlib import Path

train_df = pd.read_csv('train_df.csv')

model = CatBoostClassifier()
model.load_model('model.cmb')

bot = telebot.TeleBot(token)


def show_graphs_beeswarm(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)

    fig = plt.gcf()
    shap.plots.beeswarm(shap_values[:], color=plt.get_cmap("cool"), max_display=20, show=False)
    fig.savefig('beeswarm.png', bbox_inches='tight')

    return 'beeswarm.png'


def show_graphs_force_0(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)

    shap.force_plot(shap_values[0, :], matplotlib=True, show=False).savefig('force.png', bbox_inches='tight')

    return 'force.png'


def show_graphs_force_1(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)

    shap.force_plot(shap_values[1, :], matplotlib=True, show=False).savefig('force.png', bbox_inches='tight')

    return 'force.png'


def send_file_csv_predict(data):
    data_predict = model.predict(data)
    data['predict'] = data_predict
    data.to_csv('data_predict.csv')

    return open('data_predict.csv', 'rb')


@bot.message_handler(content_types=['document'])
def handle_docs_photo(message):
    try:
        chat_id = message.chat.id

        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        if Path(file_info.file_path).suffixes[0] not in ['.csv', '.tsv', '.xlss']:
            bot.send_message(chat_id, 'Вы отправили файл не табличного вида! Попробуйте снова')

        else:
            src = message.document.file_name
            with open(src, 'wb') as new_file:
                new_file.write(downloaded_file)

            data = pd.read_csv(message.document.file_name)

            # img = open(show_graphs_force_0(data), 'rb')
            # img1 = open(show_graphs_beeswarm(data), 'rb')
            # img = open(show_graphs_force(data), 'rb')
            # bot.send_photo(chat_id, img)
            # bot.send_photo(chat_id, img1)
            # bot.send_message(chat_id, send_file_csv_predict(data))
            bot.send_document(chat_id, document=send_file_csv_predict(data))

    except Exception as err:
        bot.reply_to(message, err)


bot.polling(none_stop=True)
