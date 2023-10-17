# -*- coding: utf-8 -*-
with open('test_token.txt') as file:
    token = file.readline()

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
import shap
import telebot
from telebot import types  # для указание типов
from pathlib import Path
import warnings

warnings.simplefilter("ignore", UserWarning)

train_df = pd.read_csv('train_df.csv')

model = CatBoostClassifier()
model.load_model('model.cmb')

bot = telebot.TeleBot(token)


def show_graphs_beeswarm(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)

    fig = plt.gcf()
    shap.plots.beeswarm(shap_values[:], max_display=43, show=False)
    fig.savefig('beeswarm.png', bbox_inches='tight')

    return open('beeswarm.png', 'rb')


def show_graphs_force_0(data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)

    shap.force_plot(shap_values[0, :], matplotlib=True, show=False).savefig('force.png', bbox_inches='tight')

    return open('force.png', 'rb')


def send_file_csv_predict(data):
    data_predict = model.predict(data)
    data['predict'] = data_predict
    data.to_csv('data_predict.csv')

    return open('data_predict.csv', 'rb')

@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Предсказать срыв доставки по заказу")
    btn2 = types.KeyboardButton("Аналитическая система")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id,
                     text="Привет, {0.first_name}! Я помогу тебе предсказать будет ли срыв доставки или нет".format(
                         message.from_user), reply_markup=markup)

@bot.message_handler(content_types=['photo'])
def photo(message):
    bot.send_message(message.chat.id, text="Картинка недопустима")

@bot.message_handler(content_types=['text', 'photo'])
def func(message):
    if (message.text == "Предсказать срыв доставки по заказу"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(back)
        bot.send_message(message.chat.id, text="Отправьте мне ,пожалуйста,файл с данными", reply_markup=markup)

        @bot.message_handler(content_types=['document'])
        def ot(message):
            try:
                file_info = bot.get_file(message.document.file_id)
                downloaded_file = bot.download_file(file_info.file_path)
                if Path(file_info.file_path).suffixes[0] not in ['.csv', '.tsv', '.xlss']:
                    bot.send_message(message.chat.id, text='Вы отправили файл не табличного вида! Попробуйте снова')
                else:
                    src = message.document.file_name
                    with open(src, 'wb') as new_file:
                        new_file.write(downloaded_file)
                    markup = types.InlineKeyboardMarkup()
                    btn3 = types.InlineKeyboardButton(text="Да", callback_data="DA")
                    btn2 = types.KeyboardButton("Предсказать срыв доставки по заказу")
                    back = types.KeyboardButton("Вернуться в главное меню")
                    markup.add(btn3)
                    data = pd.read_csv(message.document.file_name)
                    # img1 = open(show_graphs_beeswarm(data), 'rb')
                    # img = open(show_graphs_force_1(data), 'rb')
                    # bot.send_photo(chat_id, img)
                    # bot.send_photo(chat_id, img1)
                    # bot.send_message(chat_id, send_file_csv_predict(data))
                    bot.send_document(message.chat.id, document=send_file_csv_predict(data))
                    bot.send_message(message.chat.id, text="Хотите посмотреть графики на основе вашего датасета ?",
                                     reply_markup=markup)

                    @bot.callback_query_handler(func=lambda call: call.data == 'DA')
                    def handle_trials(callback_query):
                        menu_button1 = types.InlineKeyboardButton(text="График Beeswarm", callback_data="bee")
                        menu_button2 = types.InlineKeyboardButton(text="График Force", callback_data="force")
                        keyboard = types.InlineKeyboardMarkup(row_width=4)
                        keyboard.add(menu_button1, menu_button2)
                        # Отправляем сообщение
                        bot.send_message(callback_query.message.chat.id,
                                         '1: Beeswarm - График показывающий, какие функции наиболее важны для модели. На графике будут показаны объекты которые сортируются по сумме величин значений SHAP по всем выборкам и используются значения SHAP, чтобы показать распределение влияния каждого объекта на выходные данные модели. Цвет представляет значение функции (красный максимум, синий минимум).\n' +
                                         '2: Force - В приведенном графике показаны функции, которые способствуют преобразованию выходных данных модели из базового значения (средний результат модели по переданному нами набору обучающих данных) к выходным данным модели. Функции, повышающие прогноз, показаны красным, а те, которые повышают прогноз, — синим.',
                                         reply_markup=keyboard)

                    @bot.callback_query_handler(func=lambda call: call.data == 'bee')
                    def bee(callback_query):
                        btn1 = types.InlineKeyboardButton(text="Вернуться к графикам", callback_data="DA")
                        markup = types.InlineKeyboardMarkup(row_width=4)
                        markup.add(btn1)
                        bot.send_message(callback_query.message.chat.id,
                                         "График Beeswarm: (вывод может занять некоторое время)")
                        bot.send_photo(message.chat.id, photo=show_graphs_beeswarm(data), reply_markup=markup)

                    @bot.callback_query_handler(func=lambda call: call.data == 'force')
                    def force(callback_query):
                        btn1 = types.InlineKeyboardButton(text="Вернуться к графикам", callback_data="DA")
                        markup = types.InlineKeyboardMarkup(row_width=4)
                        markup.add(btn1)
                        bot.send_message(callback_query.message.chat.id,
                                         "График Force: (вывод может занять некоторое время)")
                        bot.send_photo(message.chat.id, photo=show_graphs_force_0(data), reply_markup=markup)

            except Exception as err:
                bot.reply_to(message, err)


    elif (message.text == "Аналитическая система"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("О нашем проекте")
        btn2 = types.KeyboardButton("Немного аналитики")
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Задай мне вопрос", reply_markup=markup)

    elif (message.text == "Вернуться в главное меню"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton("Предсказать срыв доставки по заказу")
        button2 = types.KeyboardButton("Аналитическая система")
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text="Вы вернулись в главное меню", reply_markup=markup)
    elif (message.text == 'Немного аналитики'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Графики")
        btn2 = types.KeyboardButton("что-то еще1")
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Что именно вы хотите узнать?", reply_markup=markup)
    elif (message.text == 'Графики'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Ценность фича")
        btn2 = types.KeyboardButton("что-то еще2")
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Выберите график, который вас интересует", reply_markup=markup)
    elif (message.text == "Вернуться к графикам"):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("Ценность фича")
        btn2 = types.KeyboardButton("что-то еще2")
        back = types.KeyboardButton("Вернуться в главное меню")
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text="Вы вернулись к графикам", reply_markup=markup)


if __name__ == "__main__":
    bot.polling(none_stop=True)