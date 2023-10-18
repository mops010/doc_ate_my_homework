# -*- coding: utf-8 -*-
from telebot import types  # для указание типов
import requests
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
import shap
import telebot
from pathlib import Path
import warnings

warnings.simplefilter("ignore", UserWarning)

TOKEN = "6398808145:AAFZxCKd5CPVCBxR2oBkJRkVQUyO_dDN5cM"
model = CatBoostClassifier()
model.load_model('model.cmb')
bot = telebot.TeleBot(TOKEN)
explainer = shap.TreeExplainer(model)

cat_features = ['Поставщик', 'Материал', 'Категорийный менеджер', 'Операционный менеджер', 'Завод',
                'Закупочная организация',
                'Балансовая единица', 'Вариант поставки', 'ЕИ', 'Группа материалов', 'НРП', 'Месяц1', 'Месяц2',
                'Месяц3', 'День недели 2',
                'Отмена полного деблокирования заказа на закупку',
                'Изменение позиции заказа на закупку: изменение даты поставки на бумаге',
                'Изменение позиции заказа на закупку: дата поставки', 'Согласование заказа 1', 'Согласование заказа 2',
                'Согласование заказа 3']


def show_graphs_beeswarm(data):
    plt.clf()
    shap_values = explainer(data)
    fig = plt.gcf()
    shap.plots.beeswarm(shap_values[:], max_display=43, show=False)
    fig.savefig('beeswarm.png', bbox_inches='tight')
    return open('beeswarm.png', 'rb')


def send_file_csv_predict(data):
    data_predict = model.predict(data)
    data['predict'] = data_predict
    data.to_csv('data_predict.csv')

    return open('data_predict.csv', 'rb')


def show_graphs_force_0(data):
    plt.clf()
    shap_values = explainer(data)

    shap.force_plot(shap_values[0, :], matplotlib=True, show=False).savefig('force.png', bbox_inches='tight')

    return open('force.png', 'rb')


def justification_of_the_predict(data, message_text):
    shap_values = explainer(data)[int(message_text)]
    most_important = np.argmax(shap_values.abs.values)
    data_predict = model.predict(data)
    if data_predict[int(message_text)] == 1:
        return 'Доставка будет вовремя'
    if data.columns[most_important] in cat_features:
        if data.columns[most_important] == "Поставщик":
            return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. Если вас не устраивает значение, то советуем заменить его на значения: 930, 1253 или 1126. Это самые надежные поставщики'
        if data.columns[most_important] == "Операционный менеджер":
            return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. Если вас не устраивает значение, то советуем заменить его на значение 36. Это самые результативный оператов'
        if data.columns[most_important] == "День недели 2":
            return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. Если вас не устраивает значение, то советуем заменить его на значение 6. Советуем делать заказ в этот день'
    else:
        if shap_values.values[most_important] > 0:
            if data.columns[most_important] == "Длительность":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела должно быть около 49 дней'
            if data.columns[most_important] == "Количество позиций":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела значение должно быть около 14'
            if data.columns[most_important] == "Количество изменений после согласований":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела должно быть около 16 изменений соглосований'
            if data.columns[most_important] == "До поставки":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела значение должно быть около 48'
            return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}, значение фича слишком большое'
        else:
            if data.columns[most_important] == "Длительность":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела должно быть около 49 дней'
            if data.columns[most_important] == "Количество изменений после согласований":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела должно быть около 16 изменений соглосований'
            if data.columns[most_important] == "Количество позиций":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела значение должно быть около 14'
            if data.columns[most_important] == "До поставки":
                return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}. В среднем для того, чтобы доставка успела значение должно быть около 48'
            return f'Модель выдаёт такой результат в большей степени из-за фича {data.columns[most_important]}, значение фича слишком маленькое'


URL = 'https://api.telegram.org/bot'


def send_photo_file(chat_id, img):
    files = {'photo': open(img, 'rb')}
    requests.post(f'{URL}{TOKEN}/sendPhoto?chat_id={chat_id}', files=files)


@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Предсказать срыв доставки по заказу")
    btn2 = types.KeyboardButton("Аналитическая система")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id,
                     text='Привет, {0.first_name}! Я помогу тебе предсказать будет ли срыв доставки или нет'.format(
                         message.from_user), reply_markup=markup)


def send_photo_file(chat_id, img):
    files = {'photo': open(img, 'rb')}
    requests.post(f'{URL}{TOKEN}/sendPhoto?chat_id={chat_id}', files=files)


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
                    btn3 = types.InlineKeyboardButton(text='Да', callback_data='DA')
                    markup.add(btn3)
                    data = pd.read_csv(message.document.file_name)
                    bot.send_document(message.chat.id, document=send_file_csv_predict(data))
                    bot.send_message(message.chat.id, text='Хотите посмотреть графики на основе вашего датасета ?',
                                     reply_markup=markup)
                    markup = types.InlineKeyboardMarkup()
                    btn3 = types.InlineKeyboardButton(text='Да', callback_data='DA1')
                    markup.add(btn3)
                    bot.send_message(message.chat.id, text='Хотите узнать основную причину предикта?',
                                     reply_markup=markup)

                    def digit(message):
                        if message.text.isdigit():
                            bot.send_message(message.chat.id, justification_of_the_predict(data, message.text))

                    @bot.callback_query_handler(func=lambda call: call.data == 'DA1')
                    def handle_trials(callback_query):
                        keyboard = types.InlineKeyboardMarkup(row_width=4)
                        # Отправляем сообщение
                        bot.send_message(callback_query.message.chat.id,
                                         'Введите номер индекса строки , по которой вы хотите получить основную причину этого предикта',
                                         reply_markup=keyboard)
                        bot.register_next_step_handler(message, digit)

                    @bot.callback_query_handler(func=lambda call: call.data == 'DA')
                    def handle_trials(callback_query):
                        menu_button1 = types.InlineKeyboardButton(text='График Beeswarm', callback_data='bee')
                        menu_button2 = types.InlineKeyboardButton(text='График Force', callback_data='force')
                        keyboard = types.InlineKeyboardMarkup(row_width=4)
                        keyboard.add(menu_button1, menu_button2)
                        # Отправляем сообщение
                        bot.send_message(callback_query.message.chat.id,
                                         '1: Beeswarm - График показывающий, какие функции наиболее важны для модели. На графике будут показаны объекты которые сортируются по сумме величин значений SHAP по всем выборкам и используются значения SHAP, чтобы показать распределение влияния каждого объекта на выходные данные модели. Цвет представляет значение функции (красный максимум, синий минимум).\n' +
                                         '2: Force - В приведенном графике показаны функции, которые способствуют преобразованию выходных данных модели из базового значения (средний результат модели по переданному нами набору обучающих данных) к выходным данным модели. Функции, повышающие прогноз, показаны красным, а те, которые повышают прогноз, — синим.',
                                         reply_markup=keyboard)

                    @bot.callback_query_handler(func=lambda call: call.data == 'bee')
                    def bee(callback_query):
                        btn1 = types.InlineKeyboardButton(text='Вернуться к графикам', callback_data='DA')
                        markup = types.InlineKeyboardMarkup(row_width=4)
                        markup.add(btn1)
                        bot.send_message(callback_query.message.chat.id,
                                         'График Beeswarm: (вывод может занять некоторое время)')
                        bot.send_photo(message.chat.id, photo=show_graphs_beeswarm(data), reply_markup=markup)

                    @bot.callback_query_handler(func=lambda call: call.data == 'force')
                    def force(callback_query):
                        btn1 = types.InlineKeyboardButton(text='Вернуться к графикам', callback_data='DA')
                        markup = types.InlineKeyboardMarkup(row_width=4)
                        markup.add(btn1)
                        bot.send_message(callback_query.message.chat.id,
                                         'График Force: (вывод может занять некоторое время)')
                        bot.send_photo(message.chat.id, photo=show_graphs_force_0(data), reply_markup=markup)

            except Exception as err:
                bot.send_message(message.chat.id, str(err))


    elif (message.text == 'Аналитическая система'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('О нашей модели')
        btn2 = types.KeyboardButton('Немного аналитики')
        back = types.KeyboardButton('Вернуться в главное меню')
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text='Задай мне вопрос', reply_markup=markup)

    elif (message.text == 'О нашей модели'):
        bot.send_message(message.chat.id,
                         text='Наша модель построенна на архитектуре catboost и имеет 91% правильных ответов. Благодаря этому наша модель сможет как можно точнее предсказать сорвется ли доставка или нет. Наш чат-бот сможет помочь вам в подборе наиболее лучших параметров для поставки при помощи графиков на основе ваших данных, а так же анализу наиболее полезных столбцов и наших собственных выводов.')

    elif (message.text == 'Вернуться в главное меню'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton('Предсказать срыв доставки по заказу')
        button2 = types.KeyboardButton('Аналитическая система')
        markup.add(button1, button2)
        bot.send_message(message.chat.id, text='Вы вернулись в главное меню', reply_markup=markup)
    elif (message.text == 'Немного аналитики'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('НАДО ПРИДУМАТЬ')
        btn2 = types.KeyboardButton('Аналитика данных')
        back = types.KeyboardButton('Вернуться в главное меню')
        markup.add(btn1, btn2, back)
        bot.send_message(message.chat.id, text='Что именно вы хотите узнать?', reply_markup=markup)
    elif (message.text == 'Аналитика данных'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn2 = types.KeyboardButton('Аналитическая система')
        back = types.KeyboardButton('Вернуться в главное меню')
        markup.add(btn2, back)
        bot.send_message(message.chat.id,
                         text='Чтобы срыв доставки не случился важно знать, какие стобцы имеют наибольшую важнось для предотвращения срыва. Мы проанализировали данные и предоставляем вам график который поможет вам принять верное решение.',
                         reply_markup=markup)
        send_photo_file(message.chat.id, 'features_importans')
    elif (message.text == 'Ценность фича'):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton('Вернуться к графикам')
        back = types.KeyboardButton('Вернуться в главное меню')
        markup.add(btn1, back)
        bot.send_message(message.chat.id,
                         text='На этом графике мы можем наглядно увидеть, насколько важен каждый фич для нашей модели',
                         reply_markup=markup)
        send_photo_file(message.chat.id, 'features_importans')


if __name__ == "__main__":
    bot.polling(none_stop=True)