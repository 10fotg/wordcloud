# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from wordcloud import WordCloud

import matplotlib as mpl

# ตั้งค่าฟอนต์เป็น Seppuri-Regular หากติดตั้งแล้ว
font_path = 'SDNFONT/Seppuri-Regular.otf'
mpl.font_manager.fontManager.addfont(font_path)
mpl.rc('font', family='Seppuri-Regular')

thai_stopwords_list = list(thai_stopwords())

# กำหนดเส้นทางไฟล์สำหรับไฟล์ Excel ที่ต้องการ
file_path = "Comment/tiktok.xlsx"

def read_file(file_path):
    file_temp = pd.read_excel(file_path)
    file_temp = file_temp.dropna()
    return file_temp

def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final if word.lower() not in thai_stopwords_list)
    return final

def process_text_in_dataframe(input_df):
    input_df['text_tokens'] = input_df.iloc[:, 0].apply(text_process)
    return input_df

def plot_word_cloud(input_df):
    word_all = " ".join(text for text in input_df['text_tokens'])
    reg = r"[ก-๙a-zA-Z']+"
    wordcloud = WordCloud(stopwords=thai_stopwords_list, background_color='white',
                          max_words=2000, height=2048, width=1638, font_path=font_path,
                          regexp=reg).generate(word_all)
    plt.figure(figsize=(20.48,16.38))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.savefig('wordcloud_high_res.png', dpi=300)  # ตัวอย่างความละเอียด 300 dpi
    plt.axis('off')
    plt.show()

def from_file_path_to_word_cloud(file_path):
    print('---Read File...---')
    data_df = read_file(file_path)
    print('---Read File: Done---')
    print('---Process Text...---')
    data_df = process_text_in_dataframe(data_df)
    print('---Process Text: Done---')
    print('---Plotting Word Cloud---')
    plot_word_cloud(data_df)

# เริ่มการสร้าง WordCloud
from_file_path_to_word_cloud(file_path)
