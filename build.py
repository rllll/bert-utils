from pathlib import Path
import json
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from extract_feature import BertVector
import seaborn as sns
import matplotlib.pyplot as plt

dimens = ['空间', '动力', '操控', '能耗', '舒适性', '外观', '内饰', '性价比', '配置', '续航', '安全性', '环保', '质量与可靠性', '充电', '服务', '品牌', '智能驾驶', '其它', '总的来说']


def read_data_from_excel():
    type_list = ['宝马1系', '宝马2系', '宝马3系（1）','宝马3系（2）', '宝马3系（3）', '宝马3系（4）','宝马3系（5）', '宝马3系（6）', '宝马3系（7）','宝马3系（8）', '宝马3系（9）', '宝马3系（10)','宝马3系（11）', '宝马3系（12）', '宝马4系','宝马5系','宝马X1（1）', '宝马X1（2）', '宝马X1(5)', '宝马X2', '宝马X3']
    df = pd.read_excel('./data/bmw_all.xlsx', sheet_name=type_list, dtype=str)
    contents, labels = [], []
    for type in type_list:
        df_drop = df[type].dropna(subset=[dimens[-1], '具体评价'])
        contents += df_drop["具体评价"].tolist()
        labels += df_drop[dimens[-1]].tolist()
    real_contents = []
    encode_labels = []
    for idx, label in enumerate(labels):
        if label == '1':
            # 积极的
            encode_labels.append('POS')
            real_contents.append(contents[idx])
        elif label == '0':
            # 中性的
            encode_labels.append('NEU')
            real_contents.append(contents[idx])
        elif label == '-1':
            # 消极的
            encode_labels.append('NEG')
            real_contents.append(contents[idx])
    train_words, test_words, train_labels, test_labels = train_test_split(real_contents, encode_labels, test_size=0.2)
    return train_words, test_words, train_labels, test_labels

def save_words_to_text(mode, save_words):
    for word in save_words:
        if word.strip() != '':
            cut_word = ' '.join(jieba.cut(word.strip(), cut_all=False, HMM=True))
            with Path('{}.words.txt'.format(mode)).open('a', encoding='utf-8') as g:
                g.write('{}\n'.format(cut_word))

def save_labels_to_text(mode, save_labels):
    for word in save_labels:
        if word.strip() != '':
            with Path('{}.labels.txt'.format(mode)).open('a', encoding='utf-8') as g:
                g.write('{}\n'.format(word))

def save_all_data(train_words, test_words, train_labels, test_labels):
    temp = {
        "train_words": train_words,
        "train_labels": train_labels,
        "test_words": test_words,
        "test_labels": test_labels
    }
    with open('temp.json', 'w', encoding='utf-8-sig') as f:
        f.write(json.dumps(temp, ensure_ascii=False, indent=2))

def read_all_data():
    with open('temp.json', 'r', encoding='utf-8-sig') as f:
        all_data = json.loads(f.read())
    train_words = all_data['train_words']
    test_words = all_data['test_words']
    train_labels = all_data['train_labels']
    test_labels = all_data['test_labels']
    return train_words, test_words, train_labels, test_labels

def balance_data(X_train, y_train):
    tl = RandomUnderSampler(random_state=17)
    X_train_resampled, y_train_resampled = tl.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

if __name__ == '__main__':
    # train_words, test_words, train_labels, test_labels = read_data_from_excel()
    # print(len(train_words), len(train_labels))
    # print(len(test_words), len(test_labels))
    # save_all_data(train_words, test_words, train_labels, test_labels)
    train_words, test_words, train_labels, test_labels = read_all_data()
    print(len(train_words), len(train_labels))
    print(len(test_words), len(test_labels))
    bv = BertVector()
    bert_vec = bv.encode(train_words)
    print(len(bert_vec), bert_vec)
    X_train_resampled, y_train_resampled = balance_data(bert_vec, train_labels)
    y_train_resampled = {'labels': y_train_resampled}
    under_sampling = pd.concat([pd.DataFrame(y_train_resampled), pd.DataFrame(X_train_resampled)], axis=1)
    print(under_sampling)
    plt.figure(figsize=(10, 10))
    sns.countplot(x='labels', data=under_sampling)
    plt.title('Under Sampling Balanced Classes')
    plt.show()
    # save_words_to_text('train', train_words)
    # save_words_to_text('eval', test_words)
    # save_labels_to_text('train', train_labels)
    # save_labels_to_text('eval', test_labels)
