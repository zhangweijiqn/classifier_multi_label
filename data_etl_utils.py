# -*- coding:utf-8 -*-
"""
func: 读取三级分类目录文件，多个文件整合为1个
file format: 一级分类，二级分类，三级分类，title，content，来源，其它信息(URL)
每个分类下多label用 | 分隔，如一级分类， 历史|军事
python src/com/four_pd/etl/third_category/read_data_dir.py
"""

import pandas as pd
from pandas.core.frame import DataFrame
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from classifier_multi_label.hyperparameters import Hyperparamters as hp

failed_files = []
label_sep = "|"


def extract_cn(line):
    cop = re.compile("[^\u4e00-\u9fa5^，。！：；？]")
    line = re.sub('\r|\n|\t', '', line)
    return cop.sub("", line)


def read_file(input_file):
    print('opening file: ', input_file)
    try:
        df = pd.read_csv(input_file, header=0)  # header=None表示原始文件数据没有列索引，这样的话read_csv会自动加上列索引
    except Exception as e:
        print("read_file exception: " + str(e))
        failed_files.append(input_file)
        return None
    df.dropna(axis=0, how='any', subset=['title'], inplace=True)
    df.dropna(axis=0, how='any', subset=['content'], inplace=True)
    print(len(df["title"].values))
    df['title'] = df['title'].apply(lambda x: extract_cn(x))
    df['content'] = df['content'].apply(lambda x: extract_cn(x))
    print('len_csv', len(df))
    return df


def read_dir(input_dir):
    file_list = os.listdir(input_dir)
    # print(file_list)
    l = 0
    df1 = pd.DataFrame()
    for file in file_list:  # 遍历文件夹
        if os.path.isdir(input_dir + "/" + file):
            sub_file_list = os.listdir(input_dir + "/" + file)
            print("打开文件：", input_dir + "/" + file, "文件数目", len(sub_file_list))
            for fs in sub_file_list:
                if not os.path.isdir(input_dir + "/" + file + "/" + fs):
                    print('打开文件：', input_dir + "/" + file + "/" + fs)
                    df2 = read_file(input_dir + "/" + file + "/" + fs)
                    if df2 is not None:
                        l = l + len(df2)
                        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并
        else:
            df2 = read_file(input_dir + "/" + file)
            if df2 is not None:
                l = l + len(df2)
                df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并
    print(len(df1))
    print('l', l)
    return df1


def merge_title_content(df):
    df['text'] = df['title'] + "。" + df['content']
    print(len(df))
    df.dropna(axis=0, how='any', subset=['三级分类'], inplace=True)
    return df


def enhance(df_raw):
    # df_raw['label'] = df_raw['label'].apply(lambda x: str(x))
    vcounts = df_raw.loc[:, 'label'].value_counts()
    mini_sampel = vcounts[vcounts < 200]
    print('原始数据长度', len(df_raw))
    con = 0
    for index, value in mini_sampel.iteritems():
        # print(index, value)
        df2 = df_raw.loc[df_raw['label'] == index]
        for i in range(300 // value):
            df_raw = pd.concat([df_raw, df2], axis=0, ignore_index=True)
        con += value * (300 // value)
        # print(index,value )
    print("增加了", con)
    print('增加后数据长度', len(df_raw))
    return df_raw


def process_label(df, cates, is_enhance=False, is_sampled=False, save_path=''):
    cates = list(cates.split(','))
    print("label includes", cates)
    if len(cates) > 1:  # labels 为list
        # 多标签构造
        assert len(cates) >= 2
        df.dropna(axis=0, how='any', subset=cates, inplace=True)
        for index, row in df.iterrows():
            clses = list(row[cate] for cate in cates)
            l = clses[0]
            for cls in clses[1:]:
                l = l + label_sep + cls
            df['label'] = str(l)
            if index % 5000 == 0:
                print(index, df['label'])
        df_raw = get_sampled_data(df, is_sampled)
    else:
        assert len(cates) == 1
        df.dropna(axis=0, how='any', subset=[cates[0]], inplace=True)
        df.rename(columns={cates[0]: 'label'}, inplace=True)
        df_raw = get_sampled_data(df, is_sampled)

    # 是否需要数据增强,默认将少于200的复制到300条
    if is_enhance:
        df_raw = enhance(df_raw)

    df_raw['label'] = df_raw['label'].apply(lambda x: x.split(label_sep))
    print(len(df_raw))
    print(df_raw.head())

    for i in range(len(df_raw)):
        for l in hp.label_vocabulary:
            if l in df_raw['label'].values[i]:
                df_raw.loc[i, l] = 1
            else:
                df_raw.loc[i, l] = 0
    df_raw.rename(columns={'text': 'content'}, inplace=True)
    fields = ['content'] + hp.label_vocabulary
    sub_df = df_raw[fields]

    train_data, val_data, test_data = split_dataset(sub_df)

    if save_path != '':
        train_data.to_csv(save_path + '/train.csv', index=False)
        val_data.to_csv(save_path + '/validation.csv', index=False)
        test_data.to_csv(save_path + '/test.csv', index=False)

    return train_data, val_data, test_data


def get_sampled_data(df, is_sampled):
    mt_cols = ['label', 'text']
    if is_sampled:
        df_raw = df[mt_cols].sample(frac=0.01, random_state=666).reset_index(drop=True)
    else:
        df_raw = df[mt_cols].sample(frac=1.0, random_state=666).reset_index(drop=True)
    return df_raw


def split_dataset(df_raw):
    # 数据格式为两列，一列文本一列标签
    # print(df_raw.head(10))
    # split data0
    train_set, x = train_test_split(df_raw,
                                    # stratify=df_raw['label'],
                                    test_size=0.1,
                                    random_state=45)
    val_set, test_set = train_test_split(x,
                                         # stratify=x['label'],
                                         test_size=0.5,
                                         random_state=43)

    return train_set, val_set, test_set


def read_pandas(input_dir, outputfile=""):
    df = read_dir(input_dir)

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)  # 重新生成index

    print("dataframe len: %d" % (len(df)))
    # for i in range(len(df)):
    #     print(df.values[i][0:3])

    df = merge_title_content(df)
    sub_df = df[['一级分类', '二级分类', '三级分类', 'text']]

    if outputfile != "":
        sub_df.to_csv(outputfile, index=False)  # 将结果保存为新的csv文件

    print("read failed files: %s" % str(failed_files))

    return sub_df


if __name__ == '__main__':
    #
    input_dir = '/Users/4paradigm/Projects/content-tags/data/third_category/input'
    outputfile = './data/result/third_category_merge.csv'
    df = read_pandas(input_dir, outputfile)
    process_label(df, cates='三级分类', save_path='./data/result')
