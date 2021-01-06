#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/aistudio/work/ai_asr_e2e/data")
import os
import tarfile
import argparse
import subprocess
from utils import create_manifest
from tqdm import tqdm
import shutil
import re
import string
import unicodedata
import pandas as pd

"""
-------------------------------------------------
   Description :  处理AISHELL数据的脚本【完全无问题，运行通过】
   Author :       saksim
   Date :         2020-10-19 14:59
   Email:         wh13624@my.bristol.ac.uk
   Mobile:        13999312573 13369058048
-------------------------------------------------

"""
# 常数设置
ROOT = "/home/aistudio/work/dataset/data_aishell"
CHINESE_TAG = "†"
ENGLISH_TAG = "‡"


# 获得数据
class GetTrainDfSeperate():
    def __init__(self,path=None):
        self.path=path
    def get_data(self):
        loop = True
        reader = pd.read_csv(self.path,error_bad_lines = False,sep = "\t",iterator=True,header=None)
        chunksize = 100000
        chunks = []
        while loop:
            try:
                chunk = reader.get_chunk(chunksize)
                chunks.append(chunk)
                print("READING DATA: ", len(chunks))
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        self.df = pd.concat(chunks,ignore_index=True)
        return self.df


# 数据转化
def traverse(root, path, search_fix=".txt"):
    f_list = []
    p = os.path.join(root, path)
    for s_p in sorted(os.listdir(p)):
        for sub_p in sorted(os.listdir(p + "/" + s_p)):
            if sub_p[len(sub_p)-len(search_fix):] == search_fix:
                print(">", path, s_p, sub_p)
                f_list.append(p + "/" + s_p + "/" + sub_p)
    return f_list


# 获得数据前缀
def get_absolute_path(x):
   if int(x[-4:]) <= 723:
       pre_path = "/home/aistudio/work/dataset/data_aishell/wav/train/"
   elif int(x[-4:]) <= 763:
       pre_path = "/home/aistudio/work/dataset/data_aishell/wav/dev/"
   elif int(x[-4:]) <= 916:
       pre_path = "/home/aistudio/work/dataset/data_aishell/wav/test/"
   return pre_path


# 对序列替换中文标点符号
def remove_punctuation(seq):
    # REMOVE CHINESE PUNCTUATION EXCEPT HYPEN / DASH AND FULL STOP
    seq = re.sub("[\s+\\!\/_,$%=^*?:@&^~`(+\"]+|[+！，。？、~@#￥%……&*（）:;：；《）《》“”()»〔〕]+", " ", seq)
    seq = seq.replace(" ' ", " ")
    seq = seq.replace(" ’ ", " ")
    seq = seq.replace(" ＇ ", " ")
    seq = seq.replace(" ` ", " ")
    seq = seq.replace(" '", "'")
    seq = seq.replace(" ’", "’")
    seq = seq.replace(" ＇", "＇")
    seq = seq.replace("’ ", " ")
    seq = seq.replace("＇ ", " ")
    seq = seq.replace("` ", " ")
    seq = seq.replace(".", "")
    seq = seq.replace("`", "")
    seq = seq.replace("-", " ")
    seq = seq.replace("?", " ")
    seq = seq.replace(":", " ")
    seq = seq.replace(";", " ")
    seq = seq.replace("]", " ")
    seq = seq.replace("[", " ")
    seq = seq.replace("}", " ")
    seq = seq.replace("{", " ")
    seq = seq.replace("|", " ")
    seq = seq.replace("_", " ")
    seq = seq.replace("(", " ")
    seq = seq.replace(")", " ")
    seq = seq.replace("=", " ")
    seq = seq.replace("doens't", "doesn't")
    seq = seq.replace("o' clock", "o'clock")
    seq = seq.replace("因为it's", "因为 it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("it ' s", "it's")
    seq = seq.replace("it' s", "it's")
    seq = seq.replace("y'", "y")
    seq = seq.replace("y ' ", "y")
    seq = seq.replace("看different", "看 different")
    seq = seq.replace("it'self", "itself")
    seq = seq.replace("it'ss", "it's")
    seq = seq.replace("don'r", "don't")
    seq = seq.replace("has't", "hasn't")
    seq = seq.replace("don'know", "don't know")
    seq = seq.replace("i'll", "i will")
    seq = seq.replace("you're", "you are")
    seq = seq.replace("'re ", " are ")
    seq = seq.replace("'ll ", " will ")
    seq = seq.replace("'ve ", " have ")
    seq = seq.replace("'re\n", " are\n")
    seq = seq.replace("'ll\n", " will\n")
    seq = seq.replace("'ve\n", " have\n")
    seq = remove_space_in_between_words(seq)
    return seq


# # 移除特定字符
# def remove_special_char(seq):
#     seq = re.sub("""[【】·．％°℃×→①ぃγ￣σς＝～•＋δ≤∶／⊥＿ñãíå∈△β［］±]+""", " ", seq)
#     return seq
def remove_special_char(seq):
    seq = re.sub("""[【】［］±]+""", " ", seq)
    return seq


# 移除字间的空格
def remove_space_in_between_words(seq):
    return seq.replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").strip().lstrip()


# 移除制表符等
def remove_return(seq):
    return seq.replace("\n", "").replace("\r", "").replace("\t", "")


# 处理
def preprocess(seq):
    seq = seq.lower()
    seq = re.sub("[\(\[].*?[\)\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\{\[].*?[\}\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\<\[].*?[\>\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = re.sub("[\【\[].*?[\】\]]", "", seq) # REMOVE ALL WORDS WITH BRACKETS (HESITATION)
    seq = seq.replace("\x7f", "")
    seq = seq.replace("\x80", "")
    seq = seq.replace("\u3000", " ")
    seq = seq.replace("\xa0", "")
    seq = seq.replace("[", " [")
    seq = seq.replace("]", "] ")
    seq = seq.replace("#", "")
    seq = seq.replace(",", "")
    seq = seq.replace("*", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    seq = seq.replace("\t", "")
    seq = seq.replace("~", "")
    seq = seq.replace("—", "")
    seq = seq.replace("  ", " ").replace("  ", " ")
    seq = re.sub('\<.*?\>','', seq) # REMOVE < >
    seq = re.sub('\【.*?\】','', seq) # REMOVE 【 】
    seq = remove_special_char(seq)
    seq = remove_space_in_between_words(seq)
    seq = seq.strip()
    seq = seq.lstrip()
    seq = remove_punctuation(seq)
    seq = remove_space_in_between_words(seq)
    return seq


# 判断是否为中文字符
def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'


# 判断是否包含中文字符
def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False


# 增加语言
def add_lang(seq):
    new_seq = ""
    words = seq.split(" ")
    lang = 0
    for i in range(len(words)):
        if is_contain_chinese_word(words[i]):
            if lang != 1:
                lang = 1
                new_seq += CHINESE_TAG
                # print("zh")
        else:
            if lang != 2:
                lang = 2
                new_seq += ENGLISH_TAG
                # print("en")
        if new_seq != "":
            new_seq += " "
        new_seq += words[i]
    return new_seq


# 分离中文字符
def separate_chinese_chars(seq):
    new_seq = ""
    words = seq.split(" ")
    for i in range(len(words)):
        if is_contain_chinese_word(words[i]):
            for char in words[i]:
                if new_seq != "":
                    new_seq += " "
                new_seq += char
        else:
            if new_seq != "":
                new_seq += " "
            new_seq += words[i]
    return new_seq


# 创建路径
def get_all_dir(root, path):
    p = os.path.join(root, path)
    if not os.path.exists(p):
        os.mkdir(p)
        os.mkdir(f"{p}/train")
        os.mkdir(f"{p}/dev")
        os.mkdir(f"{p}/test")
    else:
        pass


# 主函数
def main():
    path1 = "transcript_clean"
    path2 = "transcript_clean_lang"
    get_all_dir(ROOT, path1)
    get_all_dir(ROOT, path2)
    ## 获得文本数据
    #tr_file_list = traverse(root, "transcript", search_fix="")
    path_X = "/home/aistudio/work/dataset/data_aishell/transcript/result.txt"
    """
    发生了变化，因为之前的数据是每个里面分离得到的值，而现在的值是汇总值
    x.split("W")[0].split("BAC009")[-1]
    """
    # 进行数据遍历，并处理之
    # 在transcript_clean和transcript_clean_lang下 对训练、测试、预测的数据保存在其文件夹下
    x = GetTrainDfSeperate(path_X).get_data()
    x.columns = ['name_path', 'text_content']
    x['catogery'] = x['name_path'].apply(lambda x: get_absolute_path(x[6:11]).split("wav/")[-1][:-1])
    x['folder'] = x['name_path'].apply(lambda x: x[6:11])
    x["out_path_clean"] = x['name_path'].apply(
        lambda x: get_absolute_path(x[6:11]).split("wav/")[0] + "transcript_clean/" + get_absolute_path(x[6:11]).split("wav/")[-1] + f"{x}.txt")
    x["out_path_clean_lang"] = x['name_path'].apply(
        lambda x: get_absolute_path(x[6:11]).split("wav/")[0] + "transcript_clean_lang/" + get_absolute_path(x[6:11]).split("wav/")[-1] + f"{x}.txt")
    x['name_path'] = x['name_path'].apply(lambda x: get_absolute_path(x[6:11]) + x[6:11] + f"/{x}.wav")
    # 已经获得了需要开启的文件名，以及文件内容[out_path,text_content]
    # pandas 迭代器取数据进行
    for row_index, row in x.iterrows():
        # 文本
        line = row[1]
        # 路径
        new_text_file_path = row[-2]
        new_text_file_lang_path = row[-1]
        # 当前行所有数据且处理过
        line = preprocess(line).rstrip().strip().lstrip()
        # 得到每个字
        lang_line = separate_chinese_chars(line).replace("  ", " ")
        # 保存文本数据到指定路径
        with open(new_text_file_path, "w+", encoding="utf-8") as new_text_file:
            new_text_file.write(line + "\n")
        with open(new_text_file_lang_path, "w+", encoding="utf-8") as new_text_lang_file:
            new_text_lang_file.write(lang_line + "\n")
    # 获得数据 [获得全部wav数据的列表]
    tr_file_list = traverse(ROOT, "wav/train", search_fix="")
    dev_file_list = traverse(ROOT, "wav/dev", search_fix="")
    test_file_list = traverse(ROOT, "wav/test", search_fix="")
    print("MANIFEST")
    print(">>", len(tr_file_list))
    print(">>", len(dev_file_list))
    print(">>", len(test_file_list))
    # 开始标记
    alpha = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
             "v", "w", "x", "y", "z"]
    labels = {}
    labels["_"] = True
    for char in alpha:
        labels[char] = True
    # 获得训练数据汇总
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_train_manifest.csv", "w+") as train_manifest:
        for i in range(len(tr_file_list)):
            wav_filename = tr_file_list[i]
            text_filename = tr_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                train_manifest.write(wav_filename + "," + text_filename + "\n")
    # 获得测试数据汇总
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_dev_manifest.csv", "w+") as valid_manifest:
        for i in range(len(dev_file_list)):
            wav_filename = dev_file_list[i]
            text_filename = dev_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                valid_manifest.write(wav_filename + "," + text_filename + "\n")
    # 获得测试数据汇总2
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_test_manifest.csv", "w+") as test_manifest:
        for i in range(len(test_file_list)):
            wav_filename = test_file_list[i]
            text_filename = test_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                test_manifest.write(wav_filename + "," + text_filename + "\n")
    # 获得标记
    with open("/home/aistudio/work/dataset/data_aishell/labels/aishell_labels.json", "w+") as labels_json:
        labels_json.write("[")
        i = 0
        labels_json.write('\n"_"')
        for char in labels:
            if char == "" or char == "_" or char == " ":
                continue
            labels_json.write(',\n')
            if char == "\\":
                print("slash")
                labels_json.write('"')
                labels_json.write('\\')
                labels_json.write('\\')
                labels_json.write('"')
            elif char == '"':
                print('double quote', i, char)
                labels_json.write('"\\""')
            else:
                labels_json.write('"' + char + '"')
            i += 1
        labels_json.write(',\n" "\n]')
    print("label长度: {}".format(len(labels)))
    # 其他模型 train
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_train_lang_manifest.csv", "w+") as train_manifest:
        for i in range(len(tr_file_list)):
            wav_filename = tr_file_list[i]
            text_filename = tr_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                train_manifest.write(wav_filename + "," + text_filename + "\n")
    # dev
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_dev_lang_manifest.csv", "w+") as valid_manifest:
        for i in range(len(dev_file_list)):
            wav_filename = dev_file_list[i]
            text_filename = dev_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                valid_manifest.write(wav_filename + "," + text_filename + "\n")
    # tedt
    with open("/home/aistudio/work/dataset/data_aishell/manifests/aishell_test_lang_manifest.csv", "w+") as test_manifest:
        for i in range(len(test_file_list)):
            wav_filename = test_file_list[i]
            text_filename = test_file_list[i].replace(".wav", ".txt").replace("wav", "transcript_clean_lang")
            text_filename = "/".join(text_filename.split("/")[:-2]) + "/{}".format(text_filename.split("/")[-1])
            if os.path.isfile(text_filename):
                print(text_filename)
                with open(text_filename, "r", encoding="utf-8") as trans_file:
                    for line in trans_file:
                        for char in line:
                            if char != "\n" and char != "\r" and char != "\t":
                                labels[char] = True
                test_manifest.write(wav_filename + "," + text_filename + "\n")
    # label
    with open("/home/aistudio/work/dataset/data_aishell/labels/aishell_lang_labels.json", "w+") as labels_json:
        labels_json.write("[")
        i = 0
        labels_json.write('\n"_"')
        labels[CHINESE_TAG] = True
        labels[ENGLISH_TAG] = True
        for char in labels:
            if char == "" or char == "_" or char == " ":
                continue
            labels_json.write(',\n')
            if char == "\\":
                print("slash")
                labels_json.write('"')
                labels_json.write('\\')
                labels_json.write('\\')
                labels_json.write('"')
            elif char == '"':
                print('double quote', i, char)
                labels_json.write('"\\""')
            else:
                labels_json.write('"' + char + '"')
            i += 1
        labels_json.write(',\n" "\n]')
    print("label长度: {}".format(len(labels)))


if __name__ == '__main__':
    main()

