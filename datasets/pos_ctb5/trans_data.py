# transfer data in current directory
# trans dev.char.bmes, train.char.bmes, test.char.bmes to .tsv
# .char.bmes每一行的格式为：
# 嘉 义 有 一 名 驾 驭 人 今 天 开 着 进 口 的 富 豪 轿 车 在 路 边 倒 车 的 时 候 ， 没 想 到 车 子 失 控 ， 撞 进 了 一 家 店 里 面 。	B-NR E-NR S-VE S-CD S-M B-NN M-NN E-NN B-NT E-NT S-VV S-AS B-VV E-VV S-DEC B-NR E-NR B-NN E-NN S-P B-NN E-NN B-VV E-VV S-DEC B-NN E-NN S-PU S-AD B-VV E-VV B-NN E-NN B-VV E-VV S-PU B-VV E-VV S-AS S-CD S-M S-NN B-LC E-LC S-PU
# 将其转换为每行两个列表，第一个列表为字，第二个列表为标注

import os
import json
import re
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 读取数据
    data_dir = os.path.join(os.path.dirname(__file__))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_path = os.path.join(data_dir, "train.char.bmes")
    dev_path = os.path.join(data_dir, "dev.char.bmes")
    test_path = os.path.join(data_dir, "test.char.bmes")

    print("train_path:", train_path)

    label_list = []

    # 转换数据格式
    def convert_data(input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                if not line:
                    continue
                words, tags = line.strip().split("\t")
                words = words.split(' ')
                tags = tags.split(' ')
                words = [word for word in words if word not in [" ", ""]]
                tags = [tag for tag in tags if tag not in [" ", ""]]

                for tag in tags:
                    if tag not in label_list:
                        label_list.append(tag)

                if len(words) != len(tags):
                    print(f"Error: {input_file} line {line.strip()} length mismatch")
                    print(f"Words: {words}, Tags: {tags}")
                    continue
                # 输出为 tsv 格式，每行一个词+标签
                for word, tag in zip(words, tags):
                    f.write(f"{word}\t{tag}\n")
                f.write("\n")



    convert_data(train_path, os.path.join(data_dir, "train.tsv"))
    convert_data(dev_path, os.path.join(data_dir, "dev.tsv"))
    convert_data(test_path, os.path.join(data_dir, "test.tsv"))

    # 保存标签列表，用逗号分隔，保存为txt
    label_list.sort()
    label_list_path = os.path.join(data_dir, "label_list.txt")
    with open(label_list_path, "w", encoding="utf-8") as f:
        f.write("\""+"\",\"".join(label_list)+"\"")