#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセットに対する処理をまとめたクラス
Created on 7th Aug
Author: Akihiro Inui
Mail: mail@akihiroinui.com
"""

# Import library
import os
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class DatasetProcess:

    @staticmethod
    def data_loader(dataset_folder: str = "images", image_height: int = 122, image_width: int = 110):
        """
        画像データを保存しているフォルダからデータセットを作成
        :param   dataset_folder:    データセットのフォルダパス
        :param   image_height:      リサイズ後の画像の高さ
        :param   image_width:       リサイズ後の画像の幅
        :return  np.asarray(data):  numpy array 形式のデータ
        :return  np.asarray(label): numpy array 形式のラベル
        """

        # データセットフォルダ下のサブフォルダ名を取得
        folder_names = DatasetProcess.get_folder_names(dataset_folder)

        data = []   # 全ての画像の情報を保持する配列
        label = []  # dataに対応した画像のラベルを保持する配列

        # 各フォルダから画像データとラベルを取得
        for folder in folder_names:

            # サブフォルダ内の画像ファイル名を取得
            file_names = DatasetProcess.get_file_names(os.path.join(dataset_folder, folder))

            # 画像ファイルを読み込み、画像データとラベルをリストに追加
            for file in file_names:
                image = img_to_array(
                    load_img(os.path.join(dataset_folder, folder, file), target_size=(image_height, image_width)))
                data.append(image)
                label.append(int(folder))
        return np.asarray(data), np.asarray(label)

    @staticmethod
    def train_test_split(data, label, num_classes: int, test_rate: int = 0.20):
        """
        データ、ラベルを訓練/テストに分割
        :param   data:        numpy array 形式で保存したデータ
        :param   label:       numpy array 形式で保存したラベル
        :param   num_classes: 分類クラス数
        :param   test_rate:   テストに利用するデータの割合
        :return  train_data:  numpy array 形式の訓練データ
        :return  test_data:   numpy array 形式のテストデータ
        :return  train_label: numpy array 形式の訓練ラベル
        :return  test_label:  numpy array 形式のテストラベル
        """

        # 正規化
        data = data.astype('float32') / 255.0

        # ラベルをone-hotエンコーディング
        label = np_utils.to_categorical(label, num_classes)

        # 訓練/テストにデータとラベルをシャッフルして分割
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=test_rate)
        return train_data, test_data, train_label, test_label

    @staticmethod
    def augment_data(input_data, input_label):
        """
        入力したデータに対して水増し処理を行う
        :param   input_data:        numpy array 形式で保存したデータ
        :param   input_label:       numpy array 形式で保存したラベル
        :return  np.asarray(data):  numpy array 形式のデータ
        :return: np.asarray(label): numpy array 形式のラベル
        """
        # 水増し後のデータを保存するための配列
        new_data = []
        new_label = []
        # カウンタ
        sample_num = 0

        # 画像毎に水増し処理
        for sample in input_data:
            # 元画像データを配列に追加
            new_data.append(sample)
            new_label.append(input_label[sample_num, :])

            # データの水増し処理 必要に応じてコメントを除く

            # 水平方向のランダム反転を行う場合、以下の3行のコメントを外してください。
            # if np.random.rand() < 0.5:
            #   new_data.append(DatasetProcess.horizontal_flip(sample))
            #   new_label.append(input_label[sample_num, :])

            # 垂直方向のランダム反転を行う場合、以下の3行のコメントを外してください。
            # if np.random.rand() < 0.5:
            #   new_data.append(DatasetProcess.vertical_flip(sample))
            #   new_label.append(input_label[sample_num, :])

            # ランダムカットアウトを行う場合、以下の3行のコメントを外してください。
            # if np.random.rand() < 0.5:
            #   new_data.append(DatasetProcess.cutout(sample))
            #   new_label.append(input_label[sample_num, :])

            # ランダム消去を行う場合、以下の3行のコメントを外してください。
            # if np.random.rand() < 0.5:
            #   new_data.append(DatasetProcess.random_erasing(sample))
            #   new_label.append(input_label[sample_num, :])

            # カウンタをアップデート
            sample_num += 1
        return np.asarray(new_data), np.asarray(new_label)

    @staticmethod
    def balance_dataset(input_data, input_label):
        """
        クラスのデータ数に偏りがあるデータセットをオーバーサンプリングで均等化
        :param   input_data:        numpy array 形式で保存したデータ
        :param   input_label:       numpy array 形式で保存したラベル
        :return  np.asarray(data):  numpy array 形式のデータ
        :return: np.asarray(label): numpy array 形式のラベル
        """

        # 4次元配列データを2次元配列データに変換
        num_samples, height, width, channel = input_data.shape
        d2_data = input_data.reshape((num_samples, height * width * channel))

        # オーバーサンプル
        ros = RandomOverSampler(random_state=0)
        balanced_data, balanced_label = ros.fit_sample(d2_data, input_label)

        # 2次元配列を４次元配列に戻す
        balanced_data = balanced_data.reshape(len(balanced_data), height, width, channel)
        return balanced_data, balanced_label

    @staticmethod
    def get_folder_names(folder_path: str, sort=True) -> list:
        """
        入力したディレクトリ内のサブディレクトリ名をリストとして返す
        :param   folder_path:       入力ディレクトリパス
        :param   sort:              アルファベット順にソートしてリストを返す場合 True
        :return  folder_names_list: 入力ディレクトリ内のサブディレクトリ名のリスト
        """

        # サブディレクトリ名を取得
        folder_names_list = []
        for folder_name in os.listdir(folder_path):
            if not folder_name.startswith('.'):
                folder_names_list.append(folder_name)

        # アルファベット順にソート
        if sort is True:
            folder_names_list = sorted(folder_names_list)
        return folder_names_list

    @staticmethod
    def get_file_names(folder_path: str, sort=True) -> list:
        """
        入力したディレクトリ内のファイル名をリストとして返す
        :param   folder_path:      入力ディレクトリパス
        :param   sort:             アルファベット順にソートしてリストを返す場合 True
        :return  file_names_list:  入力ディレクトリ内のサブディレクトリ名のリスト
        """

        # ファイル名を取得
        file_names_list = []
        for file_name in os.listdir(folder_path):
            if not file_name.startswith('.'):
                file_names_list.append(file_name)

        # アルファベット順にソート
        if sort is True:
            file_names_list = sorted(file_names_list)
        return file_names_list

    @staticmethod
    def noise(image_data):
        """
        画像にノイズを追加
        :param  image_data:  numpy array 形式の画像データ
        :return noised_data: numpy array 形式のノイズ追加後画像データ
        """
        row, col, ch = image_data.shape
        mean = 0
        sigma = 15
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        # noised_image = src + gauss
        noised_image = image_data + gauss
        return noised_image

    @staticmethod
    def horizontal_flip(image_data):
        """
        水平方向に反転
        :param  image_data: numpy array 形式の画像データ
        :return image_data: numpy array 形式の水平反転後画像データ
        """
        image_data = image_data[:, ::-1, :]
        return image_data

    @staticmethod
    def vertical_flip(image_data):
        """
        垂直方向に反転
        :param  image_data: numpy array 形式の画像データ
        :return image_data: numpy array 形式の垂直反転後画像データ
        """
        image_data = image_data[::-1, :, :]
        return image_data

    @staticmethod
    def cutout(original_image_data, mask_size=55):
        """
        画像の一部分にマスクをかける マスクの画素値は画像の平均
        :param   original_image_data: numpy array 形式の画像データ
        :param   mask_size:           マスクのサイズ
        :return: image_data:          numpy array 形式のカットアウト後画像データ
        """
        # 最後に使うfill()は元の画像を書き換えるので、コピーしておく
        image_data = np.copy(original_image_data)

        # 平均の画素値を計算する
        mask_value = image_data.mean()

        # 画像の高さと幅を計算
        height, width, _ = image_data.shape

        # マスクをかける場所のtop, leftをランダムに決める
        # はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
        top = np.random.randint(0 - mask_size // 2, height - mask_size)
        left = np.random.randint(0 - mask_size // 2, width - mask_size)
        bottom = top + mask_size
        right = left + mask_size

        # マスク部分が元に画像からはみ出した場合の処理
        if top < 0:
            top = 0
        if left < 0:
            left = 0

        # マスク部分の画素値を平均値で埋める
        image_data[top:bottom, left:right, :].fill(mask_value)
        return image_data

    @staticmethod
    def random_erasing(original_image_data, s=(0.02, 0.4), r=(0.3, 3)):
        """
        画像の一部分を消去する
        :param   original_image_data: numpy array 形式の画像データ
        :param   s:                   消去部分の範囲を決めるための乱数
        :param   r:                   マスクのアスペクト比
        :return: image_data:          numpy array 形式の一部分消去後の画像データ
        """
        # 元画像のデータをコピー　
        image_data = np.copy(original_image_data)

        # マスクする画素値をランダムで決める
        mask_value = np.random.randint(0, 256)

        # 画像の高さと幅を計算
        height, width, _ = image_data.shape

        # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
        mask_area = np.random.randint(height * width * s[0], height * width * s[1])

        # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
        mask_aspect_ratio = np.random.rand() * r[1] + r[0]

        # マスクのサイズとアスペクト比からマスクの高さと幅を決める
        # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
        mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
        if mask_height > height - 1:
            mask_height = height - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > width - 1:
            mask_width = width - 1

        # 元画像から一部分を削除
        top = np.random.randint(0, height - mask_height)
        left = np.random.randint(0, width - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        image_data[top:bottom, left:right, :].fill(mask_value)
        return image_data
