import sys
import os

# !pip install mesa
# !pip install imageio
# !pip install pathfinding

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import imageio.v2 as imageio
from IPython.display import Image

import pandas as pd
import numpy as np
import random
import time
import pickle
import sys
import csv

BASE_DIR = '/Users/miusumino/Documents/KOKUYO/project2/kokuyo_mas'


# simulation params
SIM_MINUTES = 5  # シミュレーションを実行する時間（単位：分）
MINUTE = 120 # １分120ステップ
num_agent = 35  # シミュレーションに参加するエージェントの総数
seed = 20  # seed

# destination information and time threshold

DESTINATIONS = ['west outside', 'executive', 'open and session', 'touchdown', 'concierge', 'magnet', 'freshers', 'community', 'refresh', 'meeting 1', 'copyspace', 'meeting 2', 'east outside']
NUMBERS_ON_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # 目的地に割り当てられたマップ上の数字
WEIGHTS = [108, 42, 328, 398, 188, 27, 258, 342, 261, 39, 29, 37, 20] # エージェントがそれぞれの目的地を目指す比率
WEIGHTS = [1, 1, 1, 1, 1, 100, 1, 1, 1, 1, 1, 1, 1] # エージェントがそれぞれの目的地を目指す比率
# エージェントが目的地に滞在する時間の閾値。この時間を超えると、エージェントは出発地に戻る
DEST_TH = [(30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (300, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential'), (30, 'exponential')] 
# エージェントが出発地に滞在する時間の閾値。この時間を超えると、エージェントは移動を開始する可能性がある
HOME_TH = [None, (600, 'exponential'), (600, 'exponential'), (600, 'exponential'), (600, 'exponential'), None, (600, 'exponential'), (600, 'exponential'), (600, 'exponential'), (600, 'exponential'), None, (600, 'exponential'), None]
SIT_TH = 100

# communication params
COMM_SCORE = 0  # エージェントがコミュニケーションを行った際に得られるスコアの量
COMM_RATIO = 0  # エージェントが隣接エージェントとコミュニケーションを開始する確率
COMM_TIME_MIN = 5  # コミュニケーションの最小持続時間（ステップ数）
COMM_TIME_MAX = 10  # コミュニケーションの最大持続時間（ステップ数）
COOL_TIME = 10 # エージェントがコミュニケーション後に次のコミュニケーションを開始するまでのクールダウン時間（ステップ数）

# output path
out_parent = 'content-long'
out_dir = f'{out_parent}/result'  # 結果を保存するディレクトリ
log_file = f'{out_parent}/log.txt'
gif_file = 'simulation.gif'  # アニメーションのファイル名

# map
map_path = os.path.join(BASE_DIR, 'map/kokuyo_map.csv')  # シミュレーションで使用するアクセシビリティマップのCSVファイルのパス
cost_map_path = os.path.join(BASE_DIR, 'map/kokuyo_cost_map_0403.csv') # シミュレーションで使用するコストを付与したマップのCSVファイルのパス
area_map_path = os.path.join(BASE_DIR, 'map/area_map.csv') # シミュレーションで使用するエリア分類を示したmap

class Worker(Agent):
    def __init__(self, unique_id, model, accessibility_map, home_pos, destination_pos_dict):
        """
        エージェントの初期化メソッド

        :param unique_id: エージェントに一意に割り当てられるID
        :param model: エージェントが属するモデル
        :param home_pos: エージェントの出発地の位置
        :param destination_pos: エージェントの目的地の位置
        """
        super().__init__(unique_id, model)
        self.accessibility_map = self.set_accessibility_map(accessibility_map, home_pos)
        self.home_pos = home_pos  # エージェントの出発地の位置
        self.destination_pos_dict = destination_pos_dict  # マップのエリア分類の辞書
        self.pos = self.home_pos # 最初はhome_posで初期化
        self.home_pos_name = self.check_pos_class() # エリア分類を取得
        self.pos_name = self.home_pos_name

        self.dest_stay_thresholds = DEST_TH # 目的地での滞在時間閾値のリスト
        self.home_stay_thresholds = HOME_TH # 自席での滞在時間閾値のリスト
        index = DESTINATIONS.index(self.home_pos_name) 
        self.home_stay_threshold = np.random.exponential(scale=self.home_stay_thresholds[index][0]) # 自席での滞在時間閾値
        self.dest_stay_threshold = None # 目的地での滞在時間閾値(初期値はNone)
        self.path = None  # 目的地までの経路

        # 選択された目的地に対応する座標を取得し、self.current_targetに代入, self.current_target_nameにはスペース名
        keys_list = list(self.destination_pos_dict.keys()) # 目的地名のリスト
        # 出現確率に基づいてランダムに目的地を選択
        self.current_target_name = None
        # 選択された目的地に対応する座標をランダムに選択
        self.current_target = None

        self.prev_is_moving = False # コミュニケーションを始める直前の移動状態
        self.is_moving = False  # エージェントが移動中かどうかのフラグ
        self.communicating = False  # エージェントがコミュニケーション中かどうかのフラグ

        self.communication_partners = []  # コミュニケーションパートナーのリスト
        self.partner_unique_ids = [] # コミュニケーションパートナーのエージェントIDのリスト

        self.communication_time = 0  # コミュニケーションの残り時間
        self.cooldown = 0  # コミュニケーション後のクールダウン時間
        if self.home_stay_threshold > 1:
            self.stay_time = random.randint(0, int(self.home_stay_threshold)-1)  # 現在の滞在時間をランダムに設定
        else:
            self.stay_time = 0  # デフォルト値を設定
        self.stay_time_over = False # その場所での滞在を終了したときにTrue


    def set_accessibility_map(self, accessibility_map, home_pos):
        # accessibility_mapのディープコピーを作成
        my_map = np.copy(accessibility_map)

        # accessibility_mapの値が2であるすべてのセルをFalseに設定
        my_map[my_map == 2] = False

        # home_posの位置のセルをTrueに設定
        my_map[home_pos[1], home_pos[0]] = True

        return my_map


    def step(self):
        """
        エージェントの1ステップごとの行動を定義するメソッド
        このメソッドは、モデルの各ステップで自動的に呼び出される
        """
        # もしマグネットスペースにいたら、マグネットスペースの滞在状況を更新
        if self.pos_name == 'magnet':
            self.magnet_space_count()

        # エージェントがコミュニケーションのクールダウン中であれば、クールダウン時間をデクリメントする
        if self.cooldown > 0:
            self.cooldown -= 1  # クールダウン時間の減少

        # エージェントが現在コミュニケーション中であれば、コミュニケーション処理を実行
        if self.communicating:
            self.communicate()
        
        # エージェントがコミュニケーション中でなく、クールダウンも終了、その場での滞在時間が満了していないなら新たなコミュニケーションの可能性をチェック
        elif self.cooldown == 0 and self.stay_time_over == False:
            self.check_for_communication()

        # エージェントが移動状態にあり、かつコミュニケーション中でなければ、移動処理を実行
        if self.is_moving == True and self.communicating == False:
            self.move()
        # それ以外の場合（移動中でない、またはコミュニケーション中である場合）は、滞在処理を実行
        else:
            self.stay()
        

    def check_pos_class(self):
        """
        エージェントが今どこにいるかエリア分類で記録
        'west outside', 'executive', 'open and session', 
        'touchdown', 'concierge', 'magnet', 'freshers', 
        'community', 'refresh', 'meeting 1', 'copyspace', 
        'meeting 2', 'east outside'
        当てはまらない場合'other'
        """

        flag = 0
        for key, pos_list in self.destination_pos_dict.items():
            if self.pos in pos_list:
                pos_name = key
                flag = 1
                break
        if flag == 0:
            pos_name = 'other'
        return pos_name



    def magnet_space_count(self):
        """
        マグネットスペースの滞在状況を記録するメゾッド
        """
        self.model.stay_magnet[self.model.count].append(self.unique_id)

        
    def communicate(self):
        """
        エージェントがコミュニケーション中の処理を行うメソッド
        コミュニケーションの持続時間を減少させ、時間が経過したらコミュニケーションを終了する
        """
      
        # コミュニケーションの残り時間をデクリメント
        self.communication_time -= 1

        # コミュニケーション時間が終了したら、コミュニケーション終了処理を呼び出す
        if self.communication_time <= 0:
            self.end_communication()


    def end_communication(self):
        """
        エージェントとそのコミュニケーションパートナーのコミュニケーションを終了し、状態を更新するメソッド
        """
        # エージェントのコミュニケーション状態を終了し、移動状態に遷移
        self.communicating = False
        # エージェントの位置が目的地でも自分の席でもない場合は移動状態に遷移し、滞在時間を0にする
        if self.prev_is_moving == True: 
            self.is_moving = True  # コミュニケーション終了後、移動を再開
            # self.stay_time = 0  # 滞在時間リセット

        # コミュニケーション後のクールダウン時間を設定
        self.cooldown = COOL_TIME

        # コミュニケーションパートナーの状態も同様に更新
        for partner in self.communication_partners:
            partner.communication_partners.remove(self)  # パートナーのコミュニケーション相手リストから自分を削除
            partner.partner_unique_ids.remove(self.unique_id)
            # パートナーのコミュニケーション相手が0になった場合
            if partner.communication_partners == []:
                partner.communicating = False  # パートナーのコミュニケーション状態を終了
                # エージェントの位置が目的地でも自分の席でもない場合は移動状態に遷移し、滞在時間を0にする
                if partner.prev_is_moving == True:
                    partner.is_moving = True  # 相手も移動を再開
                    # partner.stay_time = 0  # 滞在時間リセット
                partner.cooldown = COOL_TIME  # 相手にクールダウンを設定
        self.communication_partners = []  # 自分のリストを空にする
        self.partner_unique_ids = []


    def check_for_communication(self):
        """
        エージェントが近隣のエージェントとコミュニケーションを開始するかどうかを確認し、
        条件を満たす場合はコミュニケーションを開始するメソッド。
        """
        # エージェントの近隣にあるエージェントを取得（ムーア近傍）
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
    
        # 近隣のエージェントをループしてコミュニケーション可能なエージェントを探す
        for neighbor in neighbors:
            # 近隣のエージェントがWorkerクラスのインスタンスであることを確認
            if isinstance(neighbor, Worker):
                # クールダウン時間が0であることと、その場での滞在時間を満了していないことを確認
                if neighbor.cooldown == 0 and neighbor.stay_time_over == False:
                    # 一定の確率（COMM_RATIO）でコミュニケーションを開始
                    if random.random() < COMM_RATIO:
                        self.start_communication(neighbor)
                        break  # 一度コミュニケーションを開始したらループを抜ける


    def start_communication(self, partner):
        """
        このエージェントと指定されたパートナーエージェントとの間でコミュニケーションを開始する

        :param partner: コミュニケーションを開始するパートナーエージェント
        """
        # このエージェントの状態をコミュニケーション中に更新
        self.communicating = True
        self.prev_is_moving = self.is_moving # 前の状態を保持
        self.is_moving = False  # コミュニケーション中は移動を停止
        self.communication_partners.append(partner)  # コミュニケーションのパートナーをリストに追加
        self.partner_unique_ids = [partner.unique_id for partner in self.communication_partners]
        # コミュニケーションの持続時間をCOMM_TIME_MINとCOMM_TIME_MAXの間でランダムに設定
        self.communication_time = random.randint(COMM_TIME_MIN, COMM_TIME_MAX)

        # コミュニケーションのパートナーエージェントの状態も更新
        if partner.communicating == False: # パートナーがもともとコミュニケーションをしていなかった場合
            partner.prev_is_moving = partner.is_moving # 前の状態を保持
        partner.communicating = True
        partner.is_moving = False  # パートナーもコミュニケーション中は移動を停止
        partner.communication_partners.append(self)  # パートナーのコミュニケーション相手リストに自分を追加
        partner.partner_unique_ids = [partner.unique_id for partner in partner.communication_partners]
        partner.communication_time = self.communication_time  # 両エージェントで同じコミュニケーション時間を共有
        

    def stay(self):
        """
        エージェントが現在地で滞在する際の処理を行うメソッド
        滞在時間に応じてスコアを加算し、特定の条件下で移動状態に遷移する
        """
        # 滞在時間をインクリメント
        self.stay_time += 1

        # エージェントが出発地(自席)に滞在している場合
        if self.pos == self.home_pos:
            # 滞在時間が出発地での滞在閾値を超えた場合
            # print(self.stay_time, self.dest_stay_threshold, self.unique_id, self.pos_name, self.pos == self.home_pos)
            if self.stay_time >= self.home_stay_threshold:
                # print("!!!!!!")
                # 閾値を超えた分の時間に基づいて移動遷移の確率を計算
                extra_time = self.stay_time - self.home_stay_threshold
                # extra_timeが大きくなるほど遷移確率が高くなるが、最大で1.0
                self.transition_probability = min(1.0, 0.1 * extra_time)
                # 確率的に移動状態に遷移
                if random.random() < self.transition_probability:
                    self.home_stay_threshold = None
                    # 目的地を更新してルート計算
                    while self.path == [] or self.path == None:
                        self.current_target, self.current_target_name = self.get_destination()
                        self.calculate_route()
                    # もし滞在時間を超えてコミュニケーションをしていたらコミュニケーションを終了
                    if self.communicating == True:
                        self.end_communication()
                    self.is_moving = True # 移動状態に遷移
                    self.stay_time_over = True # その場所での滞在時間を満了
                    # self.stay_time = 0  # 滞在時間をリセット

        # エージェントが目的地に滞在している場合
        elif self.pos_name in DESTINATIONS:                
            # 滞在時間が目的地での滞在閾値を超えた場合
            # print(self.stay_time, self.dest_stay_threshold, self.unique_id, self.pos_name, self.pos == self.home_pos)
            if self.stay_time >= self.dest_stay_threshold:  # 確率的な処理にしてもよい
                # print("!!!!!!")
                # 目的地を更新してルート計算
                while self.path == [] or self.path == None:
                    # print(self.unique_id, self.dest_stay_threshold)
                    self.current_target, self.current_target_name = self.get_destination()
                    self.calculate_route()
                # もし滞在時間を超えてコミュニケーションをしていたらコミュニケーションを終了
                if self.communicating == True:
                    self.end_communication()
                self.is_moving = True # 移動状態に遷移
                self.stay_time_over = True # その場所での滞在時間を満了


    def move(self):
        """
        エージェントが移動する際の処理を実行するメソッド。
        エージェントは目的地に向けて移動し、到遀後は出発地に戻る。
        他のエージェントがいる座標には移動しないようにする。
        """
        # if not self.path:  # ルートが未計算または終了した場合、または再計算が必要な場合
        #     self.calculate_route()

        # while self.path:
            # # 次の位置に他のエージェントがいないか確認
            # if self.model.is_cell_empty(next_pos):
            #     self.path.pop(0)  # 実際に移動する場合にのみ、次の位置をパスから削除
            #     self.model.move_agent(self, next_pos)  # エージェントを次の位置に移動

            #     if not self.path:  # 目的地に到達した場合、またはパスが空になった場合
            #         self.is_moving = False
            #         self.stay_time = 0
            #     break  # 移動した後はループから抜ける
            # else:
            #     # 次の位置に他のエージェントがいる場合、パスを再計算
            #     self.calculate_route()

        # print(self.pos, self.pos_name, self.current_target, self.current_target_name)

        next_pos = self.path[0]  # 次の位置を取得するが、まだpopしない
        self.path.pop(0)  # 実際に移動する場合にのみ、次の位置をパスから削除
        prev_pos_name = self.pos_name # 移動する直前の位置を記録
        self.model.grid.move_agent(self, next_pos)  # エージェントを次の位置に移動
        # 今どこにいるか分類で記録
        self.pos_name = self.check_pos_class()
        # 直前と場所が変わっていたら滞在時間を初期化
        if self.pos_name != prev_pos_name:
            self.stay_time = 0
            self.stay_time_over = False

        if not self.path:  # 目的地に到達した場合、またはパスが空になった場合
            self.is_moving = False
            if self.pos != self.current_target:
                print(self.model.count, self.unique_id, self.pos_name, self.pos) #エラー検知用
            # self.stay_time = 0


    def get_destination(self):
        """
        エージェントの次の目的地を取得するメソッド。
        Workerクラスでは、現在目的地にいる場合は、常に自席の位置を返します。
        """
        # 目的地名のリスト
        keys_list = list(self.destination_pos_dict.keys())

        if self.pos == self.home_pos:
            # 滞在時間のしきい値を再設定
            index = DESTINATIONS.index(destination_name) 
            if self.dest_stay_thresholds[index][1] == 'exponential':
                self.dest_stay_threshold = np.random.exponential(scale=self.dest_stay_thresholds[index][0])
            else:
                self.dest_stay_threshold = random.uniform(self.dest_stay_thresholds[index][0], self.dest_stay_thresholds[index][1])
            # 出現確率に基づいてランダムに目的地を選択
            destination_name = random.choices(keys_list, weights=WEIGHTS)[0]
            # 選択された目的地に対応する座標をランダムに選択
            destination_coordinate = random.choice(self.destination_pos_dict[destination_name])
            return destination_coordinate, destination_name
        else:
            index = DESTINATIONS.index(self.home_pos_name) 
            self.home_stay_threshold = np.random.exponential(scale=self.home_stay_thresholds[index][0])
            return self.home_pos, self.home_pos_name
        

    def calculate_route(self):
        """
        エージェントが目的地または出発地に向けて移動する際の最短ルートを計算する。
        A*アルゴリズムを使用してルートを見つける。
        """

        # アクセシビリティマップのコピーを作成（動的な変更を適用するため）
        dynamic_map = np.copy(self.accessibility_map.T)  # xyを転置

        # 他のエージェントの位置を動的マップに反映（エージェントがいる場所は移動不可とする）
        for agent in self.model.schedule.agents:
            if agent.pos != self.pos:  # 自分自身の位置は除外
                dynamic_map[agent.pos[0], agent.pos[1]] = False  # 移動不可とする

        grid = Grid(matrix=dynamic_map)

        start_node = grid.node(self.pos[1], self.pos[0])  # 座標を反転して渡す
        end_node = grid.node(self.current_target[1], self.current_target[0])  # 座標

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, _ = finder.find_path(start_node, end_node, grid)

        if path and len(path) > 1:
            # グリッドノードの座標をタプルに変換して保存
            self.path = [(step.y, step.x) for step in path[1:]]  # xとyを転置して保存
            #self.plot_path(self.path, self.pos, self.current_target)  # 経路の確認用
        else:
            self.path = []  # 移動可能な経路が見つからない場合


    def plot_path(self, path, start_pos, end_pos):
        """
        アクセシビリティマップとA*アルゴリズムによって見つかった経路を可視化する

        :param path: A*アルゴリズムによって見つかった経路（セルのリスト）
        :param start_pos: 開始位置の座標
        :param end_pos: 終了位置の座標
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.accessibility_map, cmap='gray_r')

        # 経路上のセルをプロット
        for (x, y) in path:
            ax.plot(x, y, 'bo')  # 青い点で経路を表示

        # 開始位置と終了位置をプロット
        ax.plot(start_pos[0], start_pos[1], 'go')  # 緑色の点で開始位置を表示
        ax.plot(end_pos[0], end_pos[1], 'ro')  # 赤色の点で終了位置を表示

        plt.show()

    def set_color(self):
        """
        エージェントの現在の状態に応じて色を設定するメソッド

        :return: エージェントの状態を表す色の文字列
                コミュニケーション中は赤('red')、移動中は青('blue')、それ以外（滞在中）はグレー('gray')を返す
        """
        if self.communicating:
            # エージェントがコミュニケーション中の場合、赤
            return 'red'
        elif self.is_moving:
            # エージェントが移動中の場合、青
            return 'blue'
        else:
            # エージェントが移動中でもコミュニケーション中でもない場合（滞在中を含む）、グレー
            return 'gray'


class MigratoryWorker(Worker):
    def __init__(self, unique_id, model, accessibility_map, home_pos, destination_pos_dict, home_probability=0.68):
        """
        MigratoryWorkerクラスのコンストラクタ。

        :param unique_id: エージェントに一意に割り当てられるID。
        :param model: エージェントが属するモデル。
        :param accessibility_map: エージェントが移動する環境のマップ。
        :param home_pos: エージェントの出発地（または帰宅地）の位置。
        :param destination_pos_dict: エージェントが訪れる可能性のある目的地のリスト。
        :param home_probability: 出発地に帰る確率。
        """
        super().__init__(unique_id, model, accessibility_map, home_pos, destination_pos_dict)
        self.home_probability = home_probability


    def get_cost_map(self):
        # CSVファイルを読み込み、Pandas DataFrameとして格納
        df = pd.read_csv(cost_map_path, header=None)
        cost_map = df.values.T

        # コストマップを返す
        return cost_map


    def calculate_route(self):
        """
        エージェントが目的地または出発地に向けて移動する際の最短ルートを計算する。
        A*アルゴリズムを使用してルートを見つける。
        """
        # dynamic_map = np.copy(self.accessibility_map.T)
        # for agent in self.model.schedule.agents:
        #     if agent.pos != self.pos:
        #         dynamic_map[agent.pos[0], agent.pos[1]] = False

        cost_map = self.get_cost_map()

        # grid = Grid(matrix=dynamic_map)
        grid = Grid(matrix=self.accessibility_map.T)

        # A*アルゴリズムで経路探索
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        # 各セルのコストを設定
        for y, row in enumerate(cost_map):
            for x, cost in enumerate(row):
                grid.node(x, y).weight = cost

        start_node = grid.node(self.pos[1], self.pos[0])
        end_node = grid.node(self.current_target[1], self.current_target[0])

        path, _ = finder.find_path(start_node, end_node, grid)

        if path and len(path) > 1:
            self.path = [(step.y, step.x) for step in path[1:]]
        else:
            self.path = []
            print(f"[Warning] Agent {self.unique_id}: No path found from {self.pos} to {self.current_target}")

        

    def get_destination(self):
        """
        帰宅確率に基づき、自席の位置またはdestination_pos_dictからランダムに選択された目的地を返します。
        """
        # 目的地名のリスト
        keys_list = list(self.destination_pos_dict.keys())
        dynamic_map = np.copy(self.accessibility_map.T)
        for agent in self.model.schedule.agents:
            if agent.current_target != None and agent.current_target != self.current_target:
                dynamic_map[agent.current_target[0], agent.current_target[1]] = False

        destination_coordinate = None
        
        # 現在地が自席の場合
        if self.pos == self.home_pos:
            while destination_coordinate == None:
                # 出現確率に基づいてランダムに目的地を選択
                destination_name = random.choices(keys_list, weights=WEIGHTS)[0]
                # 滞在時間のしきい値を再設定
                index = DESTINATIONS.index(destination_name) 
                # 0702ADD
                # mean = self.dest_stay_thresholds[index][0]
                if self.dest_stay_thresholds[index][1] == 'exponential':
                    # # 10を平均とする指数分布
                    # short_stay = np.random.exponential(scale=10)
                    # # 2*mean - 10を平均とする指数分布
                    # normal_stay = np.random.exponential(scale=(3*mean - 10)/2)
                    # # どちらかをランダムに選択
                    # if np.random.rand() < (1/3):
                    #     self.dest_stay_threshold = short_stay
                    # else:
                    #     self.dest_stay_threshold = normal_stay
                    # # 0702ADD
                    self.dest_stay_threshold = np.random.exponential(scale=self.dest_stay_thresholds[index][0])
                else:
                    self.dest_stay_threshold = random.uniform(self.dest_stay_thresholds[index][0], self.dest_stay_thresholds[index][1])
                
                # 滞在時間が10分を超える場合は座席に座らせる
                if self.dest_stay_threshold > SIT_TH: 
                    matching_positions = [pos for pos in self.model.home_pos_list if pos in self.destination_pos_dict.get(destination_name, [])]
                    filtered_accessible_home_pos_list = [home_pos for home_pos in matching_positions if dynamic_map[home_pos[0], home_pos[1]]]
                    # print(filtered_accessible_home_pos_list, dest_stay_threshold, destination_name)
                    if filtered_accessible_home_pos_list == []:
                        destination_coordinate = random.choice(self.destination_pos_dict[destination_name])
                    else:
                        destination_coordinate = random.choice(filtered_accessible_home_pos_list)
                else:
                    destination_coordinate = random.choice(self.destination_pos_dict[destination_name])
        
        # 現在地が目的地の場合
        else:
            # 帰宅確率に基づいて自席の位置を返すか判断
            if random.random() < self.home_probability:
                index = DESTINATIONS.index(self.home_pos_name) 
                self.home_stay_threshold = np.random.exponential(scale=self.home_stay_thresholds[index][0])
                return self.home_pos, self.home_pos_name
            else:
                filtered_keys_list = [key for key in keys_list if key != self.pos_name and key != self.home_pos_name]
                # weightsから同じindexの要素を削除した新しいweightsリストを作成
                filtered_weights = [weight for i, weight in enumerate(WEIGHTS) if keys_list[i] != self.pos_name and keys_list[i] != self.home_pos_name]
                while destination_coordinate == None:
                    # destination_pos_dictから現在地以外の目的地をランダムに選択
                    destination_name = random.choices(filtered_keys_list, weights=filtered_weights)[0]
                    # 滞在時間のしきい値を再設定
                    index = DESTINATIONS.index(destination_name)
                    # 0702ADD
                    # mean = self.dest_stay_thresholds[index][0]
                    if self.dest_stay_thresholds[index][1] == 'exponential':
                        # # 10を平均とする指数分布
                        # short_stay = np.random.exponential(scale=10)
                        # # 2*mean - 10を平均とする指数分布
                        # normal_stay = np.random.exponential(scale=(3*mean - 10)/2)
                        # # どちらかをランダムに選択
                        # if np.random.rand() < (1/3):
                        #     self.dest_stay_threshold = short_stay
                        # else:
                        #     self.dest_stay_threshold = normal_stay
                        # 0702ADD
                        self.dest_stay_threshold = np.random.exponential(scale=self.dest_stay_thresholds[index][0])
                    else:
                        self.dest_stay_threshold = random.uniform(self.dest_stay_thresholds[index][0], self.dest_stay_thresholds[index][1])
                    
                    # 滞在時間が10分を超える場合は座席に座らせる
                    if self.dest_stay_threshold > SIT_TH: 
                        matching_positions = [pos for pos in self.model.home_pos_list if pos in self.destination_pos_dict.get(destination_name, [])]
                        filtered_accessible_home_pos_list = [home_pos for home_pos in matching_positions if dynamic_map[home_pos[0], home_pos[1]]]
                        # print(filtered_accessible_home_pos_list, dest_stay_threshold, destination_name)
                        if filtered_accessible_home_pos_list == []:
                            destination_coordinate = random.choice(self.destination_pos_dict[destination_name])
                        else:
                            destination_coordinate = random.choice(filtered_accessible_home_pos_list)
                    else:
                        destination_coordinate = random.choice(self.destination_pos_dict[destination_name])
        return destination_coordinate, destination_name


class Floor(Model):
    def __init__(self, N, map_file, out_dir, seed=None):
        """
        Floorクラスのコンストラクタ。シミュレーション環境を初期化する

        :param N: 生成するエージェントの数
        :param map_file: アクセシビリティマップを含むファイルのパス。このマップは移動可能な場所を定義する
        :param out_dir: シミュレーション結果（画像ファイルなど）を保存するディレクトリのパス
        """
        super().__init__()
        # アクセシビリティマップを読み込み、シミュレーションのグリッドサイズを決定
        self.accessibility_map, width, height = self.load_accessibility_map(map_file)
        self.num_agents = N  # エージェントの数を設定
        self.grid = MultiGrid(width, height, True)  # シミュレーションのグリッドを初期化, (True: 複数のエージェントが同じセルに存在することができる)
        self.schedule = RandomActivation(self)  # エージェントのアクティベーションスケジュールを初期化
        
        # すべての目的地を辞書型に格納
        self.destination_pos_dict = {destination: self.get_positions(area_map_path, number) for destination, number in zip(DESTINATIONS, NUMBERS_ON_MAP)}
        self.home_pos_list = self.get_positions(map_file, 2) # ホームポジションとなり得る座標の取得
        
        self.seed = seed
        if self.seed is not None:
            self.set_random_seed(self.seed)

        # 出力ディレクトリが存在しない場合は作成
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # エージェントの初期化と配置
        home_pos_tmp = self.home_pos_list # すでに他のエージェントに使用されている椅子の削除用にコピー
        for i in range(self.num_agents):
            while True:
                # エージェントのホームポジションをランダムに選択
                home_pos = random.choice(home_pos_tmp) # ランダムに椅子から選ぶ
                home_pos_tmp.remove(home_pos) # すでに他のエージェントに使用されている椅子を削除

                # 選択したホームポジションが移動可能な場所であれば、エージェントを配置
                if self.accessibility_map[home_pos[1], home_pos[0]]:
                    agent = MigratoryWorker(i, self, self.accessibility_map, home_pos, self.destination_pos_dict)
                    self.grid.place_agent(agent, home_pos)
                    self.schedule.add(agent)
                    break  # 移動可能な場所にエージェントを配置できたらループを抜ける

    def set_random_seed(self, seed):
        """
        乱数生成器のseedを設定します。

        :param seed: 乱数生成のためのseed値
        """
        random.seed(seed)  # Python標準の乱数生成器のseedを設定
        np.random.seed(seed)  # NumPyの乱数生成器のseedを設定


    def get_positions(self, csv_file, type):
        """
        特徴を持つ座席位置の取得
        """
        # CSVファイルを読み込み、Pandas DataFrameとして格納
        df = pd.read_csv(csv_file, header=None)
        # CSVファイルに2を見つけたら座標をリストに格納
        pos_list = []  # 座標を格納するリスト
        for index, row in df.iterrows():
            for column in df.columns:
                if df.at[index, column] == type:
                    pos_list.append((column, index))
        return pos_list


    def load_accessibility_map(self, csv_file):
        """
        CSVファイルからアクセシビリティマップを読み込み、移動可能な場所を示すブール値の配列を返すメソッド

        :param csv_file: アクセシビリティマップが格納されているCSVファイルのパス
        :return: アクセシビリティマップ（ブール値の配列）、マップの幅、マップの高さ
        """
        # CSVファイルを読み込み、Pandas DataFrameとして格納
        df = pd.read_csv(csv_file, header=None)

        # CSVファイルに1,0以外の値があるときは1に変換
        df = df.applymap(lambda x: 1 if x != 1 and x != 0 else x)

        # DataFrameの値をブール型に変換し、アクセシビリティマップとして使用
        # bool型のオブジェクトTrueとFalseで表される
        # Trueは移動可能なセル、Falseは移動不可能なセルを示す
        accessibility_map = df.values
        #accessibility_map = df.values.astype(bool)

        # マップの高さと幅を取得
        height, width = df.shape # 行数・列数を取得: df.shape

        # アクセシビリティマップ、マップの幅、マップの高さを返す
        return accessibility_map, width, height

    def step(self):
        """
        シミュレーションの1ステップを進めるメソッド
        データ収集と全エージェントの行動更新を行う
        """
        # データ収集を実行
        # self.datacollector.collect(self)
        # 全エージェントの1ステップの行動を実行
        self.schedule.step()


    def is_cell_empty(self, pos):
        """
        指定された位置のセルが空いているかどうかを確認するメソッド。

        :param pos: 確認するセルの座標（x, y）のタプル。
        :return: セルが空いている場合はTrue、そうでない場合はFalse。
        """
        # 指定された位置のセルに含まれるコンテンツ（エージェントなど）のリストを取得
        contents = self.grid.get_cell_list_contents([pos])
        # コンテンツのリストが空（= セルが空いている）かどうかを返す
        return len(contents) == 0


    def export_agents_to_csv(self, step):
        """
        シミュレーションの各ステップでエージェントの位置情報をCSVファイルにエクスポートします。

        :param step: 現在のシミュレーションステップ
        """
        # エージェントのデータを収集するリスト
        agent_data = []

        # シミュレーション内の全エージェントの情報をループで処理
        for agent in self.schedule.agents:
            # エージェントのIDと位置情報を収集
            agent_info = {
                'AgentID': agent.unique_id,  # エージェントの一意識別子
                'Home': agent.home_pos,
                'X': agent.pos[0],  # エージェントのX座標
                'Y': agent.pos[1],   # エージェントのY座標
                'Pos_name': agent.pos_name,
                'Goal': agent.current_target,  # エージェントの次の移動先
                'Goal_name':agent.current_target_name,
                # 'Path': agent.path,
                'Moving': agent.is_moving,  # 移動中フラグ
                'Communicating': agent.communicating,  # コミュニケーションフラグ
                'Communication_pertners': agent.partner_unique_ids, # コミュニケーションパートナーたちのエージェントID
                'Home_stayth': agent.home_stay_threshold,
                'Dest_stayth': agent.dest_stay_threshold,
                'Staytime': agent.stay_time
            }
            agent_data.append(agent_info)

        # 収集したデータをPandasのDataFrameに変換
        df = pd.DataFrame(agent_data)

        # CSVファイルに書き出し
        # 出力ファイル名はステップ数を含む形式で、指定した出力ディレクトリに保存
        out_file = self.get_file_path(os.path.join(self.out_dir, 'agent_positions'), f'step_{step:03}.csv')
        df.to_csv(out_file, index=False)


    def plot_map(self, step):
        """
        シミュレーションのグリッド状態を描画し、指定されたステップ番号で画像を保存するメソッド

        :param step: 現在のステップ番号
        """
        fig, ax = plt.subplots()
        # アクセシビリティマップを背景として描画（移動可能な場所を示す）
        ax.imshow(self.accessibility_map, cmap='gray_r', extent=(0, self.grid.width, self.grid.height, 0))

        # 各エージェントのパスを描画
        for agent in self.schedule.agents:
            # パスが存在する場合のみ描画
            if hasattr(agent, 'path') and agent.path:
                # パス上のポイントを描画
                for (x, y) in agent.path:
                    ax.plot(x + 0.5, y + 0.5, color='cyan', marker='.', markersize=5)  # パスをシアン色の点で表示

        # 各エージェントの位置を描画
        for agent in self.schedule.agents:
            color = agent.set_color()  # エージェントの状態に応じた色を取得
            # エージェントの位置に正方形を描画（マスの中心に配置）
            ax.scatter(agent.pos[0] + 0.5, agent.pos[1] + 0.5, color=color, s=100, marker="s")
            # エージェントのIDをテキストとして表示
            ax.text(agent.pos[0] + 0.5, agent.pos[1] + 0.5, str(agent.unique_id), color='white', ha='center', va='center')
            # 目的地（self.magnet_pos）の座標を黄色く描画

            if agent.current_target != None and agent.current_target != agent.home_pos:
                ax.scatter(agent.current_target[0] + 0.5, agent.current_target[1] + 0.5, color='yellow', s=100, marker="o", edgecolors='black', linewidth=1)

        # グリッド線を描画
        ax.set_xticks(np.arange(0, self.grid.width, 1))
        ax.set_yticks(np.arange(0, self.grid.height, 1))

        # X軸とY軸の値の範囲を設定
        x_labels = [str(i) for i in range(self.grid.width)]
        y_labels = [str(i) for i in range(self.grid.height)]

        ax.set_xticks(np.arange(0, self.grid.width, 1))
        ax.set_yticks(np.arange(0, self.grid.height, 1))
        ax.grid(which='both', color='black', linestyle='-', linewidth=1)

        ax.set_xlim([0, self.grid.width])
        ax.set_ylim([self.grid.height, 0])
        ax.set_title(f"Step {step}")

        # ステップ番号をファイル名に含めて画像を保存
        out_file = self.get_file_path(os.path.join(self.out_dir, 'step_map'), f'step_{step:03}.png')
        plt.savefig(out_file)
        plt.close()  # リソースを解放


    def get_file_path(self, dir_path, file_name):
        """
        指定されたディレクトリパスとファイル名を組み合わせて、フルパスを生成します。

        :param dir_path: ディレクトリのパス
        :param file_name: ファイル名
        :return: 組み合わせたフルパス
        """
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return os.path.join(dir_path, file_name)
    
# モデルのセーブ
def save_model_state(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# モデルの読み込み
def load_model_state(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def get_object_size(obj):
    serialized_obj = pickle.dumps(obj)
    size_bytes = sys.getsizeof(serialized_obj)
    return size_bytes


def simulation(minutes_now):
    # Initialize model with specified parameters
    if minutes_now == 0:
        model = Floor(N=num_agent, map_file=map_path, out_dir=out_dir, seed=seed)
        model.stay_magnet = [[] for _ in range(MINUTE*SIM_MINUTES)] # マグネットスペースの滞在状況をいれるリスト(内部リストの数はSIM_TIMEと等しい)
    else:
        model = load_model_state(f"{out_parent}/saved_model.pkl")

    # previous_model のサイズを取得 #0428ADD
    model_size = get_object_size(model)
    print("previous_model のデータサイズ（バイト）:", model_size)

    # Run the simulation for a specified number of steps
    for i in range(minutes_now*MINUTE, minutes_now*MINUTE+MINUTE):
        model.count = i # 今何ステップ目か
        with open(log_file, "w") as file:
            file.write(f"現在{i}ステップ目を実行中")
        model.step()  # Execute one step of the model
        model.export_agents_to_csv(i)
        model.plot_map(i)  # Plot the state of the grid at the current step
    
    save_model_state(model, f"{out_parent}/saved_model.pkl")
    del model  # モデルを破棄


if __name__ == "__main__": 
    # タイマー開始
    start_time = time.time()
    for minutes_now in range(SIM_MINUTES):
        simulation(minutes_now)
    # タイマー終了
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{SIM_MINUTES}分のシミュレーションの実行時間: ", execution_time, "秒")
    model = load_model_state(f"{out_parent}/saved_model.pkl")




