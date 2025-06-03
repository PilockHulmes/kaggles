import csv
with open("./space_titanic/data/train.csv", mode="r") as f:
    csv_file = csv.DictReader(f)
    for lines in csv_file:
        print(lines)

import os
import pandas as pd
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    def __init__(self, is_test = False):
        data_path = "./space_titanic/data/train.csv"
        if is_test:
            data_path = "./space_titanic/data/test.csv"
        with open(data_path, mode="r") as f:
            csv_file = csv.DictReader(f)
            self.lines = list(csv_file)

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, i):
        item = self.lines[i]
        list_item = [
            item["PassengerId"],
            item["HomePlanet"],
            item["CyroSleep"],
            item["Cabin"],
            item["Destination"],
            item["Age"],
            self.str2bool(item["VIP"]),
            item["RoomService"],
            item["FoodCourt"],
            item["ShoppingMall"],
            item["Spa"],
            item["VRDeck"],
            item["Name"]
        ]
        return list_item, self.str2bool(item["Transported"])
        pass

    def get_features(self, line):
        return [
            line["PassengerId"],
            line["HomePlanet"],
            line["CyroSleep"],
            line["Cabin"],
            line["Destination"],
            line["Age"],
            self.str2bool(line["VIP"]),
            line["RoomService"],
            line["FoodCourt"],
            line["ShoppingMall"],
            line["Spa"],
            line["VRDeck"],
            line["Name"]
        ]

    def str2bool(x):
        return x.lower() in ("true")

    def str2num(x):
        return float(x)