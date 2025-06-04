import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torch

df = pd.read_csv("./space_titanic/data/train.csv")
# feature engineering
df["CryoSleep"] = df["CryoSleep"].fillna("False").infer_objects(copy=False)
df["VIP"] = df["VIP"].fillna("False").infer_objects(copy=False)
df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
df = df.join(pd.get_dummies(df['HomePlanet'], prefix='HomePlanet'))
df["Destination"] = df["Destination"].fillna("Unknown")
df = df.join(pd.get_dummies(df['Destination'], prefix='Destination'))
df[["Cabin_0", "Cabin_1", "Cabin_2"]] = df["Cabin"].str.split("/", expand=True)
df["Cabin_0"] = df["Cabin_0"].fillna("")
df["Cabin_1"] = df["Cabin_1"].fillna(0).infer_objects(copy=False)
df["Cabin_2"] = df["Cabin_2"].fillna("")
le = LabelEncoder()
df["Cabin_0_label"] = le.fit_transform(df["Cabin_0"])
df["Cabin_2_label"] = le.fit_transform(df["Cabin_2"])
# df.replace({False: 0, True: 1}, inplace=True)

print(df.iloc[0]["Name"])
print(df.dtypes)

class TitanicDataset(Dataset):
    def __init__(self, is_test = False):
        data_path = "./space_titanic/data/train.csv"
        if is_test:
            data_path = "./space_titanic/data/test.csv"
        df = pd.read_csv(data_path)
        # feature engineering
        df["CryoSleep"] = df["CryoSleep"].fillna("False").infer_objects()
        df["VIP"] = df["VIP"].fillna("False").infer_objects()
        df["HomePlanet"] = df["HomePlanet"].fillna("Unknown")
        df = df.join(pd.get_dummies(df['HomePlanet'], prefix='HomePlanet'))
        df["Destination"] = df["Destination"].fillna("Unknown")
        df = df.join(pd.get_dummies(df['Destination'], prefix='Destination'))
        df[["Cabin_0", "Cabin_1", "Cabin_2"]] = df["Cabin"].str.split("/", expand=True)
        df["Cabin_0"] = df["Cabin_0"].fillna("")
        df["Cabin_1"] = df["Cabin_1"].fillna(0).astype(int)
        df["Cabin_2"] = df["Cabin_2"].fillna("")
        le = LabelEncoder()
        df["Cabin_0_label"] = le.fit_transform(df["Cabin_0"])
        df["Cabin_2_label"] = le.fit_transform(df["Cabin_2"])
        df.replace({False: 0, True: 1}, inplace=True)
        self.df = df

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        list_item = [
            row["CryoSleep"],
            row["Age"],
            row["VIP"],
            row["RoomService"],
            row["FoodCourt"],
            row["ShoppingMall"],
            row["Spa"],
            row["VRDeck"],
            row["HomePlanet_Earth"],
            row["HomePlanet_Mars"],
            row["HomePlanet_Europa"],
            row["HomePlanet_Unknown"],
            row["Destination_55 Cancri e"],
            row["Destination_PSO J318.5-22"],
            row["Destination_TRAPPIST-1e"],
            row["Destination_Unknown"],
            row["Cabin_0_label"],
            row["Cabin_1"],
            row["Cabin_2_label"],
        ]
        return torch.tensor(list_item), self.df["Transported"] == 1