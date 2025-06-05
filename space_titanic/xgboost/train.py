from ..datasets import TitanicDataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = TitanicDataset()
x_list = []
y_list = []
for i in range(len(dataset)):
    x, y = dataset[i]
    x_list.append(x.numpy())
    y_list.append(y)

model = xgb.XGBRegressor(objective = "multi:softmax", num_class=2)
model.fit(x_list, y_list)

test_dataset = TitanicDataset(True)
x_test = []
for i in range(len(test_dataset)):
    x, y = test_dataset[i]
    x_test.append(x.numpy())

predictions = model.predict(x_test)
output_df = pd.DataFrame(columns=["PassengerId", "Transported"])
for i in range(len(x_test)):
    # print(i)
    id = test_dataset.getPassengerId(i)
    transported = predictions[i]
    output_df = pd.concat([output_df, pd.DataFrame({"PassengerId": [id] , "Transported": [transported == 1]})], ignore_index=True)

output_df.to_csv("./submission.csv", index=False)
print(output_df)