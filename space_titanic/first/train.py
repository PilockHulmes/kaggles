from sklearn.ensemble import RandomForestClassifier
from ..datasets import TitanicDataset
import pandas as pd

dataset = TitanicDataset()

x_list = []
y_list = []

for i in range(len(dataset)):
    x, y = dataset[i]
    x_list.append(x.numpy())
    y_list.append(y)

print("start training")

model = RandomForestClassifier()
model.fit(x_list, y_list)

print("finish training")

output_df = pd.DataFrame(columns=["PassengerId", "Transported"])
test_dataset = TitanicDataset(True)
for i in range(len(test_dataset)):
    id = test_dataset.getPassengerId(i)
    features, _ = test_dataset[i]
    predicted = model.predict([features.numpy()])
    print(predicted)
    transported = predicted[0]
    output_df = pd.concat([output_df, pd.DataFrame({"PassengerId": [id] , "Transported": [transported]})], ignore_index=True)
    print(i, id, transported)

print(output_df)