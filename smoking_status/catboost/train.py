import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

train_df = pd.read_csv("./smoking_status/data/train.csv")
test_df = pd.read_csv("./smoking_status/data/test.csv")

x = train_df.drop("smoking", axis=1)
y = train_df["smoking"]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

text_features = list(x)
print(text_features)

clf = CatBoostClassifier(
    learning_rate=0.05,
    iterations=1000,
    # depth=6,
    early_stopping_rounds=50,
    eval_metric='Accuracy'
)

clf.fit(x_train, y_train,  eval_set=(x_val, y_val))

result = clf.predict(test_df)

output = pd.DataFrame({
    "id": test_df["id"],
    "smoking": result,
})

output.to_csv("./smoking_status/submission.csv", index=False)