import pandas as pd
import cbm
from sklearn.metrics import mean_squared_error

# load data using https://www.kaggle.com/c/demand-forecasting-kernels-only
train = pd.read_csv("data/kaggle_demand_forecasting/train.csv", parse_dates=["date"])
test = pd.read_csv("data/kaggle_demand_forecasting/test.csv", parse_dates=["date"])

# feature engineering
min_date = train["date"].min()


def featurize(df):
    out = pd.DataFrame(
        {
            # TODO: for prediction such features need separate modelling
            # "seasonal" cannot be added as future periods will have values not seen in training
            # "seasonal": (df["date"] - min_date).dt.days // 60,
            "store": df["store"],
            "item": df["item"],
            "date": df["date"],
            # <name-1> _X_ <name-2> to mark interaction features
            "item_X_month": df["item"].astype(str)
            + "_"
            + df["date"].dt.month.astype(str),
            "store_X_month": df["store"].astype(str)
            + "_"
            + df["date"].dt.month.astype(str),
        }
    )
    return out


store = [10]
item = [1]


# x_train_df = featurize(train.query("store in @store & item in @item"))
# x_test_df = featurize(test.query("store in @store & item in @item"))
# y_train = train.query("store in @store & item in @item")["sales"]

x_train_df = featurize(train.query("item in @item"))
x_test_df = featurize(test.query("item in @item"))
y_train = train.query("item in @item")["sales"]

# model training
model = cbm.CBM()
model.fit(x_train_df, y_train)

# test on train error
y_pred_train = model.predict(x_train_df).flatten()
print("RMSE", mean_squared_error(y_pred_train, y_train, squared=False))

x_test_df["preds"] = model.predict(x_test_df).flatten().round()

# plotting
model.plot_importance(figsize=(20, 20))  # , continuous_features=["seasonal"])

############################################
#### Extract feature weights as follows ####
############################################

####  Feature Names ----
model._feature_names
# ['store', 'item', 'date_day', 'date_month', 'item_X_month', 'store_X_month']

# Weights corresponding to "date_day" feature ----
# One values for each day of the week Starts from Monday, ends on Sunday
model.weights[2]
# [0.8139964946466554, 0.934548012608042, 0.9372978282310153, 0.9914896849964379, 1.0568690066185797, 1.1179250467088013, 1.173494369432411]

print("DONE")
