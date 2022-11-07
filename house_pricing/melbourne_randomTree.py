import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

train_home_data = pd.read_csv('sklearn/house_pricing/train.csv')
test_home_data = pd.read_csv('sklearn/house_pricing/test.csv')

#Create target object and columns to be analized
y = train_home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_home_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Testing over and under fitting for DecisionTreeRegressor
candidate_max_leaf_nodes = range(2, 500, 5)
maes = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) 
                for leaf_size in candidate_max_leaf_nodes}   
best_tree_size = min(maes, key = maes.get)
best_mae_decision_tree = min(maes.values())
print(best_tree_size)
print(best_mae_decision_tree)

#Mean absolute error for RandomForestTree
price_model = RandomForestRegressor(random_state=1)
price_model.fit(train_X, train_y)
val_predictions = price_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

#Chosing RandomForestTree as the best method and loading full train data
price_model.fit(X, y)
test_home_data_X = test_home_data[features]
test_preds = price_model.predict(test_home_data_X)

#Saving the predictions
output = pd.DataFrame({'Id': test_home_data.Id, 'SalePrice': test_preds})
output.to_csv('sklearn/house_pricing/submission.csv', index=False)
