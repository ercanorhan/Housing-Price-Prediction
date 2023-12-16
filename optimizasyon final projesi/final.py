import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv('housing_price_dataset.csv')
# print(data.info())

x=data.drop(['Price'],axis=1)
y=data['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

train_data=x_train.join(y_train)
train_data.hist(bins=50,figsize=(20,15))
train_data_wo_neigh=train_data.drop(['Neighborhood'],axis=1)
train_data_wo_neigh.corr()

plt.figure(figsize=(20,15))
sns.heatmap(train_data_wo_neigh.corr(),annot=True,cmap='RdYlGn')
dublicate_data = data[data.drop('Price', axis=1).duplicated(keep=False)].sort_values(['SquareFeet', 'Bedrooms', 'Neighborhood', 'YearBuilt', 'Price'])
print(f"Number of houses with the same parameters but different costs:{dublicate_data.shape[0]}")
dublicate_data.head(20)

data.drop(data[data.Price <= 0].index, axis=0, inplace=True) # We'll remove houses with a negative price
data.drop_duplicates(keep=False, inplace=True) # Delete all records containing duplicates
data.Neighborhood = LabelEncoder().fit_transform(data.Neighborhood) # Let's encode the Neighborhood feature
data.head()


models = [
    MLPRegressor(),
    XGBRegressor(),
    LinearRegression(),
    RandomForestRegressor(),
    KNeighborsRegressor()
]

best_model = None # Best model
best_score = None # Accuracy of the best model
best_loss = None # A mistake by the best model
for clf in models:
    clf.fit(x_train, y_train) 
    y_pred = clf.predict(x_test)

    # Calculating metrics
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"{clf.__class__.__name__:30}: R2_score: {r2:17}, RMSE: {round(rmse, 6):10}")

    # Проверям на лучшую модель
    if best_loss != None:
        if best_loss > rmse:
            best_model = clf
            best_score = r2
            best_loss = rmse
    else:
        best_model = clf
        best_score = r2
        best_loss = rmse

# Bringing out the best model
print("-"*92)
print(f"{best_model.__class__.__name__:30}: R2_score: {best_score}, RMSE: {round(best_loss, 6):10}")