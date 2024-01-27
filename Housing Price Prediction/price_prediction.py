import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import warnings
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")
scaler = StandardScaler()

data=pd.read_csv('california_housing.csv')
print(data.info())

data.dropna(inplace=True)
print(data.info())
data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'],axis=1)
print(data)

scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
print(scaled_data)

x=scaled_data.drop(['median_house_value'],axis=1)
y=scaled_data['median_house_value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


train_data=x_train.join(y_train)
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')
train_data.hist(figsize=(15,8))

#for log transformation
# train_data['total_rooms']=np.log(train_data['total_rooms'])
# train_data['total_bedrooms']=np.log(train_data['total_bedrooms'])
# train_data['population']=np.log(train_data['population'])
# train_data['households']=np.log(train_data['households'])
# train_data.hist(figsize=(15,8))   

plt.figure(figsize=(15,8))
sns.scatterplot(x='latitude',y='longitude',data=train_data,hue='median_house_value',palette="coolwarm") #shows the distribution of houses based on their location

#for log transformation
# test_data=x_test.join(y_test)
# test_data['total_rooms']=np.log(test_data['total_rooms'])
# test_data['total_bedrooms']=np.log(test_data['total_bedrooms'])
# test_data['population']=np.log(test_data['population'])
# test_data['households']=np.log(test_data['households'])

print('###############################################################')
print('linear regression metodu')
reg=LinearRegression()
reg.fit(x_train,y_train)
reg_score=reg.score(x_test,y_test)
y_pred = reg.predict(x_test)
rmseLR = np.sqrt(mean_squared_error(y_test, y_pred))
normalised_rmse_LR = rmseLR/(max(y_test)-min(y_test))
training_score_LR = reg.score(x_train, y_train)
y_comp=pd.DataFrame(y_test)
y_comp2=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp2.head(25))
print('linear regression training score:'+ f'{training_score_LR}')
print('linear regression skoru:'+ f'{reg.score(x_test,y_test)}')    
print('linear regression rmse:'+ f'{rmseLR}')
print('normalised rmse:'+ f'{normalised_rmse_LR}')


print('###############################################################')
print('random forest metodu')
forest=RandomForestRegressor()
forest.fit(x_train,y_train)
forest_score=forest.score(x_test,y_test)
y_pred = forest.predict(x_test)
rmseRFR = np.sqrt(mean_squared_error(y_test, y_pred))
normalised_rmse_RFR = rmseRFR/(max(y_test)-min(y_test))
training_score_RFR = forest.score(x_train, y_train)
y_comp=pd.DataFrame(y_test)
y_comp1=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp1.head(25))
print('random forest training score:'+ f'{training_score_RFR}')
print('random forest skoru:'+ f'{forest.score(x_test,y_test)}')
print('random forest rmse:'+ f'{rmseRFR}')
print('normalised rmse:'+ f'{normalised_rmse_RFR}')

print('###############################################################')
Svm=svm.SVR()
Svm.fit(x_train,y_train)
svm_score=Svm.score(x_test,y_test)
y_pred = Svm.predict(x_test)
rmseSVM = np.sqrt(mean_squared_error(y_test, y_pred))
normalised_rmse_SVM = rmseSVM/(max(y_test)-min(y_test))
training_score_SVM = Svm.score(x_train, y_train)
y_comp=pd.DataFrame(y_test)
y_comp3=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp3.head(25))
print('svm training score:'+ f'{training_score_SVM}')
print('svm skoru:'+ f'{Svm.score(x_test,y_test)}')
print('svm rmse:'+ f'{rmseSVM}')
print('normalised rmse:'+ f'{normalised_rmse_SVM}')


print('###############################################################')
print('knn metodu')
knn=KNeighborsRegressor()
knn.fit(x_train,y_train)
knn_score=knn.score(x_test,y_test)
y_pred = knn.predict(x_test)
rmseKNN = np.sqrt(mean_squared_error(y_test, y_pred))
normalised_rmse_KNN = rmseKNN/(max(y_test)-min(y_test))
training_score_KNN = knn.score(x_train, y_train)
y_comp=pd.DataFrame(y_test)
y_comp4=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp4.head(25))
print('knn training score:'+ f'{training_score_KNN}')
print('knn skoru:'+ f'{knn.score(x_test,y_test)}')
print('knn rmse:'+ f'{rmseKNN}')
print('normalised rmse:'+ f'{normalised_rmse_KNN}')

print('###############################################################')
print('mlp metodu')
mlp=MLPRegressor()
mlp.fit(x_train,y_train)
mlp_score=mlp.score(x_test,y_test)
y_pred = mlp.predict(x_test)
rmseMLP = np.sqrt(mean_squared_error(y_test, y_pred))
normalised_rmse_MLP = rmseMLP/(max(y_test)-min(y_test))
training_score_MLP = mlp.score(x_train, y_train)
y_comp=pd.DataFrame(y_test)
y_comp5=y_comp.join(pd.DataFrame({'prediction':y_pred}).set_index(y_test.index))
print(y_comp5.head(25))
print('mlp training score:'+ f'{training_score_MLP}')
print('mlp skoru:'+ f'{mlp.score(x_test,y_test)}')
print('mlp rmse:'+ f'{rmseMLP}') 
print('normalised rmse:'+ f'{normalised_rmse_MLP}')


print('###############################################################')

algorithms = ['LR', 'RFR', "SVM",'KNN', 'MLP']
scores = [reg_score, forest_score, svm_score, knn_score, mlp_score]

normalised_rmse = [normalised_rmse_LR, normalised_rmse_RFR, normalised_rmse_SVM, normalised_rmse_KNN, normalised_rmse_MLP]
training_scores = [training_score_LR, training_score_RFR, training_score_SVM, training_score_KNN, training_score_MLP]

results = pd.DataFrame({'Algorithms': algorithms, 'Scores': scores, 'Normalised RMSE': normalised_rmse, 'Training Scores': training_scores})
results = results.sort_values(by='Scores', ascending=False)


fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.3
bar_positions = range(len(algorithms))

plt.bar(bar_positions, results['Scores'], width=bar_width, label='Scores')
plt.bar([i + bar_width for i in bar_positions], results['Normalised RMSE'], width=bar_width, label='Normalised RMSE')
plt.bar([i + 2 * bar_width for i in bar_positions], results['Training Scores'], width=bar_width, label='Training Scores')

plt.xlabel('Algorithms')
plt.ylabel('Values')
plt.title('Algorithm Comparison')
plt.xticks([i + bar_width for i in bar_positions], results['Algorithms'])
plt.legend()
plt.show()




#in case of overfitting
# from sklearn.model_selection import GridSearchCV 

# param_grid = { 
#     "n_estimators": [200, 500],
#     "max_features": ["auto", "sqrt", "log2"],
# }
# grid = GridSearchCV(forest, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
# grid.fit(x_train, y_train)  

# param_grid = { 
#     "n_estimators": [100, 500],
#     "min_samples_split":[2,4,6,8],
#     "max_depth":[None,4,8]
# }
# grid = GridSearchCV(forest, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
# grid.fit(x_train, y_train)  


