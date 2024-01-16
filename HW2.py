import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# 讀取CSV檔案
data = pd.read_csv("./HW2_heart.csv")
#-----------------------------------------------------------PART1#-----------------------------------------------------------
# print(data.head())

# print(data.shape)

# print(data.info())
# print(data.groupby('HeartDisease').size())

#法一One-Hot Encoding
columns_to_encode = ['Sex', 'ChestPainType', 'RestingECG','ExerciseAngina','ST_Slope']  # 指定要編碼的欄位名稱列表
encoded_data = pd.get_dummies(data, columns=columns_to_encode)
# print(encoded_data.head())
# print(encoded_data.info())

#-----------------------------------------------------------PART2#-----------------------------------------------------------
# 使用 describe() 方法計算統計摘要
statistics = encoded_data.describe()

# 輸出統計摘要
# print(statistics)




#-----------------------------------------------------------PART3#-----------------------------------------------------------

# 設定熱力圖的尺寸
plt.figure(figsize=(18, 12))

# 計算相關係數矩陣
corr = encoded_data.corr()

# 繪製熱力圖，調整字體大小和顯示數值的格式
sns.heatmap(data=corr, annot=True, square=True, fmt='.2f', annot_kws={'fontsize': 6})



# 顯示圖形
#plt.show()

# correlation_df = pd.DataFrame({'ST_Slope_Up': [encoded_data['ST_Slope_Up'].corr(encoded_data['HeartDisease'])]})
# correlation_df.index = ['HeartDisease']
# print(correlation_df)

#---------------------------------------------------建立相關係數較高的表格---------------------------------------------------
# 建立空的 DataFrame
# correlation_df = pd.DataFrame()

# # 欄位名稱列表
# column_names = ['ST_Slope_Up', 'ST_Slope_Flat','ST_Slope_Down']
# ExerciseAngina= ['ExerciseAngina_N', 'ExerciseAngina_Y']
# ChestPainType= ['ChestPainType_ATA', 'ChestPainType_NAP','ChestPainType_ASY', 'ChestPainType_TA']
# Oldpeak= ['Oldpeak']
# # 計算相關係數並添加到 DataFrame
# for column in column_names:
#     correlation_df[column] = [encoded_data[column].corr(encoded_data['HeartDisease'])]
#  # 計算相關係數並添加到 DataFrame
# for column in ExerciseAngina:
#     correlation_df[column] = [encoded_data[column].corr(encoded_data['HeartDisease'])]
# for column in ChestPainType:
#     correlation_df[column] = [encoded_data[column].corr(encoded_data['HeartDisease'])]      
# for column in Oldpeak:
#     correlation_df[column] = [encoded_data[column].corr(encoded_data['HeartDisease'])]   
# # 設定索引名稱
# correlation_df.index = ['HeartDisease']

# # 印出相關係數表格
# # print(correlation_df)
# correlation = correlation_df.T
# print(correlation)


#-----------------------------------------------------------整理機率-----------------------------------------------------------
# # 選擇特徵變數和目標變數
# features = ['ChestPainType_ATA', 'ChestPainType_NAP','ChestPainType_ASY', 'ChestPainType_TA']
# target = 'HeartDisease'

# # 計算不同 ChestPainType 的心臟病比率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA', 'HeartDiseaseRatio']

# # 輸出結果
# print("不同 ChestPainType 的心臟病比率:")
# print(result_df)


# # 選擇特徵變數和目標變數
# features = ['ExerciseAngina_Y', 'ExerciseAngina_N']
# target = 'HeartDisease'

# # 計算不同 ExerciseAngina 的心臟病比率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ExerciseAngina_Y', 'ExerciseAngina_N', 'HeartDiseaseRatio']

# # 輸出結果
# print("有沒有 ExerciseAngina 的心臟病比率:")
# print(result_df)




# # 選擇特徵變數和目標變數
# features = ['ST_Slope_Up', 'ST_Slope_Flat','ST_Slope_Down']
# target = 'HeartDisease'

# # 計算不同 ExerciseAngina 的心臟病比率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ST_Slope_Up', 'ST_Slope_Flat','ST_Slope_Down', 'HeartDiseaseRatio']

# # 輸出結果
# print("不同 ST_Slope 的心臟病比率:")
# print(result_df)


# # 選擇特徵變數和目標變數
# features = ['Oldpeak']
# target = 'HeartDisease'

# # 計算不同 ExerciseAngina 的心臟病比率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['Oldpeak', 'HeartDiseaseRatio']

# # 輸出結果
# print("Oldpeak 的心臟病比率:")
# print(result_df)


#-----------------------------------------------------------映射字典#-----------------------------------------------------------


# # 建立欄位名稱映射字典(因為欄位名稱太長)
# column_mapping = {

#     "ChestPainType": "CP",
#     "RestingBP": "rBP",
#     "Cholesterol": "Chole",
#     "FastingBS": "FBS",
#      "RestingECG": "rECG",
#      "ExerciseAngina": "ExAng",
#        "ST_Slope": "Slope",
#         "HeartDisease": "HD",
#     # 添加其他欄位名稱的對應關係
# }

# # 使用rename()函數更改欄位名稱
# data = data.rename(columns=column_mapping)

# # 保存修改後的CSV檔案
# data.to_csv("./modified_heart.csv", index=False)
# data = pd.read_csv("./HW2_heart.csv")
# data.to_csv("./HW2_heart.csv", index=False)
# plt.figure(figsize=(18, 16))
# corr = data.corr()
# sns.heatmap(data=corr, annot=True, square=True, fmt='.2f')

# plt.savefig('heartcornew.png')



# plt.show()
#-----------------------------------------------------------畫長條圖-----------------------------------------------------------

# # 選擇特徵變數和目標變數
# features = ['ExerciseAngina_Y', 'ExerciseAngina_N']
# target = 'HeartDisease'

# # 計算不同特徵變數組合下的心臟病機率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ExerciseAngina_Y', 'ExerciseAngina_N', 'HeartDisease']

# # 視覺化長條圖
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(result_df)), result_df['HeartDisease'], width=0.3)
# plt.xticks(range(len(result_df)), result_df['ExerciseAngina_Y'], rotation=45)
# plt.xlabel('Exercise Angina')
# plt.ylabel('Heart Disease Probability')
# plt.title('Heart Disease Probability by Exercise Angina')
# plt.tight_layout()
# plt.savefig('Heart Disease Probability by Exercise Angina.png')





# # 選擇特徵變數和目標變數
# features = ['ST_Slope_Up', 'ST_Slope_Flat', 'ST_Slope_Down']
# target = 'HeartDisease'

# # 計算不同特徵變數組合下的心臟病機率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ST_Slope_Up', 'ST_Slope_Flat', 'ST_Slope_Down', 'HeartDisease']

# # 視覺化長條圖
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(result_df)), result_df['HeartDisease'], width=0.3)
# plt.xticks(range(len(result_df)), ['ST_Slope_Up', 'ST_Slope_Flat', 'ST_Slope_Down'], rotation=45)
# plt.xlabel('ST Slope')
# plt.ylabel('Heart Disease Probability')
# plt.title('Heart Disease Probability by ST Slope')
# plt.tight_layout()
# plt.savefig('Heart_Disease_Probability_by_ST_Slope.png')



# # 選擇特徵變數和目標變數
# features = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA']
# target = 'HeartDisease'

# # 計算不同特徵變數組合下的心臟病機率
# ratio = encoded_data.groupby(features)[target].mean()

# # 將結果轉換為 DataFrame
# result_df = pd.DataFrame(ratio).reset_index()
# result_df.columns = ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA', 'HeartDisease']

# # 視覺化長條圖
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(result_df)), result_df['HeartDisease'], width=0.3)
# plt.xticks(range(len(result_df)), ['ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_ASY', 'ChestPainType_TA'], rotation=45)
# plt.xlabel('ST Slope')
# plt.ylabel('Heart Disease Probability')
# plt.title('Heart Disease Probability by ChestPainType')
# plt.tight_layout()
# plt.savefig('Heart_Disease_Probability_by_ChestPainType.png')
# plt.show()




encoded_data_y = np.array(encoded_data['HeartDisease']).astype(int)
encoded_data=encoded_data.drop('HeartDisease', axis='columns')


#-----------------------------------------------------------標準化數據-----------------------------------------------------------
# 創建標準化的物件
scaler = StandardScaler()

# 將所有欄位進行標準化
scaled_data = scaler.fit_transform(encoded_data)

# 將標準化後的資料轉換為 DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=encoded_data.columns)

# 輸出標準化後的資料
# print(scaled_df)


#-----------------------------------------------------------PART4-----------------------------------------------------------



# 資料分割，將讀出來的資料切成訓練集、驗證集與測試集
data_num = scaled_data.shape[0]

# 取得一筆與 data 數量相同的亂數索引，主要目的是用於打散資料
indices = np.random.permutation(data_num)

# 並將亂數索引值分為 train_indices、val_indices 與 test_indices，劃分比例為 6:1:3
train_indices = indices[:round(data_num*(1-0.4))]
val_indices = indices[round(data_num*(1-0.4)):round(0.7*data_num)]
test_indices = indices[round(data_num*0.7):]

# 將資料分割為訓練集、驗證集與測試集
# train_data = pd.DataFrame(np.array(scaled_data)[train_indices], columns=scaled_data.columns)
# val_data = pd.DataFrame(np.array(scaled_data)[val_indices], columns=scaled_data.columns)
# test_data = pd.DataFrame(np.array(scaled_data)[test_indices], columns=scaled_data.columns)

train_x = pd.DataFrame(scaled_data[train_indices], columns=encoded_data.columns[:])
val_x = pd.DataFrame(scaled_data[val_indices], columns=encoded_data.columns[:])
test_x = pd.DataFrame(scaled_data[test_indices], columns=encoded_data.columns[:])

train_y = np.array(encoded_data_y[train_indices]).astype(int)
val_y = np.array(encoded_data_y[val_indices]).astype(int)
test_y = np.array(encoded_data_y[test_indices]).astype(int)



# test_y=np.array(test_y)
# test_pred=np.array(test_pred)
# print(np.unique(test_y))
# print(np.unique(test_pred))
# print(train_x.shape)
# print(train_y.shape)
# #-----------------------------------------------------------logistic_regression-----------------------------------------------------------

# # 定義要調整的超參數範圍
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'lbfgs'],
#     'class_weight': [None, 'balanced']
# }

# # 建立邏輯回歸模型
# logistic_regression = LogisticRegression()

# # 建立 GridSearchCV 物件
# grid_search = GridSearchCV(logistic_regression, param_grid, cv=5)

# # 在訓練集上進行超參數調整
# grid_search.fit(train_x, train_y)

# # 找到最佳的超參數組合
# best_params = grid_search.best_params_
# print("最佳超參數組合:", best_params)

# # 使用最佳超參數重新建立模型
# best_model = LogisticRegression(**best_params)
# best_model.fit(train_x, train_y)

# # 預測測試集
# test_pred = best_model.predict(test_x)
# # 預測測試集
# train_pred = best_model.predict(train_x)


# # 計算訓練的準確率
# train_accuracy = accuracy_score(train_y, train_pred)
# print("訓練集準確率:", train_accuracy)

# # 計算測試集的準確率
# test_accuracy = accuracy_score(test_y, test_pred)
# print("測試集準確率:", test_accuracy)



# # 建立Logistic模型
# logisticModel = LogisticRegression(random_state=0)
# # 使用訓練資料訓練模型
# logisticModel.fit(train_x, train_y)
# # 使用測試資料預測分類
# predicted = logisticModel.predict(test_x)
# # 計算訓練集和測試集的準確率
# train_accuracy = accuracy_score(train_y, logisticModel.predict(train_x))
# test_accuracy = accuracy_score(test_y, predicted)
# print('沒有使用超參數訓練集準確率:', train_accuracy)
# print('沒有使用超參數測試集準確率:', test_accuracy)

# # print(encoded_data.info())
# # , labels=["bat", "ball"]
# # test_y=np.array(test_y)
# # test_pred=np.array(test_pred)
# # print(np.unique(test_y))
# # print(np.unique(test_pred))

# #-----------------------------------------------------------logistic_regression的混淆矩陣-----------------------------------------------------------
# mat_con = confusion_matrix(test_y, test_pred)
# print("mat_con:")
# print(mat_con)
# fig, px = plt.subplots(figsize=(7.5, 7.5))
# px.matshow(mat_con, cmap=plt.cm.YlOrRd, #plt.cm.Blues, 
#            alpha=0.5)
# for m in range(mat_con.shape[1]):
    
#     for n in range(mat_con.shape[0]):
#         px.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')

# # plt.xticks(range(2),["bat", "ball"])
# # plt.yticks(range(2),["bat", "ball"])
# # Sets the labels
# plt.xlabel('Actuals', fontsize=16)
# plt.ylabel('Predictions', fontsize=16)
# plt.title('Confusion Matrix', fontsize=15)
# plt.show()
# fig.savefig('Confusion Matrix logic.png')
#-----------------------------------------------------------SVM-----------------------------------------------------------

# # 建立 SVM 模型
# svm_model = SVC(random_state=0)

# # 使用訓練資料訓練模型
# svm_model.fit(train_x, train_y)

# # 使用訓練資料預測分類
# train_pred = svm_model.predict(train_x)

# # 使用測試資料預測分類
# test_pred = svm_model.predict(test_x)

# # 計算訓練集和測試集的準確率
# train_accuracy = accuracy_score(train_y, train_pred)
# test_accuracy = accuracy_score(test_y, test_pred)

# print("訓練集準確率:", train_accuracy)
# print("測試集準確率:", test_accuracy)

# # 定義要調整的超參數範圍
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
# }

# # 建立 SVM 模型
# svm_model = SVC()

# # 建立 GridSearchCV 物件
# grid_search = GridSearchCV(svm_model, param_grid, cv=5)

# # 在訓練集上進行超參數調整
# grid_search.fit(train_x, train_y)

# # 找到最佳的超參數組合
# best_params = grid_search.best_params_
# print("最佳超參數組合:", best_params)

# # 使用最佳超參數重新建立模型
# best_model = SVC(**best_params)
# best_model.fit(train_x, train_y)

# # 預測訓練集
# train_pred = best_model.predict(train_x)

# # 計算訓練集的準確率
# train_accuracy = accuracy_score(train_y, train_pred)
# print("有參數訓練集準確率:", train_accuracy)

# # 預測測試集
# test_pred = best_model.predict(test_x)

# # 計算測試集的準確率
# test_accuracy = accuracy_score(test_y, test_pred)
# print("有參數測試集準確率:", test_accuracy)


# #-----------------------------------------------------------SVM的混淆矩陣-----------------------------------------------------------
# mat_con = confusion_matrix(test_y, test_pred)
# print("mat_con:")
# print(mat_con)
# fig, px = plt.subplots(figsize=(7.5, 7.5))
# px.matshow(mat_con, cmap=plt.cm.YlOrRd, #plt.cm.Blues, 
#            alpha=0.5)
# for m in range(mat_con.shape[1]):
    
#     for n in range(mat_con.shape[0]):
#         px.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')

# # plt.xticks(range(2),["bat", "ball"])
# # plt.yticks(range(2),["bat", "ball"])
# # Sets the labels
# plt.xlabel('Actuals', fontsize=16)
# plt.ylabel('Predictions', fontsize=16)
# plt.title('Confusion Matrix', fontsize=15)
# plt.show()
# fig.savefig('Confusion Matrix SVM.png')



# #-----------------------------------------------------------RandomForestRegressor-----------------------------------------------------------


# clf = RandomForestClassifier(max_depth=3, n_jobs=-1)
# clf.fit(train_x, train_y)
# y_pred = clf.predict(test_x)
# # print(classification_report(test_y, y_pred))

# importances = clf.feature_importances_
# std = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
# idx = np.argsort(importances)[::-1]

# plt.title("Feature importances")
# plt.bar(range(train_x.shape[1]), importances[idx], 
#         yerr=std[idx], align="center")
# plt.xticks(range(train_x.shape[1]), labels=train_x.columns[idx])
# plt.xlim([-1, train_x.shape[1]])
# plt.ylim([0, 0.6])

# from sklearn.feature_selection import SelectFromModel

# # 建立特徵選取器，門檻值預設為重要性的平均值
# selector = SelectFromModel(clf)
# selector.fit(train_x, train_y)
# # print('門檻值 =', selector.threshold_)
# # print('特徵遮罩：', selector.get_support())





# # 選出新特徵，重新訓練隨機森林
# train_x_new = selector.transform(train_x)
# clf.fit(train_x_new, train_y)

# test_x_new = selector.transform(test_x)
# y_pred = clf.predict(test_x_new)

# # 更新變數形狀
# importances = clf.feature_importances_
# std = np.std([t.feature_importances_ for t in clf.estimators_], axis=0)
# idx = np.argsort(importances)[::-1]

# print("Idx shape:", idx.shape)
# print("Std shape:", std.shape)
# print("Importances shape:", importances.shape)
# print("Train X shape:", train_x.shape)

# # 繪製特徵重要性圖
# fig, ax = plt.subplots()
# ax.bar(range(train_x_new.shape[1]), importances[idx], yerr=std[idx], align="center")
# ax.set_xticks(range(train_x_new.shape[1]))
# ax.set_xticklabels(train_x.columns[idx], rotation=90)
# ax.set_xlim([-1, train_x_new.shape[1]])
# ax.set_ylim([0, 0.6])
# ax.set_title("Feature Importances")
# ax.set_xlabel("Features")
# ax.set_ylabel("Importance")

# plt.savefig('forest.png')



# #-----------------------------------------------------------RandomForestRegressor的混淆矩陣-----------------------------------------------------------
# mat_con = confusion_matrix(test_y, y_pred)
# print("mat_con:")
# print(mat_con)
# fig, px = plt.subplots(figsize=(7.5, 7.5))
# px.matshow(mat_con, cmap=plt.cm.YlOrRd, #plt.cm.Blues, 
#            alpha=0.5)
# for m in range(mat_con.shape[1]):
    
#     for n in range(mat_con.shape[0]):
#         px.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')

# # plt.xticks(range(2),["bat", "ball"])
# # plt.yticks(range(2),["bat", "ball"])
# # Sets the labels
# plt.xlabel('Actuals', fontsize=16)
# plt.ylabel('Predictions', fontsize=16)
# plt.title('Confusion Matrix', fontsize=15)
# plt.show()
# fig.savefig('Confusion Matrix RandomForestRegressor.png')


#-----------------------------------------------------------KNN-----------------------------------------------------------


# 建立 KNN 分類器，預設 k=5
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# 使用訓練集進行模型訓練
knn.fit(train_x, train_y)

# 預測測試集
test_pred = knn.predict(test_x)

# 計算訓練集的精確度
train_accuracy = knn.score(train_x, train_y)
print("訓練集精確度:", train_accuracy)
# 輸出預測結果
print("測試集準確率:", knn.score(test_x, test_y))

# # 輸出預測結果的機率
# print(knn.predict_proba(test_x))
#-----------------------------------------------------------KNN的混淆矩陣-----------------------------------------------------------
mat_con = confusion_matrix(test_y, test_pred)
print("mat_con:")
print(mat_con)
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(mat_con, cmap=plt.cm.YlOrRd, #plt.cm.Blues, 
           alpha=0.5)
for m in range(mat_con.shape[1]):
    
    for n in range(mat_con.shape[0]):
        px.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')

# plt.xticks(range(2),["bat", "ball"])
# plt.yticks(range(2),["bat", "ball"])
# Sets the labels
plt.xlabel('Actuals', fontsize=16)
plt.ylabel('Predictions', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()
fig.savefig('Confusion Matrix KNN.png')