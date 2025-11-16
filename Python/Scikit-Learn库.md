Scikit-learn 是 Python 中最流行的机器学习库之一，提供了各种机器学习算法和工具。
# 数据加载和预处理
## 内置数据集
```python
# 加载内置数据集
iris = datasets.load_iris()  # 鸢尾花数据集
digits = datasets.load_digits()  # 手写数字数据集
breast_cancer = datasets.load_breast_cancer()  # 乳腺癌数据集

# 可以通过print(iris.DESCR)查看数据集描述文档

# 获取特征和目标
X = iris.data # 特征数据矩阵（150样本 × 4特征）
y = iris.target # 目标向量（0,1,2对应三种鸢尾花类别）
feature_names = iris.feature_names # 特征名称列表（如'sepal length'）
target_names = iris.target_names # 类别名称列表（如'setosa'）

# 生成可控的合成数据集，用于算法测试或实验
# 生成包含 1000 个样本的二分类数据集
X, y = make_classification(n_samples=1000, # 样本量：1000 个 
						n_features=20, # 总特征数：20 个（含信息特征 + 冗余特征） 
						n_informative=15, # 信息特征数：15 个（对分类有贡献的特征） 
						n_redundant=5, # 冗余特征数：5 个（由信息特征线性组合生成） 
						random_state=42 # 固定随机种子，确保结果可复现 
						)
```
## 数据分割
```python
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, # 输入特征矩阵和目标变量 
												test_size=0.2, # 测试集占比 20% 
												random_state=42,# 固定随机种子 
												stratify=y # 按y的类别分层抽样
												)
```
# 数据预处理
## 特征缩放
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化（均值0，方差1）
scaler = StandardScaler() # 建立标准正态化实例
X_train_scaled = scaler.fit_transform(X_train) # 计算每一个特征的均值和标准差，然后进行标准化
# 有时也会分成.fit和.transform两个方法
X_test_scaled = scaler.transform(X_test)  # 利用从训练集计算出的均值和标准差，标准化测试集

# 归一化（到[0,1]范围）
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

# 鲁棒缩放（对异常值不敏感）
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)
```
## 分类类别编码
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# 标签编码 (Label Encoding)（将类别转换为整数）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 独热编码（One-Hot Encoding）（创建虚拟变量）
onehot_encoder = OneHotEncoder(sparse_out=False)
X_onehot = onehot_encoder.fit_transform(X_categorical)

# 序数编码###  (Ordinal Encoding)（有序分类变量）
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_ordinal = ordinal_encoder.fit_transform(X_ordered)
```
- 标签编码：为每个唯一的类别分配一个整数（0, 1, 2, ...）
	- 示例：如果 `y = ['cat', 'dog', 'cat', 'bird']`，编码后可能变为 `[0, 1, 0, 2]`
> 编码主要用于目标变量（y），而不是特征变量（X）。该编码可能会引入错误的顺序关系（例如，算法可能认为 2 > 1 > 0），这对于没有内在顺序的类别是不合适的。
- 独热编码 ：“热”源自电子电路设计术语，表示“激活状态”。在数字电路中，寄存器位被置为1时代表“热”或“开启”，而0表示“冷”或“关闭”。独热编码为每个类别创建一个新的二进制特征（列）。对于每个样本，只有一个特征为 1（"热"），其余为 0。
	-  `sparse=False`：确保返回一个密集数组（通常是 NumPy 数组）而不是稀疏矩阵。在新版本的 scikit-learn 中，这个参数已被 `sparse_output` 取代。
	- 示例：如果有一个特征 "颜色" 包含 `['红', '绿', '蓝']`，独热编码会创建三个新特征：
		- 颜色_红: `[1, 0, 0]`
		- 颜色_绿: `[0, 1, 0]`
		- 颜色_蓝: `[0, 0, 1]`
- 序数编码：根据指定的顺序为每个类别分配一个整数，适用于有内在顺序的类别。
	-  `categories=[['low', 'medium', 'high']]`：明确指定类别的顺序。这是可选的，但建议使用，以确保编码的一致性。
	- 示例：按照指定的顺序 `['low', 'medium', 'high']`，编码结果为：
		- `'low'` → 0
		- ` 'medium'` → 1
		- `'high'` → 2
## 处理缺失值
```python
from sklearn.impute import SimpleImputer

# 使用均值、中位数或众数填充缺失值
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_missing)
```
# 模型训练与预测
## 监督学习示例
```python
# 线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 逻辑回归
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# 支持向量机
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 梯度提升
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
```
## 无监督学习实例
```python
# K-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# 主成分分析(PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 异常检测
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(X)
```
# 模型评估
## 分类模型评估
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 准确率
accuracy = accuracy_score(y_test, y_pred)

# 精确率、召回率、F1分数
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 分类报告
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# ROC-AUC（需要预测概率）
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
```
## 回归模型评估
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred) # 平均绝对误差
mse = mean_squared_error(y_test, y_pred) # 均方误差
rmse = np.sqrt(mse) # 均方根误差
r2 = r2_score(y_test, y_pred) # R方决定系数
```
## 交叉验证
```python
from sklearn.model_selection import cross_val_score, cross_validate

# 简单交叉验证
scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

# 详细交叉验证
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(rf, X, y, cv=5, scoring=scoring)
for metric in scoring:
    print(f"{metric}: {cv_results[f'test_{metric}'].mean():.3f}")
```
# 模型选择与调参
## 网格搜索
```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # 使用所有可用的CPU核心
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 查看最佳参数和得分
print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证得分:", grid_search.best_score_)

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
```
## 随机搜索
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 定义参数分布
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11)
}

# 创建随机搜索对象
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 尝试的参数组合数量
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```
# 流水线(Pipeline)
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 创建预处理和建模的流水线
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 缺失值处理
    ('scaler', StandardScaler()),  # 特征缩放
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),  # 特征选择
    ('classifier', RandomForestClassifier(random_state=42))  # 分类器
])

# 定义流水线的参数网格
param_grid = {
    'feature_selection__k': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

# 在流水线上进行网格搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 使用最佳流水线进行预测
y_pred = grid_search.predict(X_test)
```
# 示例
训练机器学习模型的一般流程（以KNN的鸢尾花分类为例）：
```python
# 用于数据处理和数组操作  
import numpy as np  
import pandas as pd  
  
# 从 sklearn 导入数据集、模型、评估工具  
from sklearn import datasets # 用于导入预制测试用的数据，如本程序需要的鸢尾花  
from sklearn.model_selection import train_test_split # 用于分隔数据集为测试用和检验用  
from sklearn.preprocessing import StandardScaler # 用于归一化数据  
from sklearn.neighbors import KNeighborsClassifier # 导入KNN算法必备的模块  
from sklearn.metrics import accuracy_score, classification_report # 用于评价模型预测准确度  
  
# 用于可视化（可选）  
import matplotlib.pyplot as plt  

# 一、载入数据（这里采用库提供的内置数据）
# 加载鸢尾花数据集  
iris = datasets.load_iris() # 提供了一个4特征150样本的数据集  
# 查看数据描述  
print(iris.DESCR)

# 二、将数据特征和目标变量分配给 X 和 y
# X 是特征矩阵 ( samples × features )
X = iris.data  
# y 是目标向量（标签）  
y = iris.target  
  
# 也可以方便地转换为 pandas DataFrame（更易于查看）  
df = pd.DataFrame(X, columns=iris.feature_names) # 转换成pandas DataFrames表格  
df['target'] = y  

# 三、随机将数据分为训练集（80%）和测试集（20%）  
# stratify=y 表示按标签分层抽样，确保训练集和测试集中各类别比例一致  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  

# 四、标准化数据（这里是标准正态化）
# 初始化一个标准化器  
scaler = StandardScaler()  
# 用计算好的参数转换训练数据和测试数据  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)
# 查看标准化后的效果  
print("训练集均值:", X_train_scaled.mean(axis=0)) # 应该接近 0
print("训练集标准差:", X_train_scaled.std(axis=0)) # 应该接近 1  

# 五、训练模型
# 1. 实例化模型，设置超参数（这里选择 3 个邻居）  
knn = KNeighborsClassifier(n_neighbors=3) # 建立KNN模型实例  
# 2. 在缩放后的训练数据上拟合模型  
knn.fit(X_train_scaled, y_train) # 使用归一化后的数据进行训练  

# 六、使用模型对测试集进行预测  
y_pred = knn.predict(X_test_scaled) # 使用训练好的模型根据测试集进行预测  
  
# 七、评估准确性  
accuracy = accuracy_score(y_test, y_pred)  
print(f"模型准确率: {accuracy:.2f}")  
# 生成更详细的分类报告  
report = classification_report(y_test, y_pred, target_names=iris.target_names)  
print("分类报告:\n", report)
```