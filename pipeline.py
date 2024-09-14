

@transform_pandas(
    Output(rid="ri.vector.main.execute.e1a6f755-48ac-4e7d-bec9-baf178347e45"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def descisontree(all_cohort):
    #vaccine infection 分开成两个label
  
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 准备数据
    X = all_cohort[['vaccine','infection','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease',  
    'obesity']]
    y = all_cohort['outcome']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of DecisionTreeClassifier')
    plt.legend(loc="lower right")
    plt.show()

    # 获取特征重要性
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Decision Tree Feature Importance')
    plt.show()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.2c3370d8-e929-4e36-8769-836ab32a41be"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def descisontree_correct(all_cohort):
  
    #  vaccine infection 成为一个label
    # 加上年龄
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, roc_curve, auc
    import matplotlib.pyplot as plt

    # 随机抽取十万个样本
    sample_size = 100000
    all_cohort_sample = all_cohort.sample(n=sample_size, random_state=42)
    
    # 准备数据
    X = all_cohort_sample[['vaccine_infection','age','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease', 'obesity']]
    y = all_cohort_sample['outcome']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of DecisionTreeClassifier')
    plt.legend(loc="lower right")
    plt.show()

    # 获取特征重要性
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Decision Tree Feature Importance')
    plt.show()

    # 绘制决策树（可选）
    plt.figure(figsize=(20,10))
    plot_tree(model, feature_names=X.columns, class_names=['0', '1'], filled=True)
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.f6d9e4cb-b448-4693-b5bf-c971f42f084e")
)
def descisontree_infection(infection):
  
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 准备数据
    X = infection[['infection','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease',  
    'obesity']]
    y = infection ['outcome']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of DecisionTreeClassifier')
    plt.legend(loc="lower right")
    plt.show()

    # 获取特征重要性
    feature_importance = model.feature_importances_
    print("Feature Importance:", feature_importance)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center')
    plt.yticks(range(len(feature_importance)), X.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Decision Tree Feature Importance')
    plt.show()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.1c43de1e-bb98-4aeb-b59f-1bcc8b00fdc1"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def log_10thousand(all_cohort):
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np

    # 假设 all_cohort 是一个 pandas DataFrame，并且包含 `person-id`
    # 删除 `person-id` 列
    all_cohort = all_cohort.drop(columns=['person-id'], errors='ignore')

    # 对分类变量进行独热编码
    categorical_vars = ['vaccine_infection', 'gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease', 'obesity']
    all_cohort = pd.get_dummies(all_cohort, columns=categorical_vars, drop_first=True)

    # 确保所有数据都是数值类型
    all_cohort = all_cohort.apply(pd.to_numeric, errors='coerce')
    all_cohort = all_cohort.dropna()

    # 随机选择十万个样本
    sampled_data = all_cohort.sample(n=100000, random_state=42)

    # 准备数据
    X = sampled_data.drop(columns=['outcome'])  # 特征矩阵
    y = sampled_data['outcome']                # 目标变量

    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练逻辑回归模型，增加最大迭代次数
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 预测类别概率
    probabilities = model.predict_proba(X_test)

    # 输出预测为1的概率
    print(probabilities[:, 1])

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of LogisticRegression')
    plt.legend(loc="lower right")
    plt.show()

    # 使用 statsmodels 进行模型总结
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())

    # 获取模型系数
    coefficients = model.coef_[0]

    # 获取特征名称
    feature_names = sampled_data.drop(columns=['outcome']).columns

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients, align='center')
    plt.yticks(range(len(coefficients)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Logistic Regression Coefficients')
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.4f77e3ca-81e8-4d89-b394-06a7784d0020"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def log_change_label(all_cohort):
    
    #now vaccine 1 infection 2
    # adding age 
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np

    # 对分类变量进行独热编码
    categorical_vars = ['vaccine_infection','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease', 'obesity']
    all_cohort = pd.get_dummies(all_cohort, columns=categorical_vars, drop_first=True)

    # 确保所有数据都是数值类型
    all_cohort = all_cohort.apply(pd.to_numeric, errors='coerce')
    all_cohort = all_cohort.dropna()

    # 随机选择十万个样本
    sampled_data = all_cohort.sample(n=100000, random_state=42)

    # 准备数据
    X = sampled_data.drop(columns=['outcome'])  # 特征矩阵
    y = sampled_data['outcome']                # 目标变量

    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练逻辑回归模型，增加最大迭代次数
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 预测类别概率
    probabilities = model.predict_proba(X_test)

    # 输出预测为1的概率
    print(probabilities[:, 1])

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of LogisticRegression')
    plt.legend(loc="lower right")
    plt.show()

    # 使用 statsmodels 进行模型总结
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())

    # 获取模型系数
    coefficients = model.coef_[0]

    # 获取特征名称
    feature_names = sampled_data.drop(columns=['outcome']).columns

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients, align='center')
    plt.yticks(range(len(coefficients)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Logistic Regression Coefficients')
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.dbd7e730-5c9f-4c3d-baf0-23b26cdc9400"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def log_reg(all_cohort):
    # LogisticRegression：for classification

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 准备数据
    X = all_cohort[['vaccine','infection','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease',     
    'obesity']]
    y = all_cohort['outcome']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 预测类别概率
    probabilities = model.predict_proba(X_test)
    # 如果你需要一个模型摘要，可以使用其他库的方法，比如 statsmodels
   

    # 输出预测为1的概率
    print(probabilities[:, 1])

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of LogisticRegression')
    plt.legend(loc="lower right")
    plt.show()

    import statsmodels.api as sm
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())
    
    # 获取模型系数
    coefficients = model.coef_[0]

    # 获取特征名称
    feature_names = X.columns

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients, align='center')
    plt.yticks(range(len(coefficients)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Logistic Regression Coefficients')
    plt.show()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.5260b90a-64c9-4209-87bf-79c39eee7c1d"),
    all_cohort=Input(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25")
)
def log_reg_1(all_cohort):
    # LogisticRegression：for classification

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 准备数据
    X = all_cohort[['vaccine','gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease',     
    'obesity']]
    y = all_cohort['outcome']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 预测类别概率
    probabilities = model.predict_proba(X_test)
    # 如果你需要一个模型摘要，可以使用其他库的方法，比如 statsmodels
   

    # 输出预测为1的概率
    print(probabilities[:, 1])

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of LogisticRegression')
    plt.legend(loc="lower right")
    plt.show()

    import statsmodels.api as sm
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())
    
    # 获取模型系数
    coefficients = model.coef_[0]

    # 获取特征名称
    feature_names = X.columns

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients, align='center')
    plt.yticks(range(len(coefficients)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Logistic Regression Coefficients')
    plt.show()

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.8e56e15a-d7ee-4833-9dc5-e723a5db5a83"),
    vaccine_group=Input(rid="ri.foundry.main.dataset.83ec449a-947d-44d8-83a9-88e9743f0ab8")
)
def log_vaccine(vaccine_group):

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import numpy as np

    # 对分类变量进行独热编码
    categorical_vars = ['vaccine', 'gender', 'race', 'ethnicity', 'past_AKI', 'hypertension', 'diabetes_mellitus', 'heart_failure', 'cardiovascular_disease', 'obesity']
    vaccine_group = pd.get_dummies(vaccine_group, columns=categorical_vars, drop_first=True)

    # 确保所有数据都是数值类型
    vaccine_group = vaccine_group.apply(pd.to_numeric, errors='coerce')
    vaccine_group = vaccine_group.dropna()

    # 随机选择十万个样本
    sampled_data = vaccine_group.sample(n=100000, random_state=42)

    # 准备数据
    X = sampled_data.drop(columns=['outcome'])  # 特征矩阵
    y = sampled_data['outcome']                # 目标变量

    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练逻辑回归模型，增加最大迭代次数
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 预测类别概率
    probabilities = model.predict_proba(X_test)

    # 输出预测为1的概率
    print(probabilities[:, 1])

    # 计算AUC曲线
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic of LogisticRegression')
    plt.legend(loc="lower right")
    plt.show()

    # 使用 statsmodels 进行模型总结
    logit_model = sm.Logit(y_train, sm.add_constant(X_train))
    result = logit_model.fit()
    print(result.summary())

    # 获取模型系数
    coefficients = model.coef_[0]

    # 获取特征名称
    feature_names = sampled_data.drop(columns=['outcome']).columns

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(coefficients)), coefficients, align='center')
    plt.yticks(range(len(coefficients)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.title('Logistic Regression Coefficients')
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.4ce59be2-d7db-425e-abac-9fda47298267"),
    infection_group=Input(rid="ri.foundry.main.dataset.877ff7a6-ff31-4f60-a5f3-b28978fa253d")
)
def mlp(infection_group):    
    df = infection_group
    df_array = df.to_numpy(dtype=int)
    X_train = df_array[:100000,2:]
    Y_train = df_array[:100000,1]

    X_test = df_array[2000000:,2:]
    Y_test = df_array[2000000:,1]

    mlp = Network()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1)

    traindata = Data(X_train, Y_train)
    trainloader = DataLoader(traindata, batch_size=128, 
                            shuffle=True, num_workers=0)

    testdata = Data(X_test, Y_test)
    testloader = DataLoader(testdata, batch_size=1024, 
                            shuffle=True, num_workers=0)

    epochs = 200
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            labels = labels[:,None]
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
        # display statistics
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
    
        with torch.no_grad():
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels[:,None]

                outputs = mlp(inputs)
                test_pred = torch.round(outputs)
                acc = accuracy_fn(labels,test_pred)

                print(f'[{epoch + 1}, {i + 1:5d}] acc: {acc:.5f}')
                
                break

    return

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

input_dim = 9
hidden_layers = 25
output_dim = 1

class Data(Dataset):  
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        # need to convert float64 to Long else 
        # will get the following error
        # RuntimeError: expected scalar type Long but found Float
        self.y = torch.from_numpy(y_train.astype(np.float32))
        self.len = self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.len

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)  

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x) 
        return x

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

