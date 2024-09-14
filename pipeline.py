

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

    

