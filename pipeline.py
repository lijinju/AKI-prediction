

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

    

