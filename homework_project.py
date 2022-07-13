import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import auc


# 存储停用词列表
def stopwordlist():
    stopwords = [line.strip() for line in open("Chinesestopwords.txt", encoding="UTF-8").readlines()]
    return stopwords


# 读取数据 + 数据处理
def product(path):
    data = pd.read_csv(path)
    label = data["label"]
    title = data["Title"]
    name = data["Ofiicial Account Name"]
    content = data["Report Content"]

    # 将每行的name值暂存进列表，以便下面与分词后的title和content拼接
    new_name = []
    for tem in name:
        res = ""
        res += str(tem)
        new_name.append(res)

    # 将title与content拼接，去停用词、jieba分词 + 与name拼接
    temp_data = title + content
    new_data = []  # new_data存放处理好的数据
    stopwords = stopwordlist()
    i = 0
    for sentense in temp_data:
        tmp = jieba.cut(str(sentense).strip())  # jieba分词
        res = ""
        for word in tmp:
            if word not in stopwords:
                if word != '\t':
                    res += word
                    res += "  "
        res += new_name[i]
        i += 1
        new_data.append(res)

    # print(new_data)  # 打印分割后的数据，看是否正确
    return new_data, label


# Tf-idf向量化 + 标准化
def handle(train, test):
    # Tf-idf向量化
    transfer_data = TfidfVectorizer()
    train = transfer_data.fit_transform(train)
    test = transfer_data.transform(test)
    # print(transfer_data.get_feature_names()) # 打印tf-idf后的特征词名字

    # StandardScaler标准化
    transfer_stand = StandardScaler(with_mean=False)
    train_stand = transfer_stand.fit_transform(train)
    test_stand = transfer_stand.transform(test)
    # print(train_stand.toarray()) # 打印标准化后的矩阵

    return train_stand.toarray(), test_stand.toarray()


# 随机森林训练
def random_forest(train_data, test_data, train_label, test_label):
    rfc = RandomForestClassifier(n_estimators=161, random_state=90)
    rfc = rfc.fit(train_data, train_label)
    # 准确率
    result = rfc.score(test_data, test_label)
    print("score准确率：", result)
    # 预测值
    y_predict = rfc.predict(test_data)  # 预测值 y_predict 真实值 test_label
    return y_predict


# 下面是情感分析
# 加载词库，并为每种词分别赋予它们对应的分数；
def loadDict(fileName, score):
    wordDict = {}
    with open(fileName, encoding="UTF-8") as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score
    return wordDict


def loadExtentDict(fileName, level):
    extentDict = {}
    for i in range(level):
        with open(fileName + str(i + 1) + ".txt", encoding="UTF-8") as fin:
            for line in fin:
                word = line.strip()
                extentDict[word] = i + 1
    return extentDict


# 计算情感得分
def getScore(combine):
    postDict = loadDict("sentimentDict/正面情感词语.txt", 1)  # 积极情感词典
    negDict = loadDict("sentimentDict/负面情感词语.txt", -1)  # 消极情感词典
    inverseDict = loadDict("sentimentDict/否定词.txt", -1)  # 否定词词典
    extentDict = loadExtentDict("sentimentDict/程度级别词语", 6)
    punc = loadDict("sentimentDict/标点符号.txt", 1)
    exclamation = {"!": 2, "！": 2}

    words = jieba.cut(combine)
    wordList = list(words)
    # print(wordList)

    totalScore = 0  # 记录最终情感得分
    lastWordPos = 0  # 记录情感词的位置
    lastPuncPos = 0  # 记录标点符号的位置
    i = 0  # 记录扫描到的词的位置

    for word in wordList:
        if word in punc:
            lastPuncPos = i

        if word in postDict:
            if lastWordPos > lastPuncPos:
                start = lastWordPos
            else:
                start = lastPuncPos

            score = 1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            for word_after in wordList[i + 1:]:
                if word_after in punc:
                    if word_after in exclamation:
                        score = score + 2
                    else:
                        break
            lastWordPos = i
            totalScore += score
        elif word in negDict:
            if lastWordPos > lastPuncPos:
                start = lastWordPos
            else:
                start = lastPuncPos
            score = -1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            for word_after in wordList[i + 1:]:
                if word_after in punc:
                    if word_after in exclamation:
                        score = score - 2
                    else:
                        break
            lastWordPos = i
            totalScore += score
        i = i + 1

    return totalScore


# 处理文件中的情感特征
def haddle_emotion(path):
    data = pd.read_csv(path)
    title = data["Title"]
    content = data["Report Content"]
    combine = title + content
    emolist = []
    for sentence in combine:
        sc = []
        sc.append(getScore(sentence))
        emolist.append(sc)

    transfer_stand = StandardScaler(with_mean=False)
    emolist = transfer_stand.fit_transform(emolist)
    print(emolist)
    return emolist


# 混淆矩阵计算
def binary_confusion_matrix(acts, pres):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(acts)):
        if acts[i] == 1 and pres[i] == 1:
            TP += 1
        if acts[i] == 0 and pres[i] == 1:
            FP += 1
        if acts[i] == 1 and pres[i] == 0:
            FN += 1
        if acts[i] == 0 and pres[i] == 0:
            TN += 1

    # 混淆矩阵可视化
    labels = [0, 1]
    cm = confusion_matrix(acts, pres, labels=labels)
    print(cm)
    sns.heatmap(data=cm, annot=True, annot_kws={'size': 20, 'weight': 'bold', 'color': 'blue'})
    plt.rc('font', family='Arial Unicode MS', size=14)
    plt.title('confusion_matrix', fontsize=20)
    plt.xlabel('Predict', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.show()

    # 精确率Precision
    P = TP / (TP + FP)
    print("精确率Precision:", P)

    # 召回率Recall
    R = TP / (TP + FN)
    print("召回率Recall:", R)

    # F1
    F1 = 2 / (1 / P + 1 / R)
    print('F1:', F1)

    # 准确率Accuracy
    A = (TP + TN) / (TP + FP + FN + TN)
    print("准确率Accuracy:", A)

    # ROC AUC
    act = np.array(acts)
    pre = np.array(pres)
    FPR, TPR, thresholds = metrics.roc_curve(act, pre)
    AUC = auc(FPR, TPR)
    print('AUC:', AUC)
    plt.rc('font', family='Arial Unicode MS', size=14)
    plt.plot(FPR, TPR, label="AUC={:.2f}".format(AUC), marker='o', color='b', linestyle='--')
    plt.legend(loc=4, fontsize=10)
    plt.title('ROC', fontsize=20)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.show()

    return TP, FP, TN, FN


# 文件读取 + 数据处理
train_path = "train/news.csv"
test_path = "test/news.csv"
train, train_label = product(train_path)
test, test_label = product(test_path)

# 特征工程（情感分析）
train_emo = haddle_emotion(train_path)
test_emo = haddle_emotion(test_path)

# 特征工程（TF-IDF + 标准化）
train_data, test_data = handle(train, test)

# 特征拼接
train_data = np.hstack((train_data, train_emo))
test_data = np.hstack((test_data, test_emo))

# 随机森林训练
y_predict = random_forest(train_data, test_data, train_label, test_label)

# 混淆矩阵
# 真实值 test_label
# 预测值 y_predict
acts = test_label
pres = y_predict
binary_confusion_matrix(acts, pres)
