from apiParser import getActivityData, updateActivityData, getRecentActivityID, loadTrainingData
from tcxParser import getAllActivityIDs
from json import load
import pickle as pkl
import math
from sys import argv

data = loadTrainingData()

def NoneToQuot(s):
    if s is None:
        return ""
    return s

def NoneToZero(n):
    if n is None or '':
        return 0
    return math.floor(n)

def titleTypeDescIfy(a):
    return [NoneToZero(a["workout_type"]), NoneToQuot(a["name"]) +"\n"+ NoneToQuot(a["description"])]
labelData = [titleTypeDescIfy(a) for a in data if "workout_type" in a]


# nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
corpus = []

def cleanOneTitleDesc(r):
    rlow = r.lower()
    rlist = rlow.split()
    no_stop = [word for word in rlist if word not in stopwords.words("english")]
    no_complicate = [lemmatizer.lemmatize(word) for word in no_stop]
    restring = ' '.join(no_complicate)
    return restring

for i in range(len(labelData)):
    r = cleanOneTitleDesc(labelData[i][1])
    corpus.append((labelData[i][0],r))

labels = [l[0] for l in corpus]
data = [l[1] for l in corpus]
print(data)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=123)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(data)
X_train_cv = cv.fit_transform(data)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class="multinomial", penalty="l2")
lr.fit(X_train_cv, labels)

# from sklearn.tree import DecisionTreeClassifier

# lr = DecisionTreeClassifier()
# lr.max_leaf_nodes = 16
# lr.fit(X_train_cv, labels)
# print(lr.get_n_leaves())
# from sklearn.neural_network import MLPClassifier

# lr = MLPClassifier(max_iter=500)
# lr.fit(X_train_cv, labels)

# This would be different w/ a split.  I dont't have much data so we just run it on X_train cv
predictions = lr.predict(X_train_cv)
# print(lr.get_depth)
from sklearn import metrics

a = metrics.confusion_matrix(labels,[round(p, 0) for p in predictions])
for i in range(len(labels)):
    if labels[i] != predictions[i]:
        print(labels[i], predictions[i], data[i])
print(a)

recentId = getRecentActivityID(int(argv[1]) if len(argv) > 0 else 0)
print(recentId)
recent = getActivityData(recentId)
datum = titleTypeDescIfy(recent)[1]
print(cleanOneTitleDesc(datum))

a = lr.predict(cv.transform([cleanOneTitleDesc(datum)]))
print(a[0])
updateActivityData(recentId, {"workout_type": int(a[0])})