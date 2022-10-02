# %% [markdown]
# <h1> Importing Libraries </h1>

# %%
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
from tld import get_tld, is_tld

# %%
data = pd.read_csv("/Users/junlongng/Desktop/NTU/Year 2/URL_Detection/malicious_phish.csv")
data.head()

# %% [markdown]
# <h1> Data Exploration </h1>

# %%
data.info()

# %%
data.isnull().sum()

# %%
count = data.type.value_counts()
count

# %%
sns.barplot(x=count.index, y=count)
plt.xlabel("Types")
plt.ylabel("Count")

# %%
## Removing the "www" prefix attached to each email, this will not interfere with the model later on
data["url"] = data["url"].replace("www","", regex=True)
data

# %%
data.head()

# %% [markdown]
# <h1> Creating functions & Categorising Data </h1>

# %%
## Categorising the values:
rem = {"Category": {"benign":0, "defacement":1, "phishing":2, "malware":3}}
data["Category"] = data["type"]
data = data.replace(rem)

# %% [markdown]
# <h1> Feature Extraction </h1>

# %%
data['url_len'] = data['url'].apply(lambda x: len(str(x)))


# %%
def process_tld(url):
    try:
        res = get_tld(url, as_object = True, fail_silently=False,fix_protocol=True)
        pri_domain= res.parsed_url.netloc
    except :
        pri_domain= None
    return pri_domain

# %%
data['domain'] = data['url'].apply(lambda i: process_tld(i))


# %%
data.head()


# %%
feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for a in feature:
    data[a] = data['url'].apply(lambda i: i.count(a))

# %%
data.head()


# %%
def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0

# %%
data['abnormal_url'] = data['url'].apply(lambda i: abnormal_url(i))


# %%
sns.countplot(x='abnormal_url', data=data)



# %%
def httpSecure(url):
    htp = urlparse(url).scheme
    match = str(htp)
    if match=='https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0

# %%
data['https'] = data['url'].apply(lambda i: httpSecure(i))


# %%
sns.countplot(x='https', data=data);


# %%
## The below code will show how many digits are there in a url
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
    

# %%
data['digits']= data['url'].apply(lambda i: digit_count(i))


# %%
def letter_counter(url):
    letter_count = 0
    for i in url: 
        if i.isalpha():
            letter_count += 1
        return letter_count    

# %%
data['letters']= data['url'].apply(lambda i: letter_counter(i))


# %%
## Creating a function to determine if the url has been shortened

def Shortining_Service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

# %%
data['Shortining_Service'] = data['url'].apply(lambda x: Shortining_Service(x))


# %%
sns.countplot(x='Shortining_Service', data=data);


# %%
## Creating a function that can track if the url has IP address

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

# %%
data['having_ip_address'] = data['url'].apply(lambda i: having_ip_address(i))


# %%
data['having_ip_address'].value_counts()
## There are 638703 without an IP address
## There are 12488 with an IP address

# %%
plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), linewidths=.5)

# %%
X = data.drop(['url','type','Category','domain'],axis=1)#,'type_code'
y = data['Category']

# %% [markdown]
# <h1> Train Test Split </h1>

# %%
## Train Test Split shall be 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# %%
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

# %% [markdown]
# <h1> Training The Model </h1>

# %%
models = [DecisionTreeClassifier,RandomForestClassifier,AdaBoostClassifier,KNeighborsClassifier,SGDClassifier,
         ExtraTreesClassifier,GaussianNB]
model_names = ["DT", "RF", "AB", "KN", "SGD", "ET", "G"]
i = 0
cachedir = '/Users/junlongng/Desktop/NTU/Year 2/URL_Detection/Model_Cache'
for m in models:
    print('#############################################')
    print('######-Model =>\033[07m {} \033[0m'.format(m))
    model_ = m()
    model_.fit(X_train, y_train)
    joblib.dump(model_, cachedir+'/'+model_names[i])
    i+=1
    pred = model_.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print('Test Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(acc*100))
    print('\033[01m              Classification_report \033[0m')
    print(classification_report(y_test, pred))
    print('\033[01m             Confusion_matrix \033[0m')
    cf_matrix = confusion_matrix(y_test, pred)
    plot_ = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,fmt= '0.2%')
    plt.show()
    print('\033[31m###################- End -###################\033[0m')




