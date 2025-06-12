import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

data_dir = 'C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\kaggle\\optimal-fertilizer'

# %%

train_df = pd.read_csv(data_dir + '\\train.csv')
test_df = pd.read_csv(data_dir + '\\test.csv')

train_id = np.array(train_df['id'])
test_id = np.array(test_df['id'])

cat_cols = ['Soil Type', 'Crop Type']
num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

le = LabelEncoder()
y = le.fit_transform(np.array(train_df['Fertilizer Name']))
train_df = train_df.drop(columns=['Fertilizer Name', 'id'])

for col in cat_cols:
    train_df[col] = train_df[col].astype('category').cat.codes

# %%

def mae(actual, predicted):
    total_number = len(actual)
    res = np.sum(actual==predicted)/total_number
    return res

def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    print(actual.shape, predicted.shape)
    print('-'*100)
    #print(type(actual), type(predicted))
    #print('-'*100)
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  # only the first correct prediction counts
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

mapk_scorer = make_scorer(mapk, response_method='predict_proba')
    
# %%

param_grid = {
    'max_depth': [10, 20],#[3, 5, 10, 20, 30],
    'learning_rate': [0.11]#[0.1, 0.01, 0.001],
}

#K=5
#skf = StratifiedKFold(n_splits=K, shuffle = True, random_state = 1001)

model = XGBClassifier(min_child_weight=5.533830209405815, gamma=0.2845805417802597, 
    alpha=2.9500411716472144, subsample=0.5998720852200778, 
    colsample_bytree=0.4193001268301755, eta=0.5271936074396966, 
    n_estimators=10, reg_lambda=0.01059616433916218,
    objective='multi:softprob', enable_categorical=True,
    tree_method='hist', device='cpu',
    n_jobs=-1)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring=mapk_scorer, n_jobs=-1, verbose=0)

grid_search.fit(train_df, y)

print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)





















































































