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

def mae(actual, predicted):
    total_number = len(actual)
    res = np.sum(actual==predicted)/total_number
    return res

def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    #print('-'*100)
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

mapk_scorer = make_scorer(mapk, k = 3)

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

# %%

for col in cat_cols:
    train_df[col] = train_df[col].astype('category').cat.codes

# %%

X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2, random_state=0, shuffle=True)

# %%

model = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=30, min_samples_split=200, 
                               min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                               max_leaf_nodes=None, n_jobs=-1, random_state=0)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

mae_train = mae(y_train, pred_train)
mae_pred = mae(y_val, pred_val)

proba_train = model.predict_proba(X_train)
proba_val = model.predict_proba(X_val)

top3_train = np.argsort(proba_train, axis=1)[:, ::-1][:, :3]
top3_val = np.argsort(proba_val, axis=1)[:, ::-1][:, :3]

mapk_train = mapk(y_train, top3_train)
mapk_pred = mapk(y_val, top3_val)

print(f'{mae_train:0.5f}')
print(f'{mae_pred:0.5f}')

print(f'{mapk_train:0.5f}')
print(f'{mapk_pred:0.5f}')

# %%

'''
model = XGBClassifier(learning_rate=0.04035529891870569, max_depth=20, 
    min_child_weight=5.533830209405815, gamma=0.2845805417802597, 
    alpha=2.9500411716472144, subsample=0.5998720852200778, 
    colsample_bytree=0.4193001268301755, eta=0.5271936074396966, 
    n_estimators=100, reg_lambda=0.01059616433916218,
    objective='multi:softprob', enable_categorical=True,
    tree_method='hist', device='gpu', 
    n_jobs=-1)
'''

scan = [0.1+val*0.01 for val in range(11)]

for value in scan:
    start_time = time()
    
    model = XGBClassifier(learning_rate=value, max_depth=20, 
        min_child_weight=5.533830209405815, gamma=0.2845805417802597, 
        alpha=2.9500411716472144, subsample=0.5998720852200778, 
        colsample_bytree=0.4193001268301755, eta=0.5271936074396966, 
        n_estimators=100, reg_lambda=0.01059616433916218,
        objective='multi:softprob', enable_categorical=True,
        tree_method='hist', device='gpu', 
        n_jobs=-1)
    
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    
    mae_train = mae(y_train, pred_train)
    mae_pred = mae(y_val, pred_val)
    
    proba_train = model.predict_proba(X_train)
    proba_val = model.predict_proba(X_val)
    
    top3_train = np.argsort(proba_train, axis=1)[:, ::-1][:, :3]
    top3_val = np.argsort(proba_val, axis=1)[:, ::-1][:, :3]
    
    mapk_train = mapk(y_train, top3_train)
    mapk_pred = mapk(y_val, top3_val)
    
    print(f'Scan Value: {value}')
    
    print(f'MAE train:  {mae_train:0.5f}')
    print(f'MAE pred:   {mae_pred:0.5f}')
    
    print(f'Mapk train: {mapk_train:0.5f}')
    print(f'Mapk pred:  {mapk_pred:0.5f}')
    
    duration = time()-start_time
    
    #print(f'Duration: {duration:0.2f}s')
    print()

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
    tree_method='hist', device='gpu',
    n_jobs=-1)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring=mapk_scorer, n_jobs=-1, verbose=0)

grid_search.fit(train_df, y)

print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# %%

from sklearn.metrics import fbeta_score

fbs = make_scorer(fbeta_score, beta=2)





















































































