import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_dir = 'C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\kaggle\\optimal-fertilizer'

# %%

def mae(corr, pred):
    total_number = len(corr)
    res = np.sum(corr==pred)/total_number
    return res

# %%

train_df = pd.read_csv(data_dir + '\\train.csv')
test_df = pd.read_csv(data_dir + '\\test.csv')

train_id = np.array(train_df['id'])
test_id = np.array(test_df['id'])

y = np.array(train_df['Fertilizer Name'])
train_df = train_df.drop(columns=['Fertilizer Name', 'id'])

train_df = train_df.drop(columns=['Soil Type', 'Crop Type'])

# %%

X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2, random_state=0, shuffle=True)

# %%

model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=20, min_samples_split=50, 
                               min_samples_leaf=3, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                               max_leaf_nodes=None, n_jobs=-1, random_state=0)

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

mae_train = mae(y_train, pred_train)
mae_pred = mae(y_val, pred_val)

print(f'{mae_train:0.5f}')
print(f'{mae_pred:0.5f}')

# %%




















































































