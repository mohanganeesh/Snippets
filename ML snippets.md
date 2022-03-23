## Preprocessing
### One Hot encoding
```py
df = pd.get_dummies(df, columns=[categ_column_names])
df.columns = [str(x).replace(' ','_').replace(',','_').replace('.','_') for x in df.columns]
```

### Convert to Float
```py
df[numeric_cols] = df[numeric_cols].astype(‘float’)
```

### Balanced Sampling
```py
df_balanced = pd.concat([df.loc[df.ylabel==1], \
	df.loc[df.ylabel==0].sample(sum(df.ylabel==1)) ], axis = 0).reset_index(). # Swap 0 & 1 when 1 needs to sampled down
```

### Train Test Split
```py
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[xcols],
    df.ycol,
    test_size=0.40, # 20% recommended. Use higher for smaller samples
    random_state=42,
)
```

## Model Fitting

### Classifier - XGBoost
```py
from xgboost import XGBClassifier
clf = XGBClassifier()

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_train_pred=clf.predict(X_train)
```


### Classifier - Logit
```py
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_test_pred_prob = clf.predict_proba(X_test)
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
```

## Evaluation

### Classifier Evaluation
```py
from sklearn import metrics
metrics.accuracy_score(y_test, clf.predict(X_test)) 
metrics.precision_score(y_test, clf.predict(X_test))
metrics.recall_score(y_test, clf.predict(X_test))
metrics.f1_score(y_test, clf.predict(X_test)) # binary
```

### Decile table
```py
result_df = pd.DataFrame({'pred_prob': y_test_pred_prob[:,1], 'pred': y_test_pred, 'actual': y_test}).reset_index(drop=True)
result_df['pred_prob_qtile'] = pd.qcut(result_df.pred_prob, 10, labels = range(0,10))
result_df.groupby('pred_prob_qtile').mean()
```
