## Preprocessing
### Categorical variables 
(More types: https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)
#### 1. One Hot encoding
```py
df = pd.get_dummies(df, columns=[categ_column_names]) # Use df.columns[df.dtypes == object] to review categ columns
df.columns = [str(x).replace(' ','_').replace(',','_').replace('.','_') for x in df.columns]
```
#### 2. Label encoding
```py
df['new_col'] = pd.factorize(df.categ_col)[0].reshape(-1,1)
```
(or)
```py
from sklearn.preprocessing import LabelEncoder

df['new_col'] = LabelEncoder.fit_transform(df.categ_col)
```
#### 3. Ordinal encoding
#### 4. Frequency encoding
#### 5. Weight of Evidence encoding


### Convert to Float
```py
df[numeric_cols] = df[numeric_cols].astype(‘float’)
```

### Balanced Sampling
```py
df_balanced = pd.concat([df.loc[df[y_col]==1], \
	df.loc[df[y_col]==0].sample(sum(df[y_col]==1)) ], axis = 0).reset_index()  # Swap 0 & 1 when 1 needs to sampled down
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

### Missing values
1. Create a new Category
2. Delete
3. Impute from Average
4. Impute from Last Value

### Dimensionality Reduction

## Modeling

### TimeSeries Forecasting

### Regression

### Classification
#### XGBoost Classifier
```py
from xgboost import XGBClassifier
clf = XGBClassifier()

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_train_pred=clf.predict(X_train)
```

#### Logit Classifier
```py
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_test_pred_prob = clf.predict_proba(X_test)
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
```

### Clustering
#### Gaussian Mixture Modeling (GMM)
```py
from sklearn.mixture import GaussianMixture

df['cluster'] = GaussianMixture(n_components=3).fit_predict(df[cluster_cols]) # Where 3 is the number of clusters
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
