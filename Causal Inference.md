## Regression
```py
import statsmodels.formula.api as smf

formula = “success_metric ~ treatment_ind + confounder1 + confounder2” # all columns need to be float
model = smf.ols(formula, data=df)
results = model.fit()
print(results.summary())
```

## Instrument Variable
```py
from linearmodels.iv import IV2SLS

formula="success_metric ~ treatment_ind + [intermediate_outcome ~ iv_var1 + iv_var2]" # Try to achieve strong stage 1 with limited iv vars
model = IV2SLS.from_formula(formula, data=df)
results = model.fit()
#print(results)
parse(results)
```


## Diff-in-diff
```py
# Using a model
import statsmodels.formula.api as smf

formula = “V ~ D1 + D2 + D1.D2” # WIP (should have one variable for date, one for variable, one to capture interaction. outcome is total metric across variable
model = smf.ols(formula, data=df)
results = model.fit()
print(results.summary())

# Using simple arithmetic
agg_df = df.groupby([‘D1’,’D2’]).sum().unstack()
agg_df.V.astype('float').apply(np.log).groupby('D2').diff().groupby('D1').agg(['diff']).apply(np.exp)-1

```

## Mathing Estimator
```py

#PREP: Scale Features
X = ["severity", "age", "sex"]
y = "recovery"

med = med.assign(**{f: (med[f] - med[f].mean())/med[f].std() for f in X})
med.head()

#SCORING, MATCHING, ATE with bias correction
from sklearn.linear_model import LinearRegression

# fit the linear regression model to estimate mu_0(x)
ols0 = LinearRegression().fit(untreated[X], untreated[y])
ols1 = LinearRegression().fit(treated[X], treated[y])

# find the units that match to the treated & vice-versa
treated_match_index = mt0.kneighbors(treated[X], n_neighbors=1)[1].ravel()
untreated_match_index = mt1.kneighbors(untreated[X], n_neighbors=1)[1].ravel()

predicted = pd.concat([
    (treated
     # find the Y match on the other group
     .assign(match=mt0.predict(treated[X])) 
     
     # build the bias correction term
     .assign(bias_correct=ols0.predict(treated[X]) - ols0.predict(untreated.iloc[treated_match_index][X]))),
    (untreated
     .assign(match=mt1.predict(untreated[X]))
     .assign(bias_correct=ols1.predict(untreated[X]) - ols1.predict(treated.iloc[untreated_match_index][X])))
])
```

## Propensity Scoring (IPTW) with bootstrap for SE
```py
from joblib import Parallel, delayed # for parallel processing

# define function that computes the IPTW estimator
def run_ps(df, X, T, y):
    # estimate the propensity score
    ps = LogisticRegression(C=1e6).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    
    weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
    return np.mean(weight * df[y]) # compute the ATE

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(run_ps)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                          for _ in range(bootstrap_sample))
ates = np.array(ates)

print(f"ATE: {ates.mean()}")
print(f"95% C.I.: {(np.percentile(ates, 2.5), np.percentile(ates, 97.5))}")


#NOTE: Always run positivity check to ensure results are useful
sns.distplot(data_ps.query("intervention==0")["propensity_score"], kde=False, label="Non Treated")
sns.distplot(data_ps.query("intervention==1")["propensity_score"], kde=False, label="Treated")
plt.title("Positivity Check")
plt.legend();

```

## Doubly Robust Estimator with bootstraping for SE
```py

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e6, max_iter=1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

from joblib import Parallel, delayed # for parallel processing

np.random.seed(88)
# run 1000 bootstrap samples
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(doubly_robust)(data_with_categ.sample(frac=1, replace=True), X, T, Y)
                          for _ in range(bootstrap_sample))
ates = np.array(ates)

print(f"ATE 95% CI:", (np.percentile(ates, 2.5), np.percentile(ates, 97.5)))
```

## PanelOLS Estimator (for Panel data with multiple snapshots of same user)
```py

from linearmodels.panel import PanelOLS
mod = PanelOLS.from_formula("outcome ~ var1+var2+EntityEffects+TimeEffects",
                            data=data.set_index(["userid", "date"]))

result = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
result.summary.tables[1]
```

## Synthetic Control
```py
#Step 0: Define functions
from typing import List
from operator import add
from toolz import reduce, partial
from scipy.optimize import fmin_slsqp

def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W))**2))

def get_w(X, y):
    
    w_start = [1/X.shape[1]]*X.shape[1]

    weights = fmin_slsqp(partial(loss_w, X=X, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights

#Example used below to form synthetic control of states; update naming to entity used
def synthetic_control(state: int, data: pd.DataFrame) -> np.array:
    
    features = ["cigsale", "retprice"] # Metrics which should be match in pre-period
    inverted = (data.query("~after_treatment")
                .pivot(index='state', columns="year")[features]
                .T)
    y = inverted[state].values # treated
    X = inverted.drop(columns=state).values # donor pool
    
    weights = get_w(X, y)
    synthetic = (data.query(f"~(state=={state})")
                 .pivot(index='year', columns="state")["cigsale"]
                 .values.dot(weights))

    return (data
            .query(f"state=={state}")[["state", "year", "cigsale", "after_treatment"]]
            .assign(synthetic=synthetic))
          
#Step 1: Generate synthetic pool
from joblib import Parallel, delayed

control_pool = cigar["state"].unique()
parallel_fn = delayed(partial(synthetic_control, data=cigar))
synthetic_states = Parallel(n_jobs=8)(parallel_fn(state) for state in control_pool)

#Step2: Estimate lifts

```

## Regression Discontinuity
```py
from joblib import Parallel, delayed 

def wald_rdd(data):
    weights=kernel(data["minscore"], c=0, h=15)*data["n"]
    denominator = smf.wls("receivehsd~minscore*threshold", data, weights=weights).fit()
    numerator = smf.wls("avgearnings~minscore*threshold", data, weights=weights).fit()
    return numerator.params["threshold"]/denominator.params["threshold"]

np.random.seed(45)
bootstrap_sample = 1000
ates = Parallel(n_jobs=4)(delayed(wald_rdd)(sheepsking_rdd.sample(frac=1, replace=True)) #Is RDD same as DF?
                          for _ in range(bootstrap_sample))
ates = np.array(ates)

```
