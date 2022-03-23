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

formula="success_metric ~ treatment_ind + [intermediate_outcome ~ instrument_variable]"
model = IV2SLS.from_formula(formula, data=df)
results = model.fit()
print(results)
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
