##  Table of Contents
- [DataFrame](#dataframe)
  - [Creating a DataFrame](#creating-a-dataframe)
- [Datatypes & conversions](#datatypes--conversions)
   - [Get data types for each column in a data frame](#get-data-types-for-each-column-in-a-data-frame) | [Convert string/object to numeric](#convert-stringobject-to-numeric) | [Other conversions](#other-conversions)
- [Select columns](#select-columns)
   - [Simple select](#simple-select) | [Select from multi-level index using tuples](#select-from-multi-level-index-using-tuples) | [Creating multi-level index using tuples](#creating-multi-level-index-using-tuples) | [Programmatically select columns](#programmatically-select-columns) | [Get distribution of values in a column](#get-distribution-of-values-in-a-column)
- [Filter rows](#filter-rows)
- [Sort](#sort)
   - [Sort rows](#sort-rows) | [Sort columns](#sort-columns)
- [Group by](#group-by)
   - [Group by example](#group-by-example) | [Aggregate functions](#aggregate-functions) | [Transform functions](#transform-functions) | [Group Filters (SQL:having)](#group-filters-sqlhaving) | [Simple Cumulative](#simple-cumulative) | [Expanding Window](#expanding-window) | [Rolling Window](#rolling-window) | [Lead/Lag](#leadlag) | [Resample](#resample)
- [Joins](#joins)
   - [Simple join](#simple-join) | [Concat columns](#concat-columns) | [SQL like join](#sql-like-join)
- [Formatting](#formatting)
   - [Numeric formating](#numeric-formating) | [Gradient Background](#gradient-background) | [Bars](#bars) | [Nesting multiple styles](#nesting-multiple-styles)
- [Transform](#transform)
   - [Swap level & Axis](#swap-level--axis) | [Map](#map) | [Apply](#apply) | [Flatten multi-level column index](#flatten-multi-level-column-index) | [Pivot index levels to column](#pivot-index-levels-to-column) | [Binning based on value (cut)](#binning-based-on-value-cut) | [Binning based on records (qcut)](#binning-based-on-records-qcut)
- [Add/Remove rows & columns](#addremove-rows--columns)
   - [Add Rows](#add-rows) | [Add columns](#add-columns) | [Remove rows](#remove-rows) | [Remove columns](#remove-columns) | [Remove a level from multi-level index](#remove-a-level-from-multi-level-index)
- [Visualizations](#visualizations)
   - [Simple charts](#simple-charts) | [Histogram](#histogram) | [Boxplot for each group](#boxplot-for-each-group) | [Scatter Matrix](#scatter-matrix) | [Bootstrap Plot](#bootstrap-plot)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


##  DataFrame 
You can think of DataFrame like a spreadsheet or SQL table. It is the most commonly used pandas object. 
Rows are referred to as axis 0 and columns as axis 1 and each axis can have multiple levels (which is not common). 

###### Creating a DataFrame  
Below example is to create sample DataFrame for examples on this page
```python
import pandas as pd
df=pd.DataFrame(data=[['apple','fruit',85,3], # Natural line breaking allowed for code within parentheses()
                      ['orange','fruit',68,1],
                      ['banana','fruit',69,2],
                      ['beetroot','vegetable',68,1],
                      ['carrot','vegetable',45,2]],
                columns=['name','type','health_index','cost_index'])
df # prints the dataframe
```
Result:
{F34356671}
##  Datatypes & conversions 
Available datatypes: object (string, lists, dict, tuple and other complex objects), int64, float64, datetime64
```python
#string
s='hello world!' #s[0:2] returns 'he', s[:-1] returns 'hello world'

#list
a=[1,4,5,1,3,6] # a[0] gets first element; a[-1] gets last element. Lists elements can be added or removed with a.append(elem) or a.pop(elem)

#dict
d={'movie_name':'star wars', 'rating':5} # d['movie_name'] gets 'star wars'. Dict elements can be added or removed using d['release_year']=1977 or del d['rating']

#tuple
t=('apple','iphone','7 plus') # similar to list, t[0] gets first element. Tuples are immutable. cannot add/change values

#complex objects (combination of two or more of above)
c={'product':('apple','iphone','7 plus'),'features':['bluetooth','wifi']}

```
###### Get data types for each column in a data frame
```python
df.dtypes # gives the number of columns by various data types in the dataframe df
```
###### Convert int column to float
You can convert each series independently using series.astype function
```python
df['mycol'] = df.mycol.astype(float)
```

###### Convert string/object to numeric 
Pandas auto-detects data type. Usually numeric column with nulls or few string values are treated as object. This piece of code will replace them to 0 and convert to int
```python
df.ID = pd.to_numeric(df.ID, errors='coerce')\ # coerce converts non numeric value columns to NaN
            .fillna(0)\ # replaces NaN to 0
            .astype(np.int64)  # type conversion
```
If there are no issues in your column, you can just run pd.to_numeric like shown below. It will convert to float if you have decimal values or int otherwise.
```python
	df['mycol'] = pd.to_numeric(df.mycol)
```
###### Other conversions
```python
dict(obj) # converts object to dict
list(tup) # converts tuple to list
tuple(list) # converts list to tuple
```

## Select columns 
###### Simple select
```python
df[col1] # Returns series 

df[[col1]] # Returns  dataframe with 1 col
df.loc[:,[col1]] #Same as above, : refers to all rows
df.iloc[:,[0]] # Returns dataframe with 1st col

df[[col1,col2]] # Returns dataframe with 2 cols
df.loc[:,[col1,col2]] # Same as above
df.iloc[:,[0:3]] #Retuns dataframe with first 3 columns (0,1,2)
```
more creative selects from a list [here](https://teamtreehouse.com/community/how-do-i-get-the-first-4-and-last-4-items-of-a-list)
###### Select from multi-level index using tuples 
The tuples () can have 2 or more levels
```python
df[('label','count'),('label','mean')] # selects 2 columns (first under tree label->count and second label->mean)
```

###### Creating multi-level index using tuples
The tuples () can have 2 or more levels
```python
pd.MultiIndex.from_tuples([('Parent1', 'Child1'), ('Parent1', 'Child2')], names=['Parent', 'Child'])
```
Alternatively, you can construct a list and convert to tuple using zip:
```python
arrays = [['Parent1','Parent1'],
                      ['Child1', 'Child2']]
tuples = list(zip(*arrays))
pd.MultiIndex.from_tuples(tuples, names=['Parent', 'Child'])
```
###### Programmatically select columns
```python
df[[col for col in df if 'prob' in col]] # selects all columns containing 'prob' in column name
```

###### Get distribution of values in a column
```python
df.health_index.value_counts() # record count for each val
df.health_index.value_counts(normalize=True) # % distribution
```
##  Filter rows 
Filters internally use masking. You can just run the filter inside to see the mask data frame.
```python
df.loc[df.col1.isin(['apple','orange']),:] # Returns all records where col1 is 'apple' or 'orange'
df.loc[~df.col1.isin(['apple','orange']),:]  # Returns all records where col1 is not 'apple' or 'orange'

```

## Sort 
###### Sort rows
```python
df.sort_values('trips_per_user') #Sorts rows based on values in trips_per_user
```
###### Sort columns
```python
df.sort_index(axis=1,level=0)  # Sorts columns by their names. axis 1 refers to columns. level 0 is the highest/default level
```
## Group by

###### Group by example
groupby returns a dict object with group name and group data frame. This can be iterated thru manually or summarized using the functions below. 
NULL value (NA, NaT) will be ignored and will not form a group.
```python
df.groupby('email_domain') 
df.groupby(column_list)
```
Tip: You could also group by on a field which is not in the selector (even from another dataframe as long as indexes match up)
```python
df['cost_index'].groupby(df.type).sum()

```
Tip: Group by fields are added to index in the resulting dataframe. Use reset_index() to move the index to columns
```python
df.groupby(column_list).sum().reset_index()

```

###### Aggregate functions
apply a common operation on all non grouped columns 
If an aggregate function cannot be applied on a column eg.string, these nuisance columns will be silently ignored
```python
.size() # count
.nunique() # distinct count
.sum() # supports prod, min, max, mean, median, first, last, std, var
.head(5) # returns first 5 rows in each group by order as in dataframe. also supports tail(n), nlargest(n), nsmallest(n) 
.nth([4,-1]) # returns 4th element from the top and 1st element from the bottom

```
apply various operations on each of the columns
```python
.agg({'col1':['mean','count'],'col2':['var']})
```

% of total within groups
```python
df['health_index_%oftotal']=df.health_index/df.health_index.sum()
```
###### Transform functions
Transform function is similar to window aggregate functions on the entire window. It aggregates the data and assigns back to dataset similar to original and preserves index

```python
df.groupby('type')['health_index'].transform('sum') # supports all aggregate functions like min, max, mean, std, var
```

###### Group Filters (SQL:having)
```python
dff.groupby('B').filter(lambda x: len(x['C']) > 2) #Refer Lambda functions
```
## Window aggregate functions 
###### Overall
Applying math function on a series aggregates to the entire group. When applied on group by, returns aggregate value for each group
```python
df.health_index.sum()
```
We could use this to divide on individual values to create % of total columns
```python
df['health_index_%oftotal']=df.health_index/df.health_index.sum()
```

###### Simple Cumulative

```python
df.cost_index.cumsum(asending=False) # similar functions: cummin, cummax, cumprod
```
###### Expanding Window
Another way of achieving cumulative sum:
```python
df.cost_index.expanding().sum()
```
###### Rolling Window
Similar to expanding, rolling window is also supported. Here is how to do a rolling sum of current record and previous (n-1) records 
```python
df.cost_index.rolling(2).sum()
```
###### Lead/Lag

```python
df.cost_index.shift(-2) # Get value from 2 records before
df.cost_index.shift(1) # Get value from next record
```
###### Resample
Used for time series grouping to create bins of equi-split time intervals
```python
# Get another example!
df_re.groupby('group').resample('1D').sum()
```

To check if window aggregate functions are working fine, validate alongside your original dataset like below:
```python
pd.concat([df,df.cost_index.rolling(2,min_periods=1).mean()], axis=1)
```
Note: No exact equivalent of `rows between n preceding and n following` found in pandas. Please update when you find it

## Joins 
###### Simple join
```python
train_X.join(train[y_col]) # joins on index. Returns dataframe with all columns from train_X & y_col from train
```
###### Concat columns
concat on axis 1 is like join based on index
```python
pd.concat([df1, df4], axis=1)
```
###### SQL like join
SQL like join can be achieved with merge:
```python
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
```
Aggregate and join back to original dataset
```python
df.join(df.groupby('month')['A'].sum(), on='month', rsuffix='_r') # aggregates column A to month level and adds that to each of the records in df and is returned
```
## Formatting 
###### Numeric formating
```python
df.style.format({'B': '{:0<4.0f}', 'D': '{:+.2%}'},'C':'±{:.2f}') 
# {:+.2%} is for % with 2 decimals
# ±{:.2f} is for 2 decimals with a +- sign in the front, 
# {:0<4.0f} 0 in front fills zero, < for is left aligned, 4 digits (including - sign) before decimal and no value after decimal
```
Numeric format guides: [here](https://docs.python.org/3/library/string.html#format-specification-mini-language)
###### Gradient Background
```python
cm = sns.light_palette('green', as_cmap=True)
df.style.background_gradient(cmap=cm, subset=['B', 'C', 'D’]) # applies heatmap for columns B, C, D
```
###### Bars
```python
df.style.bar(subset=['B']) # applies bars to column B
```

###### Nesting multiple styles
```python
df.style.format({'B': '{:0<4.0f}', 'D': '{:+.2f}'}).bar(subset=['B']) 
```
More styles: [here](https://pandas.pydata.org/pandas-docs/version/0.18.1/style.html)

## Transform 
###### Swap level & Axis
```python
df.swaplevel(i=0,j=1,axis=1) # swaps first 2 levels (0 & 1) on columns (axis 1)
df.swapindex(0,1) # swaps first 2 index columns
```

###### Map
```python
df['housing'].map({'yes': 1, 'no': 0}) # changes values 'yes' to 1 and 'no' to 0
```
###### Apply

using lambda functions for column transformations (entire column is passed as series to lambda function)
```python
df['delayed'] = df['arr_delay'].apply(lambda col: col > 0)
```
using lambda function across columns using axis=1
```python
df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)
```
###### Flatten multi-level column index

```python
df.columns=df.columns.get_level_values(0) 
```
###### Pivot index levels to column
This is like a pivot table operation. Creates N columns based on N unique values in the row index. If you need to pivot on a non-index column, first add that to index using  pd.set_index(col, append=True)

```python
df=df.unstack() # This unstacks the right most index. To be more specific use order or name like df.unstack(0) or df.unstack('Year')

# Vice versa: to pivot a column level to row index
df=df.stack()
```

###### Binning based on value (cut) 
cut splits a continuous data into equi-distant bins. Bins are skewed when outliers in the data
```python
pd.cut([1,2,3,4,5,6,7,8], 4, labels=False)
# Output: array([0, 0, 1, 1, 2, 2, 3, 3])

pd.cut([1,2,3,4,5,6,7,20], 4, labels=False)
# Output: array([0, 0, 0, 0, 0, 1, 1, 3])
```

###### Binning based on records (qcut)
qcut splits any data into equi-sized bins
```python
pd.qcut([1,2,3,4,5,6,7,20], 4, labels=False)
# Output: array([0, 0, 1, 1, 2, 2, 3, 3])
```

this can be used in a group by to get decile level metrics
```python
df.groupby(pd.qcut(df.col1,10, labels=False)).mean()
```
## Add/Remove rows & columns 
###### Add Rows
Adding 1 row manually
```python
df.append({'name':'banana', 'type':'fruit', 'health_index':88, 'cost_index':98},ignore_index=True) # New index sequence will be created when ignore_index is True. Else will use original index and could result in duplicate indexes
```
Adding rows from another dataframe with matching columns (new columns will be created for non-matching columns. Missing columns will be NULLs
```python
df.append(new_df)
```

###### Add columns
to add a column manually
```python
df.assign('new_col',[1,2,3,4,5])

# Alternate approach
df['new_col']=[1,2,3,4,5]

# Add column dynamically creating a column based on a logic on existing column
df['new_col']=[1 if x>0 else 0 for x in df.col]
```

to create a new data frame with additional column
```python
pd.concat([df,df.cost_index/2], axis=1) # this can also be used to create a dataframe from 2 or more series
```
###### Remove rows
Drop using index
```python
df.drop([0,1]) # removes rows with index 0 & 1
```

###### Remove columns
drop using column names
```python
df.drop('cost_index', axis=1) 
```

###### Remove a level from multi-level index
drop the highest level from multi-level column index
```python
df.columns=df.columns.droplevel()
```

drop a level from multi-level row index
```python
pd.MultiIndex.droplevel(df,level=0)
```

## Visualizations 

###### Simple charts
```python
df.cost_index.plot(kind='bar') # kinds: barh, hist, box, kde, density, area, scatter, hexbin, pie
df.plot(subplots=True, figsize=(6, 6), layout=(3, -1)) # Layout -1 is to infer the number of columns based on data
```
###### Histogram
```python
%matplotlib inline 
df.hist(bins=100, figsize=(40,30))
```

###### Boxplot for each group
```python
%matplotlib inline 
df.groupby('type').boxplot()
```
###### Scatter Matrix
```python
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
```

###### Bootstrap Plot
```python
bootstrap_plot(data, size=50, samples=500, color='grey')
```

More vizualizations [here](https://pandas.pydata.org/pandas-docs/stable/visualization.html)

  Nesting them all 
Use \ for continuity on next line
```python
data[['email_domain',y_col]].\
	.groupby('email_domain')\
	.agg({'label':['mean','count']})\
	.sort_values([('label','count'),('label','mean')], ascending=False)\
        .style\
            .format({'B': '{:0<4.0f}', 'D': '{:+.2f}'})\
            .bar(subset=['B'])
```
