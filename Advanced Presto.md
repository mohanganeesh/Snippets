## Array

#### Aggregate columnn values to array
```sql
select ARRAY_AGG(col) from tablename
```

#### Combine arrays to return unique values across all arrays (Note: will lose multiple occurances of same value)
```sql
select SET_UNION(array_col) from tablename
```

#### Create array from strings
```sql
select ARRAY['val1', 'val2', 'val3'] as array_col
```

#### Explode Array into rows
```sql
SELECT col.val
FROM tablename
CROSS JOIN UNNEST(array_col) as col(val)
```

## Aggregation

#### Col value corresponding to max value of other col
```sql
MAX_BY(event_name, event_time) --returns last event name (i.e) event name corresponding to max event time
```

#### Return a arbitrary value to break ties
```sql
ARBITRARY(col)
```

#### Get multiple percentile values
```sql
APPROX_PERCENTILE(col, ARRAY[0.1, 0.5, 0.9]) -- Returns an array of 10th, 50th, 90th percentiles for col
```

## JSON

#### Extract value for Key
```sql
JSON_EXTRACT(map_col, 'key') --Gets value for key from map_col
```

#### Parse string to JSON array
```sql
CAST(JSON_PARSE(str_col) AS ARRAY<JSON>)
```

## Advanced Data Types

#### Create MAP / STRUCT (ROW) / ARRAY 
```sql
  MAP(ARRAY['col1','col2','col3'],ARRAY[col1,col2,col3]) lm, --Creates map, not struct
  CAST(ROW(col1,col2,col3) as ROW(col1 int, col2 int, col3 int)) ls, --Creates struct with names. Downside is requires defining datatypes
  ROW(col1,col2,col3) ls2, --Creates Struct without name but with index. Need to access with position like an array
  ARRAY[col1,col2,col3] la --Creates an array
```

#### Select from MAP / STRUCT (ROW) / ARRAY 

```sql
SELECT lm['col1'] --Select from MAP
SELECT ls.col1 --Select from STRUCT
SELECT ls2[1] --Selects from STRUCT without name using index. Index starts at 1?
SELECT la[1] --Selects from ARRAY with index
```

#### Get all keys in a MAP
```sql
SELECT k.key
FROM tablename
CROSS JOIN UNNEST(MAP_KEYS(mapcol)) as k(key)
GROUP BY 1
```
