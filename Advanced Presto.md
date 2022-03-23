## Array

#### Aggregate string to array
```sql
select SET_UNION(string_col) from tablename
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
