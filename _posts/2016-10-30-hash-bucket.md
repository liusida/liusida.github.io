---
layout: post
title:  About sparse_column_with_hash_bucket
---

Yesterday, I saw tf.contrib.layers.sparse_column_with_hash_bucket in a [tutorial][tensorflow-tutorial]. That's a very useful function! I thought. I never met such a function in Keras or TFLearn.

Basically, the function do something like this:

```python
hash(category_string) % dim
```

Let's say the text "the quick brown fox". If we want to put them into 5 buckets, we can get result like this:

```python
hash(the) % 5 = 0
hash(quick) % 5 = 1
hash(brown) % 5 = 1
hash(fox) % 5 = 3
```
This example is metioned by [Luis Argerich][luis-in-quora]

That's really easy for preprocessing, but there are disadvantages of that, metioned by [Artem Onuchin][luis-in-quora] also in that page.

So, the common way to do this **feature engineering** thing is metioned by [Rahul Agarwal][practice-quora]:

* Scaling by Max-Min
* Normalization using Standard Deviation
* Log based feature/Target: use log based features or log based target function.
* One Hot Encoding

Anyway, if we want to do hash_bucket without tensorflow, we can do it in Pandas which is metioned [here][stackoverflow-hash-code]:

```python
import pandas as pd
import numpy as np

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

data = pd.DataFrame(data)

def hash_col(df, col, N):
    cols = [col + "_" + str(i) for i in range(N)]
    print(cols)
    def xform(x): tmp = [0 for i in range(N)]; tmp[hash(x) % N] = 1; return pd.Series(tmp,index=cols)
    df[cols] = df[col].apply(xform)
    return df.drop(col,axis=1)

print(hash_col(data, 'state',4))
```
result:
```code
   pop  year  state_0  state_1  state_2  state_3
0  1.5  2000        1        0        0        0
1  1.7  2001        1        0        0        0
2  3.6  2002        1        0        0        0
3  2.4  2001        1        0        0        0
4  2.9  2002        1        0        0        0
```

**\[Edited\]** Actually, we can use [pandas.get_dummies][doc-get_dummies] to do this directly:

```python
import pandas as pd

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

data = pd.DataFrame(data)
print(pd.get_dummies(data, columns=['state','year']))
```
result:
```code
   pop  state_Nevada  state_Ohio  year_2000  year_2001  year_2002
0  1.5           0.0         1.0        1.0        0.0        0.0
1  1.7           0.0         1.0        0.0        1.0        0.0
2  3.6           0.0         1.0        0.0        0.0        1.0
3  2.4           1.0         0.0        0.0        1.0        0.0
4  2.9           1.0         0.0        0.0        0.0        1.0
```

After all,

I think I should learn more about one-hot-encoding and word2vec embedding.

> Coming up with features is difficult, time-consuming, requires expert knowledge. "Applied machine learning" is basically feature engineering.

Said [Andrew Ng][ng].


[tensorflow-tutorial]:https://www.tensorflow.org/versions/r0.11/tutorials/wide/index.html
[luis-in-quora]:https://www.quora.com/Can-you-explain-feature-hashing-in-an-easily-understandable-way
[practice-quora]:https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering
[stackoverflow-hash-code]:http://stackoverflow.com/questions/8673035/what-is-feature-hashing-hashing-trick/33581487
[ng]:http://www.andrewng.org/
[doc-get_dummies]:http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
