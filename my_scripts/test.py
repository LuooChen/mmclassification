import numpy as np

test = [{'a':1}
        ,{'a':2}
        ,{'a':3}
        ,{'a':4}
        ]

test = [x for x in test if x['a'] >= 2]
print(test)
