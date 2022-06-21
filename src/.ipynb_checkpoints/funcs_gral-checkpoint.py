import pandas as pd
import numpy as np
import seaborn as sns
import os,json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools as it
import pandas as pd
import numpy as np
import seaborn as sns
import os,json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools as it
from datetime import date, datetime, timedelta

######################################################################################################################


def null_clean(df, subset=None):
    print('Every row with null values in the columns present in the subset will be removed')
    old_shape = df.shape[0]
    df = df.dropna(subset=subset)
    print(f'Hey! {old_shape - df.shape[0]} were removed due to null values')
    old_shape = df.shape[0]
    return df