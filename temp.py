from faster_ds import preprocessing as pr
from faster_ds import dimension_reduction as dr
import pandas as pd

df = pr.PR.csv_as_df("/Users/macbookssd/Desktop/faster_ds/sample_data/titanic.csv")

# df.fillna(0, inplace=True)
# df = pd.get_dummies(df)

# dr.PCA.pca2df(df, 'Survived',2)



https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-a-single-and-a-double-underscore-before-an-object-name