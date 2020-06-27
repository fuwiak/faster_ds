import numpy as np
import pandas as pd


class BigFiles:
    
    @staticmethod
    def get_info(df):
        raise NotImplementedError

    @staticmethod
    def num_megabytes(df):
        return sum(df.memory_usage()/1024**2)
    
    @staticmethod
    def reduce_to_16bit(X):
        for col in X.columns:
            if X[col].dtype==np.int64:
                X[col] = X[col].astype(np.int16)
            if X[col].dtype==np.float64:
                X[col] = X[col].astype(np.float16)
        return X
    
    @staticmethod
    def reduce_to_category(df):
        df = df.select_dtypes(include=['object']).copy()
        df = df.astype('category')
        return df
    
    @staticmethod
    def dataframe_metainformation(df):
        
         """

        Returns Pandas data frame with metainformation 

        Parameters
        -----------
        df
            Pandas data frame with item data


        Returns
        -----------
        DataFrame, Series
            Pandas data frame with item data

        Examples
        -----------
        data = pd.read_csv("titanic.csv")

        meta = dataframe_metainformation(data)
        print_metainformation(meta)



       """ 
 
   
        meta = dict()
        descr = pd.DataFrame({'dtype': df.dtypes, 'NAs': df.isna().sum()})
        categorical_features = descr.loc[descr['dtype'] == 'object'].index.values.tolist()
        numerical_features = descr.loc[descr['dtype'] != 'object'].index.values.tolist()
        numerical_features_na = descr.loc[(descr['dtype'] != 'object') & (descr['NAs'] > 0)].index.values.tolist()
        categorical_features_na = descr.loc[(descr['dtype'] == 'object') & (descr['NAs'] > 0)].index.values.tolist()
        complete_features = descr.loc[descr['NAs'] == 0].index.values.tolist()
        meta['description'] = descr
        meta['categorical_features'] = categorical_features
        meta['categorical_features'] = categorical_features
        meta['categorical_features_na'] = categorical_features_na
        meta['numerical_features'] = numerical_features
        meta['numerical_features_na'] = numerical_features_na
        meta['complete_features'] = complete_features
        return meta

    @staticmethod
    def print_metainformation(meta):
        print('Available types:', meta['description']['dtype'].unique())
        print('{} Features'.format(meta['description'].shape[0]))
        print('{} categorical features'.format(len(meta['categorical_features'])))
        print('{} numerical features'.format(len(meta['numerical_features'])))
        print('{} categorical features with NAs'.format(len(meta['categorical_features_na'])))
        print('{} numerical features with NAs'.format(len(meta['numerical_features_na'])))
        print('{} Complete features'.format(len(meta['complete_features'])))
    
