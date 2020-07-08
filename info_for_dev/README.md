# TIPS

Your function should:
- Well documented as in sample below:

```{python}


def abc(self, arg1, arg2,...):
    pass


 """
    description what function returns
    Returns .....
    
    description parameters(like below)
        Parameters
        -----------
        item_data_frame
            Pandas data frame with item data
        customer_data_frame
            Pandas dataframe with customer data
        item_id_selected
            itemid of the item
            for which we are making recommendation
        item_X_item_matrix
            A symetric numpy matrix wher index [i, j]
            represents the similarity between the item `i` and item `j`.
        n
            number of items to recommend
   
           
        See Also
        --------
        abc1.iloc : Int
        abc2.iloc :
        abc3.to_frame : Inverse of DataFrame.squeeze
        
        Examples
        --------
        
        >>> primes = pd.Series([2, 3, 5, 7])
        Slicing might produce a Series with a single value:
        >>> even_primes = primes[primes % 2 == 0]
        
        
        Raises
        ------
     """

   
    return foo
```

# Feature request 

https://github.com/fuwiak/faster_ds/blob/master/.github/ISSUE_TEMPLATE/feature_request.md


