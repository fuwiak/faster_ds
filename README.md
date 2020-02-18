# faster_ds
Open Source Numpy/Pandas based module to make faster data analysis in console


## Instruction for contributors

click **fork** or type in console

- git clone https://github.com/fuwiak/faster_ds
- cd faster_ds
- create virtualenv
- virtualenv -p python3 env_name
- source env_name/bin/activate
- pip install -r requirements.txt

Do you have any questions or suggest? Please edit this file -----> *feature_request.md*

Please check out function_description_style.py
Please write code tests.

# preprocessing


# visualization


# feature selecting
- pipeline
- filter, wrapper, embedded

# evaluation and tuning


# classification(binary)

- methods
- pipeline
- visualizations

# multiclass classification

# regression


- methods
- pipeline

# clasterization

- methods
- pipeline

# dimension reduction

# fake(sample) data
Sample usage:
```{Python}
nrow=10
df = fake_data()
df = df.classification_data(nrow)
print(df)

```
>  Output

```
Prefix                Name  Birth Date           Phone Number  ...  Year      Time                            Link HaveAjob
0    Dr.    Nathan Hernandez  18-06-1989  001-153-601-9176x8231  ...  1992  03:23:14          https://www.moore.com/        0
1    Dr.       Cameron Jones  20-01-2019          (175)943-0445  ...  2009  19:59:41          http://www.martin.org/        1
2   Mrs.       Kathryn Drake  15-11-2016          (887)351-7584  ...  2017  20:18:15              http://farmer.com/        0
3    Dr.      Lauren Garrett  27-05-1988  +1-180-033-4897x95513  ...  2002  14:50:13              https://patel.com/        1
4    Dr.     Victoria Murphy  25-06-1987          (603)268-1434  ...  1988  20:19:35             http://russell.com/        1
5    Dr.        Claudia Huff  03-11-1975  +1-244-015-1248x47384  ...  1970  17:28:17         http://www.freeman.biz/        0
6   Mrs.      Crystal Thomas  26-02-2011       780.307.6060x053  ...  2017  02:36:27    http://www.ortiz-miller.com/        1
7   Mrs.  Angelica Zimmerman  12-12-2009          (554)926-4554  ...  2018  13:42:11   http://www.roberts-ellis.com/        1
8    Dr.      Keith Knapp MD  11-03-1982       001-075-523-8781  ...  1999  03:04:56  http://www.phillips-black.com/        0
9   Mrs.        Rachel Allen  10-12-1998     418-099-0859x35240  ...  1989  09:13:26          https://www.evans.com/        ```




