# Sample datasets

This folder contains small CSV files used in the unit tests and the examples. They are handy for quickly trying out the library without having to download any external data.

## titanic.csv

A subset of the famous Titanic passenger list. Each row represents a passenger and the `Survived` column is a binary label indicating whether that person survived the disaster. You can use it to build a simple classification model.

Columns include:

- `PassengerId` – unique passenger identifier
- `Survived` – target label (0 = did not survive, 1 = survived)
- `Pclass` – ticket class
- `Name`, `Sex`, `Age` – basic personal information
- `SibSp`, `Parch` – number of siblings/spouses or parents/children aboard
- `Ticket`, `Fare`, `Cabin`, `Embarked` – ticket details

## house-prices.csv

Synthetic house price data for regression tasks. The goal is to predict the `Price` column based on features such as the size of the house and the neighbourhood.

Columns include:

- `Home` – row identifier
- `Price` – target value (house price)
- `SqFt`, `Bedrooms`, `Bathrooms` – property characteristics
- `Offers` – number of purchase offers received
- `Brick` – whether the house is made of brick
- `Neighborhood` – area of the city (`East`, `West`, `North` ...)

## iris.csv

Classic Iris flower measurements. It can be used for clustering or classification. The `variety` column gives the species of the plant.

Columns include:

- `sepal.length`, `sepal.width`, `petal.length`, `petal.width`
- `variety` – Setosa, Versicolor or Virginica

## Using the data with `faster_llm`

Each dataset can be loaded with `pandas` and passed into the high level models provided by the library. Metrics are calculated automatically and can optionally be forwarded to an MCP compatible service.

Example using the Titanic data:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faster_llm.ML.classification import Model

# load the CSV
frame = pd.read_csv("./sample_data/titanic.csv", sep="\t")
X = frame.drop(columns=["Survived"])
y = frame["Survived"]

# train and send metrics to an MCP server
clf = Model(
    LogisticRegression(max_iter=200),
    X,
    y,
    send_to_llm_flag=True,
    server_url="http://localhost:8000",
)
```

For regression you can use `house-prices.csv` with `faster_llm.ML.regression.Model` in the same way. Clustering with `iris.csv` can be performed via `faster_llm.ML.clasterization.ClusterModel`.

To send arbitrary results to an MCP server you may also call `send_to_llm` directly:

```python
from faster_llm.LLM import send_to_llm
send_to_llm("Training complete", server_url="http://localhost:8000")
```
