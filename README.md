# faster_llm

`faster_llm` is a Python library that speeds up common data preprocessing and analysis tasks. It builds on familiar tools like NumPy, Pandas and scikit-learn while adding connectors for Large Language Model (LLM) frameworks such as LangChain and LlamaIndex.  Recent updates introduce lightweight helpers for other agent libraries including AutoGen and CrewAI.

## Overview

`faster_llm` bundles together many of the repetitive steps you face when preparing data and training models.  It provides helpers for loading CSV files, splitting columns into features and targets and handling missing values.  You will also find utilities for feature selection, normalisation and encoding of categorical variables.  Ready‑made pipelines wrap popular scikit‑learn and XGBoost algorithms so you can quickly train classification, regression or clustering models.  Results can be sent straight to your favourite LLM framework via the built‑in MCP client.

## Features

- Utilities for splitting, encoding and cleaning tabular data
- Ready‑to‑use pipelines for classical machine learning workflows
- Helpers for passing results to LLM frameworks
- Feature selection, model evaluation and hyperparameter tuning tools
- Wrappers around common scikit‑learn and XGBoost models
- Lightweight wrappers for Keras, PyTorch and PyTorch Lightning
- Simple generator for producing fake data for experiments
- MCP client for forwarding messages to LLM services and AI agents
- Wrappers for LangChain, LlamaIndex, AutoGen and CrewAI agents

## Quick Demo

Fancy a lightning-fast example? Generate a fake dataset, train a model and
have the metrics sent to your favourite LLM in just a few lines:

```python
from faster_llm.Tools.generate_fake_data import FakeData
from faster_llm.ML.classification import Model
from sklearn.linear_model import LogisticRegression

data = FakeData.classification_data(100)
X = data.drop(columns=["HaveAjob"])
y = data["HaveAjob"]

model = Model(LogisticRegression(max_iter=200), X, y, send_to_llm_flag=True)
print(model._compute_metrics())
```

## Install from PyPI

```bash
pip install faster_llm
```

After installation you can simply import the package:

```python
import faster_llm
```

## Run Locally

To work with the latest development version:

```bash
git clone https://github.com/fuwiak/faster_llm.git
cd faster_llm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the tests to verify everything is set up correctly:

```bash
pytest
```

## Examples

Take a look at the `examples/` directory for runnable snippets. The
`classification_example.py` script trains a small logistic regression
model on the included Iris dataset and prints evaluation metrics via the
`send_to_llm` helper:

```bash
python examples/classification_example.py
```

The same helper can forward metrics from Keras, PyTorch or PyTorch Lightning
models to AI agents by using the library's MCP integration.
The new ``agents`` module also exposes utility functions to create LangChain
``AIMessage`` objects, LlamaIndex ``Document`` instances, run simple CrewAI
crews or spin up AutoGen assistants directly from your code.

### Interpret metrics with MCP

You can explicitly send your evaluation metrics to an MCP compatible service
and retrieve its interpretation. The helper below trains a simple model,
sends the metrics using the low level ``send_mcp_request`` function and prints
the agent's reply:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from faster_llm.ML.classification import Model
from faster_llm.LLM.mcp import send_mcp_request

df = pd.read_csv("./sample_data/iris.csv")
X = df.drop(columns=["variety"])
y = (df["variety"] == "Virginica").astype(int)

model = Model(LogisticRegression(max_iter=200), X, y)
metrics = model._compute_metrics()

response = send_mcp_request(
    "deliver_message",
    {"message": f"Model metrics: {metrics}"},
    "http://localhost:8000",
)
print(response.get("result"))
```


## Bug Reports & Feature Requests

Please open an issue on GitHub if you encounter a bug or have a feature request.

## License

This project is released under the MIT License. See `LICENSE.txt` for details.

