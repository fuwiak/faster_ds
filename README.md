# faster_llm

`faster_llm` is a Python library that speeds up common data preprocessing and analysis tasks. It builds on familiar tools like NumPy, Pandas and scikit-learn while adding connectors for Large Language Model (LLM) frameworks such as LangChain and LlamaIndex.

## Features

- Utilities for splitting, encoding and cleaning tabular data
- Ready‑to‑use pipelines for classical machine learning workflows
- Helpers for passing results to LLM frameworks

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

## Bug Reports & Feature Requests

Please open an issue on GitHub if you encounter a bug or have a feature request.

## License

This project is released under the MIT License. See `LICENSE.txt` for details.

