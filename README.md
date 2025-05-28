# faster_ds

Open Source Python based module based on Numpy, Pandas and scikit-learn API.
The module follows the design principles of the Scikit-Learn library.


Created to make faster boring stuff in preprocessing data and data analysis.


# Notes
# Getting Started
 - List of features: docs/*
 - Overview features: docs/overview*
 - Contributing: docs/contributing*


# Bug Reports
Send your bug reports and feature requests to: email

# License


## LLM Integration

``faster_ds`` can serve as a bridge between your favourite ML stack and Large
Language Model frameworks. The API is designed to work with tools like
PyTorch, Keras and scikit-learn while easily connecting with libraries such as
[LangChain](https://github.com/langchain-ai/langchain) or
[LlamaIndex](https://github.com/jerryjliu/llama_index).

Model metrics, plots and other artifacts produced by the ML pipeline can be
passed to a language model (for example ``mistral-7B``) using a small connector
module provided in ``faster_ds``. This enables the LLM to interpret results or
generate reports from the training process with minimal setup.



