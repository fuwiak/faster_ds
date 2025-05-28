import pandas as pd
from sklearn.linear_model import LogisticRegression

from faster_llm.ML.classification import Model


def main() -> None:
    """Train a simple classifier and send metrics to the LLM."""
    df = pd.read_csv("../sample_data/iris.csv")
    X = df.drop(columns=["variety"])
    y = (df["variety"] == "Virginica").astype(int)

    model = Model(
        LogisticRegression(max_iter=200),
        X,
        y,
        send_to_llm_flag=True,
    )
    print(model._compute_metrics())


if __name__ == "__main__":
    main()
