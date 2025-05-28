import pandas as pd
import sklearn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from faster_llm.LLM import send_to_llm


class Model:
    """Simple regression model wrapper with optional LLM reporting."""

    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        send_to_llm_flag: bool = False,
        server_url: str | None = None,
    ) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size
        )
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.metrics = self._compute_metrics()
        self.server_url = server_url
        if send_to_llm_flag:
            self.send_metrics_to_llm()

    def _compute_metrics(self) -> dict:
        """Return basic regression metrics as a dictionary."""
        return {
            "r2": r2_score(self.y_test, self.y_pred),
            "mae": mean_absolute_error(self.y_test, self.y_pred),
            "mse": mean_squared_error(self.y_test, self.y_pred),
        }

    def send_metrics_to_llm(self) -> None:
        """Send computed metrics to an attached LLM service."""
        send_to_llm(
            f"Regression metrics: {self.metrics}",
            server_url=self.server_url,
        )
    
