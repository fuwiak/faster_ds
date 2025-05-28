import pandas as pd

from faster_ds.LLM import send_to_llm


class TimeSeries:
    """Base interface for time series operations."""

    def __init__(
        self,
        series: pd.Series | None = None,
        send_to_llm_flag: bool = False,
        server_url: str | None = None,
    ) -> None:
        """Initialize the time series helper."""

        self.series = pd.Series(series) if series is not None else pd.Series(dtype=float)
        self.server_url = server_url
        if send_to_llm_flag:
            self.send_summary_to_llm()

    def send_summary_to_llm(self) -> None:
        """Send a statistical summary of the series to an attached LLM."""
        summary = self.series.describe().to_dict()
        send_to_llm(f"Time series summary: {summary}", server_url=self.server_url)
