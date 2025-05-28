"""Simple dimension reduction utilities with optional MCP notifications."""

import pandas as pd
import matplotlib.pylab as plt

from .utils import mcp_notify

class Model:
    """Collection of dimension reduction helpers."""

    @staticmethod
    @mcp_notify
    def pca(
        df: pd.DataFrame,
        n_components: int = 2,
        *,
        notify: bool = False,
        server_url: str | None = None,
    ) -> pd.DataFrame:
        """Run Principal Component Analysis on ``df``."""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(df)
        cols = [f"PC{i+1}" for i in range(comps.shape[1])]
        return pd.DataFrame(comps, columns=cols, index=df.index)



		
