"""Utilities for generating fake data for examples and tests."""

from __future__ import annotations

import datetime
from faker import Faker
import numpy as np
import pandas as pd

from faster_llm.doc import doc


class FakeData:
    """Helper class used to create various fake datasets."""

    @staticmethod
    def one_sentence() -> str:
        """Return a single random sentence."""
        temp = Faker()
        return temp.text()

    @staticmethod
    def many_sentences(how_many: int) -> tuple[list[str], pd.DataFrame]:
        """Return a list and DataFrame with fake sentences."""
        temp = Faker()
        data = [temp.text() for _ in range(how_many)]
        df = pd.DataFrame(data, columns=["text"])
        return data, df

    @staticmethod
    def classification_data(how_many: int) -> pd.DataFrame:
        """Return a DataFrame with example classification data."""
        data: list[list[object]] = []
        for _ in range(how_many):
            temp = Faker("en_US")
            row = [
                temp.prefix(),
                temp.name(),
                temp.date(pattern="%d-%m-%Y", end_datetime=datetime.date(2020, 1, 1)),
                temp.phone_number(),
                temp.email(),
                temp.address(),
                temp.zipcode(),
                temp.city(),
                temp.state(),
                temp.country(),
                temp.year(),
                temp.time(),
                temp.url(),
                np.random.randint(0, 2, 1)[0],
            ]
            data.append(row)

        headers = [
            "Prefix",
            "Name",
            "Birth Date",
            "Phone Number",
            "Additional Email Id",
            "Address",
            "Zip Code",
            "City",
            "State",
            "Country",
            "Year",
            "Time",
            "Link",
            "HaveAjob",
        ]
        df = pd.DataFrame(data, columns=headers)
        return df

    @staticmethod
    @doc(classification_data)
    def regression_data(how_many: int) -> pd.DataFrame:
        """Return a DataFrame with example regression data."""
        data: list[list[object]] = []
        for _ in range(how_many):
            temp = Faker("en_US")
            row = [
                temp.prefix(),
                temp.name(),
                temp.date(pattern="%d-%m-%Y", end_datetime=datetime.date(2020, 1, 1)),
                temp.phone_number(),
                temp.email(),
                temp.address(),
                temp.zipcode(),
                temp.city(),
                temp.state(),
                temp.country(),
                temp.year(),
                temp.time(),
                temp.url(),
                np.random.randint(1000, 20000, 1)[0],
            ]
            data.append(row)

        headers = [
            "Prefix",
            "Name",
            "Birth Date",
            "Phone Number",
            "Additional Email Id",
            "Address",
            "Zip Code",
            "City",
            "State",
            "Country",
            "Year",
            "Time",
            "Link",
            "Salary",
        ]
        df = pd.DataFrame(data, columns=headers)
        return df
