import yaml
import glob
import os
from ...interfaces.retrieve_config import IRetrieveConfig
from typing import List

class VocabStocksYAMLConfigProvider(IRetrieveConfig):

    def _read_config(self, config_src: str) -> List:
        if os.path.isdir(config_src):
            stock_list = []
            for filepath in glob.glob(os.path.join(config_src, "*.yaml")):
                with open(filepath, "r") as f:
                    data = yaml.safe_load(f)
                stock_list.extend(data.get("tickers", []))
            return list(set(stock_list))
        if not os.path.isfile(config_src):
            return []
        with open(config_src, "r") as f:
            data = yaml.safe_load(f)
        return data.get("tickers", [])

    def get_config(self, config_src: str) -> List:
        return self._read_config(config_src)
