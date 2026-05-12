from abc import ABC, abstractmethod
from typing import Any
class IRetrieveConfig(ABC):

    @abstractmethod
    def get_config(self, config_src: str) -> Any:
        """"Get Config"""
        pass
