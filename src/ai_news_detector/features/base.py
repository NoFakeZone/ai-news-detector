from abc import ABC, abstractmethod


class TextFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
