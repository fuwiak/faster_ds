from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class Pipelines:
	@staticmethod
	def create_pipeline(**parameters: dict) -> Pipeline:
		"""
		:param parameters: parameters for pipeline
		:return: pipeline

		"""
		raise NotImplementedError


class WordCloud:
	@staticmethod
	def create_word_cloud(text: str) -> None:
		"""
		:param text: text to create word cloud
		:return: word cloud

		"""
		raise NotImplementedError
	
	
class CleanText:
	@staticmethod
	def clean_text(text: str) -> str:
		"""
		:param text: text to clean
		:return: cleaned text

		"""
		raise NotImplementedError

