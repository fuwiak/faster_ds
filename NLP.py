"""Count Vectoring"""
from sklearn.feature_extraction.text import CountVectorizer


class NLP:
	@staticmethod
	def vectorize_text(text):
		vectorizer = CountVectorizer()
		vectorizer.fit(text)

		print(vectorizer.vocabulary_)

		# Transform text to vector
		vector = vectorizer.transform(text)
		print(vector.shape)
		print(type(vector))
		print(vector.toarray())


	
'''
if __name__ == "__main__":
	text = "Sample text to check correctness of program "
	vectorize_text(text)
'''
