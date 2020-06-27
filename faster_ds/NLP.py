from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class Pipelines:
	@abstractmethod
	def create_pipeline(**parameters):
		"""
		Sample usage
		text_clf = Pipeline([('vect', CountVectorizer()),
			    ('tfidf', TfidfTransformer()),
			    ('clf', MultinomialNB())])
		text_clf = text_clf.fit(X_train, y_train)
		text_clf.predict(X_test)
		"""
		raise NotImplementedError
	

class WordCloud:
	
	def create_stopwords(file_name, sep=" "):
		data = open(file_name, "r")
		data = data.read()
		data = data.split(sep)
		return " ".join(data)
		
	
	def create_word_cloud(text, stopwords, name_of_graph="cloud.png"):
		from wordcloud import WordCloud, STOPWORDS
		stopwords = set(STOPWORDS)

		wordcloud = WordCloud(width = 800, height = 800, 
				stopwords = stopwords, 
				min_font_size = 8,background_color='white').generate(text)
		return wordcloud
	
	def plot_word_cloud(wordcloud):
		import matplotlib.pylab as plt
		plt.figure(figsize = (16, 16)) 
		plt.imshow(wordcloud)
		plt.savefig(name_of_graph)
		plt.show()
	
	
	
	
	





















class CleanText:
    import string
    import re
     
    @staticmethod
    def split_text(self, t):
        return t.apply(lambda x: str(x).split(" "))
    @staticmethod
    def to_lower(self, t):
        return t.apply(lambda x:   str(x).lower())
    
    @staticmethod
    def remove_mentions(self,t):
        return t.apply(lambda x:  re.sub(r'@\w+', '', str(x)))
    @staticmethod
    def remove_numbers(self, t):
        return t.apply(lambda x: re.sub(r'\d+', '', str(x)))
    
    @staticmethod
    def remove_urls(self, t):
        return t.apply(lambda x: re.sub(r'http.?://[^\s]+[\s]?', '', str(x)))
    @staticmethod
    def remove_punctuation(self,t):
         return t.apply(lambda x: str(x).translate(str.maketrans('','',string.punctuation)))
        
    @staticmethod
    def remove_stopwords(self, t):
        return t.apply(lambda x: [word for word in str(x).split(" ") if word not in stopwords.words('english')])
    
    @staticmethod
    def stemming(self,t):
        temp= t.apply(lambda x: str(x).split(" "))
        porter = PorterStemmer()
        stemmed = temp.apply(lambda x: porter.stem(" ".join(x)))
        return stemmed
    @staticmethod
    def vectorize_text(text):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer()
	vectorizer.fit(text)

	print(vectorizer.vocabulary_)

	# Transform text to vector
	vector = vectorizer.transform(text)
	print(vector.shape)
	print(type(vector))
	print(vector.toarray())
    
    def ready_data(self, t):
        t1 = self.remove_mentions(t)
        t2 = self.remove_urls(t1)
        t3 = self.remove_punctuation(t2)
        t4 = self.to_lower(t3)
        t5 = self.remove_numbers(t4)
        t6 = self.stemming(t5)
        
        return t6
