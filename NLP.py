class clean_text:
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
