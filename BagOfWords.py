from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

count = CountVectorizer()

reviews=[]
for review in df[0]:
    reviews.append(review)
    
docs = np.array(reviews)
bag = count.fit_transform(docs)