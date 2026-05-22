import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

data = {
    'review': ['I love this product', 'This is amazing', 'Great service', 'I am so happy', 
               'Excellent work', 'This is very good', 'I hate this', 'This is terrible', 
               'Worst experience ever', 'I am very disappointed', 'Poor quality', 'Bad service'], 
    'sentiment': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
}
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer().fit(df['review'])
model = LogisticRegression().fit(vectorizer.transform(df['review']), df['sentiment'])

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vec.pkl', 'wb'))
print('Model updated successfully!')