import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 1. የትሬኒንግ ዳታሴት (Dataset) ከ 3 ክፍሎች ጋር
# 1 = Positive, 0 = Negative, 2 = Recommendation
data = {
    'review': [
        # --- Positive (1) ---
        'I love this product', 
        'This is amazing', 
        'Great service', 
        'I am so happy', 
        'Excellent work', 
        'This is very good', 
        
        # --- Negative (0) ---
        'I hate this', 
        'This is terrible', 
        'Worst experience ever', 
        'I am very disappointed', 
        'Poor quality', 
        'Bad service',
        
        # --- Recommendation (2) ---
        'The university should add more computers to the IS lab',
        'We need to improve the textbook collection in the library',
        'Please provide more water extensions near the dorms',
        'I suggest expanding the cafeteria seating capacity',
        'They should schedule more practical training sessions'
    ], 
    'sentiment': [1, 1, 1, 1, 1, 1,  0, 0, 0, 0, 0, 0,  2, 2, 2, 2, 2]
}

df = pd.DataFrame(data)

print("--- AI Training Started ---")

# 2. ጽሑፎቹን ወደ ቁጥር መቀየር (Vectorization)
vectorizer = TfidfVectorizer().fit(df['review'])
X_train = vectorizer.transform(df['review'])
y_train = df['sentiment']

# 3. የ AI ሞዴሉን ማሰልጠን (Multi-class Logistic Regression)
model = LogisticRegression().fit(X_train, y_train)

# 4. አዲሶቹን model.pkl እና vec.pkl ፋይሎች ማመንጨት
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vec.pkl', 'wb'))

print('Model updated successfully with 3 classes (Positive, Negative, Recommendation)!')