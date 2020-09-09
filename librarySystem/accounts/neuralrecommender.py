import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


warnings.filterwarnings('ignore')

dataset = pd.read_csv('C:\\Users\\NUser1\\Desktop\\LibrarySystem Book Recommender\\datasets\\ratings.csv')
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
n_users = len(dataset.userID.unique())
n_books = len(dataset.bookID.unique())

# creating book embedding path
book_input = Input(shape=[1], name="Book-Input")
book_embedding = Embedding(n_books+1, 5, name="Book-Embedding")(book_input)
book_vec = Flatten(name="Flatten-Books")(book_embedding)

# creating user embedding path
user_input = Input(shape=[1], name="User-Input")
user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
user_vec = Flatten(name="Flatten-Users")(user_embedding)

# concatenate features
conc = Concatenate()([book_vec, user_vec])

# add fully-connected-layers
fc1 = Dense(128, activation='relu')(conc)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)

# Create model and compile it
model2 = Model([user_input, book_input], out)
model2.compile('adam', 'mean_squared_error')


if os.path.exists('regression_model2.h5'):
    model2 = load_model('regression_model2.h5')
else:
    history = model2.fit([train.userID, train.bookID], train.rating, epochs=5, verbose=1)
    model2.save('regression_model2.h5')
    plt.plot(history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Training Error")
    plt.savefig('figure.png')

# evaluating model
model2.evaluate([test.userID, test.bookID], test.rating)

# function to create plot
def scatterplot1():
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(book_em_weights)
    ax = sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
    fig = ax.get_figure()
    fig.savefig('scatterplot1.png')

# function to create plot
def scatterplot2():
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(book_em_weights)
    ax2 = sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
    fig2 = ax2.get_figure()
    fig2.savefig('scatterplot2.png')

# function to create plot with TSNE
def scatterplot3():
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tnse_results = tsne.fit_transform(book_em_weights)

    ax3 = sns.scatterplot(x=tnse_results[:,0], y=tnse_results[:,1])
    fig3 = ax3.get_figure()
    fig3.savefig('scatterplot3.png')


# Extract embeddings
book_em = model2.get_layer('Book-Embedding')
book_em_weights = book_em.get_weights()[0]
print("weights: ",book_em_weights[:5])

book_em_weights = book_em_weights / np.linalg.norm(book_em_weights, axis = 1).reshape((-1, 1))
book_em_weights[0][:10]
print(np.sum(np.square(book_em_weights[0])))


# Creating dataset for making recommendations for the first user
book_data = np.array(list(set(dataset.bookID)))
print("\n\nbook array length: ",len(book_data))
n = np.array(list(set(dataset.userID)))
print("user array length: ",len(n))

print("\nSelected user: ",n[2])

user = np.array([n[2] for i in range(len(book_data))])



predictions = model2.predict([user, book_data])
predictions = np.array([a[0] for a in predictions])
print("\npredictions: ",predictions)
print("\nprediction length: ",len(predictions))
#sorts the array which contains predicted score and give indice value which is the book id
recommended_book_ids = (-predictions).argsort()[:5]

print("\nRecommended book id: ",recommended_book_ids)
#predicted score
print("\nPredicted score: ",predictions[recommended_book_ids])

books = pd.read_csv('C:\\Users\\NUser1\\Desktop\\LibrarySystem Book Recommender\\datasets\\books.csv')
print("\nBook details:\n ",books[books['id'].isin(recommended_book_ids)])
