import csv, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import random
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib import messages
from django.contrib.auth.decorators import permission_required
from .models import *
from django.utils.translation import gettext as _
from .forms import BorrowForm
from .filters import BorrowFilter
from django.core.paginator import Paginator


# Create your views here.

def home(request):
	borrows = Borrowed.objects.all()
	users = User.objects.all()
	books = Book.objects.all()

	total_books = books.count()

	total_due = borrows.filter(status='Due').count()

	total_borrows = borrows.count()

	paginator = Paginator(users, 3)
	page = request.GET.get('page')
	users = paginator.get_page(page)

	context = {'borrows':borrows, 'users':users,
	 'total_books':total_books, 'total_due':total_due,
	  'total_borrows':total_borrows, 'paginator':paginator}

	return render(request, 'accounts/dashboard.html', context)

def books(request):
	books = Book.objects.all()
	paginator = Paginator(books, 4)
	page = request.GET.get('page')
	books = paginator.get_page(page)

	return render(request, 'accounts/books.html',{'books':books,'paginator':paginator})

def user(request, pk_test):
	user = User.objects.get(id=pk_test)
	
	borrows = user.borrowed_set.all()
	borrows_count = borrows.count()

	myFilter = BorrowFilter(request.GET, queryset=borrows)
	borrows = myFilter.qs

	context = {'user':user,'borrows_count':borrows_count,'borrows':borrows,'myFilter':myFilter}	
	return render(request, 'accounts/user.html',context)


def createBorrow(request,pk):
	BorrowFormSet = inlineformset_factory(User, Borrowed, fields=('book', 'status'),)
	user = User.objects.get(id=pk)
	formset = BorrowFormSet(queryset=Borrowed.objects.none(),instance=user)
	#form = OrderForm(initial={'customer':customer})
	if request.method == 'POST':
		#print('Printing POST:', request.POST)
		#form = OrderForm(request.POST)
		formset = BorrowFormSet(request.POST, instance=user)
		if formset.is_valid():
			formset.save()
			return redirect('/')

	context = {'form':formset}
	return render(request, 'accounts/borrow_form.html', context)


def updateBorrow(request, pk):
	borrow = Borrowed.objects.get(id=pk)
	form = BorrowForm(instance=borrow)
	
	if request.method == 'POST':
		#print('Printing Post: ',request.POST)
		form = BorrowForm(request.POST, instance=borrow)
		if form.is_valid():
			form.save()
			return redirect('/')

	context = {'form':form}
	return render(request, 'accounts/borrow_form.html', context)


def deleteBorrow(request, pk):
	borrow = Borrowed.objects.get(id=pk)
	if request.method == "POST":
		borrow.delete()
		return redirect('/')
	context = {'item':borrow}
	return render(request, 'accounts/delete.html', context)

class recommend:
    def __init__(self, ids, title, author, language):
        self.ids = ids
        self.title = title
        self.author = author
        self.language = language

#recommender
def recommender(request):
	dataset = pd.read_csv('C:\\Users\\NUser1\\Desktop\\LibrarySystem Book Recommender\\datasets\\ratings.csv')
	train, test = train_test_split(dataset, test_size=0.2, random_state=42)
	n_users = len(dataset.userID.unique())
	n_books = len(dataset.bookID.unique())
	#loading model
	if os.path.exists('regression_model2.h5'):
	    model2 = load_model('regression_model2.h5')
	else:
	    history = model2.fit([train.userID, train.bookID], train.rating, epochs=5, verbose=1)
	    model2.save('regression_model2.h5')
	    plt.plot(history.history['loss'])
	    plt.xlabel("Epochs")
	    plt.ylabel("Training Error")
	    plt.savefig('figure.png')


	model2.evaluate([test.userID, test.bookID], test.rating)

	book_data = np.array(list(set(dataset.bookID)))
	n = np.array(list(set(dataset.userID)))
	user = np.array([n[random.randint(0,1000)] for i in range(len(book_data))])



	predictions = model2.predict([user, book_data])
	predictions = np.array([a[0] for a in predictions])

	recommended_book_ids = (-predictions).argsort()[:5]

	#predicted score

	books = pd.read_csv('C:\\Users\\NUser1\\Desktop\\LibrarySystem Book Recommender\\datasets\\books.csv')
	#values = {}
	#values = books[books['id'].isin(recommended_book_ids)]
	list_of_recommend = []
	user_input = recommended_book_ids
	counter = 0
	counter2  = 0

	for ids in user_input:
	    for index, row in books.iterrows():
	        if row["id"] == user_input[counter]:
	            list_of_recommend.append(recommend(row["id"],row["title"],row["author"],row["language"]))
	    counter = counter + 1
	#initializing values to render in html    
	booklength = len(book_data) 
	predictionMax = predictions
	predictionMax[::-1].sort()
	predictionMaxLength = predictionMax[:5]
	countScore = len(predictionMaxLength)   
	context = {'list_of_recommend':list_of_recommend, 'booklength': booklength, 'countScore':countScore, 'predictionMaxLength':predictionMaxLength}

	return render(request, 'accounts/recommend.html', context)



#uploading books using csv
@permission_required('admin.can_add_log_entry')
def uploading(request):
	template = "accounts/uploadData.html"
	prompt = {
		'order': 'Order of CSV should be bookID(int), image(string), title(string), bookCount(int), ratingCount(int), ratingAvg(float), author(string),  bigImage(string), language(string)'
	}

	if request.method == "GET":
		return render(request, template, prompt)

	csv_file = request.FILES['file']

	if not csv_file.name.endswith('.csv'):
		message.error(request, 'This is not a csv file')

	data_set = csv_file.read().decode('UTF-8')
	io_string = io.StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=',',quotechar="|"):
		_. created = Book.objects.update_or_create(
			bookID = column[0],
			image = column[1],
			title = column[2],
			bookCount = column[3],
			ratingCount = column[4],
			ratingAvg = column[5],
			author = column[6],
			bigImage = column[7],
			language = column[8],
			)
			
			
	context = {}
	return render(request, template, context)	

#uploading users using csv
@permission_required('admin.can_add_log_entry')
def uploadingUser(request):
	template = "accounts/uploadData.html"
	prompt = {
		'order': 'Order of CSV should be name(string), year(int), gender(string), email(string)'
	}

	if request.method == "GET":
		return render(request, template, prompt)

	csv_file = request.FILES['file']

	if not csv_file.name.endswith('.csv'):
		message.error(request, 'This is not a csv file')

	data_set = csv_file.read().decode('UTF-8')
	io_string = io.StringIO(data_set)
	next(io_string)
	for column in csv.reader(io_string, delimiter=',',quotechar="|"):
		_. created = User.objects.update_or_create(
			name = column[0],
			year = column[1],
			gender = column[2],
			email = column[3]
			)			
			
	context = {}
	return render(request, template, context)	

