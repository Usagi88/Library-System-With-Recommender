from django.db import models
from datetime import datetime, timedelta

# Create your models here.
class User(models.Model):
	name = models.CharField(max_length=200, null=True)
	year = models.IntegerField(null=True)
	gender = models.CharField(max_length=10, null=True)
	email = models.CharField(max_length=200, null=True)
	date_created = models.DateTimeField(auto_now_add=True, null=True)

	def __str__(self):
		return self.name


class Book(models.Model):

	bookID = models.IntegerField(null=True)
	image = models.ImageField(blank=True,null=True)
	title = models.CharField(max_length=200, null=True)
	bookCount = models.IntegerField(null=True)
	ratingCount = models.IntegerField(null=True)
	ratingAvg = models.DecimalField(max_digits=5, decimal_places=2,null=True)
	author = models.CharField(max_length=200, null=True)
	bigImage = models.ImageField(blank=True,null=True)
	language = models.CharField(max_length=200, null=True)

	def __str__(self):
		return self.title
	
	
class Borrowed(models.Model):
	STATUS = {
				('Pending','Pending'),
				('Borrowed','Borrowed'),
				('Due','Due'),
			}
	user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
	book = models.ForeignKey(Book, null=True, on_delete=models.SET_NULL)		
	borrowedDate = models.DateTimeField(auto_now_add=True, null=True)
	dueDate = models.DateTimeField(default=datetime.now()+timedelta(days=7), null=True)
	status = models.CharField(max_length=200, null=True, choices=STATUS)

	def __str__(self):
		return self.book.title






