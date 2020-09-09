from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="dashboard"),
    path('books/', views.books, name="books"),
    path('user/<str:pk_test>/', views.user, name="user"),
    path('uploading/', views.uploading),
    path('uploadingUser/', views.uploadingUser),

    path('create_borrow/<str:pk>/', views.createBorrow, name="create_borrow"),
    path('update_borrow/<str:pk>/', views.updateBorrow, name="update_borrow"),
    path('delete_borrow/<str:pk>/', views.deleteBorrow, name="delete_borrow"),

    path('recommend/', views.recommender, name="recommend"),
    
]	