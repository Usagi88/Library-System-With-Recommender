import django_filters
from django.forms import ModelForm
from django_filters import DateFilter, CharFilter
from .models import Borrowed


class BorrowForm(ModelForm):
	class Meta:
		model = Borrowed
		fields = '__all__'



