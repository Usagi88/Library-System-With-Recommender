import django_filters
from django_filters import DateFilter, CharFilter

from .models import *

class BorrowFilter(django_filters.FilterSet):
	start_date = DateFilter(field_name="borrowedDate", lookup_expr='gte')
	end_date = DateFilter(field_name="borrowedDate", lookup_expr='lte')
	class Meta:
		model = Borrowed
		fields = '__all__'
		exclude = ['user','dueDate','borrowedDate']

