# Generated by Django 3.0.6 on 2020-05-18 17:36

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Book',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bookID', models.IntegerField(null=True)),
                ('image', models.ImageField(blank=True, null=True, upload_to='')),
                ('title', models.CharField(max_length=200, null=True)),
                ('bookCount', models.IntegerField(null=True)),
                ('ratingCount', models.IntegerField(null=True)),
                ('ratingAvg', models.DecimalField(decimal_places=2, max_digits=5, null=True)),
                ('author', models.CharField(max_length=200, null=True)),
                ('language', models.CharField(max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Borrowed',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('bookID', models.CharField(max_length=200, null=True)),
                ('borrowedDate', models.DateTimeField(auto_now_add=True, null=True)),
                ('dueDate', models.DateTimeField(default=datetime.datetime(2020, 5, 25, 22, 36, 29, 546010), null=True)),
                ('status', models.CharField(choices=[('Due', 'Due'), ('Pending', 'Pending'), ('Borrowed', 'Borrowed')], max_length=200, null=True)),
            ],
        ),
    ]
