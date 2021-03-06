# Generated by Django 3.0.6 on 2020-05-19 10:05

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0009_auto_20200519_1452'),
    ]

    operations = [
        migrations.AlterField(
            model_name='borrowed',
            name='dueDate',
            field=models.DateTimeField(default=datetime.datetime(2020, 5, 26, 15, 5, 20, 411298), null=True),
        ),
        migrations.AlterField(
            model_name='borrowed',
            name='status',
            field=models.CharField(choices=[('Pending', 'Pending'), ('Borrowed', 'Borrowed'), ('Due', 'Due')], max_length=200, null=True),
        ),
        migrations.CreateModel(
            name='UserRating',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating', models.CharField(choices=[('2', '2'), ('1', '1'), ('3', '3'), ('4', '4'), ('5', '5')], max_length=200, null=True)),
                ('book', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='accounts.Book')),
                ('user', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='accounts.User')),
            ],
        ),
    ]
