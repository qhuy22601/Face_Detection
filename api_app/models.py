from django.db import models

# Create your models here.
class UserModel(models.Model):
    id = models.TextField(primary_key=True, max_length=255)
    email = models.EmailField()
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    hire_date = models.DateField(null=True)
    gender = models.CharField(max_length=255)
    phone_number = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    image = models.CharField(max_length=255)
    balance = models.FloatField(null=True)
    created_at = models.DateField()

    class Meta:
        db_table = 'users'