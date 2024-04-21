from django.db import models


# Create your models here.
# py manage.py makemigrations
# py manage.py migrate
class UserInfo(models.Model):
    name = models.CharField(max_length=32, verbose_name='name')
    password = models.CharField(max_length=64, verbose_name='password')


class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/', verbose_name='上传的文件')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name
