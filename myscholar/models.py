from django.db import models

# Create your models here.

class MyModel(models.Model):
    title = models.CharField(max_length=255, null=True, blank=True)
    author = models.CharField(max_length=255, null=True, blank=True)
    year = models.CharField(max_length=255, null=True, blank=True)
    publication_url = models.URLField(null=True, blank=True)
    profile_url = models.URLField(null=True, blank=True)
    
    def __str__(self):
        return self.title
    