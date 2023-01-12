from django.db import models

# Create your models here.
class CheckedBugs(models.Model):
    specie = models.CharField(max_length=15)
    photo = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.specie