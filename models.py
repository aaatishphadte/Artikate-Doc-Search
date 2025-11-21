# knowledge_base/models.py

from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to="knowledge_files/")
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    content = models.TextField()
    page_number = models.IntegerField(null=True, blank=True)
    embedding = models.BinaryField()  # Store embedding bytes
