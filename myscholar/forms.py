from django.core.validators import FileExtensionValidator
from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(required=True, validators=[FileExtensionValidator(allowed_extensions=['xlsx'])])
