from draw_app.models import ImageStore
from rest_framework import serializers


class ImageStoreSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageStore
        fields = ['image_id', 'image_link']