import time
from uuid import uuid4

from django.http import HttpResponse
from django.shortcuts import render
# from django.http import HttpResponse
import json
import logging
from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from random import randrange

from ML_django import settings
from draw_app.models import ImageStore
from rest_framework import viewsets, filters
from draw_app.serializers import ImageStoreSerializer
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import generics
from rest_framework.parsers import MultiPartParser, FormParser
import os
from PIL import Image
import numpy
from rest_framework import status

#---- generate random number
from random import randrange
# Create your views here.
class ImageStoreListView(generics.ListCreateAPIView):
    queryset = ImageStore.objects.all()
    serializer_class = ImageStoreSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['=image_id']


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_image(request):
    image = request.FILES.get('my-file')

    new_image_data = ImageStore.objects.create(
        image_id=int(image.name.split('.')[0]),
        image_link=image
    )

    serializer = ImageStoreSerializer(new_image_data)

    """
    do anything with the image here and give answer value to variable ans below. 
    ans will be returned as the to frontend and display.
    
    The time.sleep(10) function is used to set a 10 second delay for testing purpose. 
    Right now, the frontend will display the result 10 sec later. Comment it out if you want!
    np_im is the 256*256 nparray
    """
    # print(type(image))
    img = Image.open(image.file).convert('L')
    np_im = numpy.array(img)
    print(np_im)



    ans = 'CAT'  #return any value by passing value to function and get a answer from CNN model
    time.sleep(4)
    return Response({
                        'data': serializer.data,
                        # 'errors': serializer.errors, # `.is_valid()` must be called
                        'message': [{} , {}],
                        'result': ans
                    })



class FrontendAppView(View):
    """
    Serves the compiled frontend entry point (only works if you have run `yarn
    run build`).
    """
    def get(self, request):
        print(os.path.join(settings.REACT_APP_DIR, 'build', 'index.html'))
        try:
            with open(os.path.join(settings.REACT_APP_DIR, 'build', 'index.html')) as f:
                return HttpResponse(f.read())
        except FileNotFoundError:
            logging.exception('Production build of app not found')
            return HttpResponse(
                """
                This URL is only used when you have built the production
                version of the app. Visit http://localhost:3000/ instead, or
                run `yarn run build` to test the production version.
                """,
                status=501,
            )