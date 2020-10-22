# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
name_img='ali'
image = cv2.imread('./'+name_img+'.png')

# Read the desired model
path = "ESPCN_x2.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 2)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("./"+name_img+"_upscaled.png", result)
