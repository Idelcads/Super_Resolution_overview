# IMKI_Technical_test
[***Project-1***](https://github.com/Idelcads/IMKI_Technical_test#Project-1) \
[***Project-2***](https://github.com/Idelcads/IMKI_Technical_test#Project-2) 

---

I'm using `PyTorch-cpu 1.1.0`, `Scipy 1.5.2`, `Lasagne 0.2.dev1` and `Tensorflow 2.1.0` in `Python 3.7`.

---
# Project-1
The Aim of this project is to make a survey of the principal methods used for increase the resolution of images using deep-learning (DL) algorithms and test few programs available.

As a Head start in understanding GAN algorithms, some links are suggested in the last section.

[***Survey***](https://github.com/Idelcads/IMKI_Technical_test#Survey) \
[***Existing_codes***](https://github.com/Idelcads/IMKI_Technical_test#Existing_codes) \
[***Remarks***](https://github.com/Idelcads/IMKI_Technical_test#Remarks) \
[***Problems***](https://github.com/Idelcads/IMKI_Technical_test#Problems) 


## Survey
En premier lieu il est important de préciser qu’il semble ressortir 2 méthodes principales permettant d’augmenter la qualité d’une image via des méthodes de Deep Learning (DL) : 

* **1st Method (CNN & GAN):** tTis method is based on the use on DL methods including CNN and GAN. This can also be found in literature, one publication [1] “Survey_SR.pdf” summarizes the different steps allowing the use of DL for Super-Resolution (SR). Another publication [2] “GAN_for_SR.pdf” is dedicated to the use of GANs in SR applications. The codes tested and provided during this project are based the methods detailed in these two publications and particularly the architecture of GANs.

* **2nd Method (without GAN):** another method, that seems to be possible without the use of GAN, is given in details in the following article [3] (https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5). The author actually justifies the use of this method by stating that «  One of the limitations of GANs is that they are effectively a lazy approach as their loss function, the critic, is trained as part of the process and not specifically engineered for this purpose. This could be one of the reasons many models are only good at super resolution and not image repair ». In this article, the author goes into details about the different implemented steps:
Une autre méthode semble être possible sans l’utilisation d’un GAN et est détaillée dans cet article [3] (https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5). L’auteur justifie l’utilisation de la méthode de la manière suivante :   «  One of the limitations of GANs is that they are effectively a lazy approach as their loss function, the critic, is trained as part of the process and not specifically engineered for this purpose. This could be one of the reasons many models are only good at super resolution and not image repair ». Dans cet article, l’auteur détaille les différentes étapes mises en œuvre dont voicis la liste :
  * A U-Net architecture with cross connections similar to a DenseNet
  * A ResNet-34 based encoder and a decoder based on ResNet-34
  * Pixel Shuffle upscaling with ICNR initialisation
  * Transfer learning from pretrained ImageNet models
  * A loss function based on activations from a VGG-16 model, pixel loss and gram matrix loss
  * Discriminative learning rates
  * Progressive resizing 
 
Though I did not dive into the details of the second method, seeing that i could not find a “ready to use” code. Also considering the given time frame it does not seem like the appropriate solution for the short term. 


## Existing_codes

3 different codes were tested. All three of them based on the GAN method. They allow the use of pre-trained models. These models are generated using databases (via image banks) containing high quality images (HQ) that are voluntarily degraded then used as entries for the generator model. The advantage of testing several codes is to be able to test and generate models from different frameworks.

* **Code n°1:** \
Code developed by Sagar Vinodabadu available on Github: (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution#some-more-examples) \
*Allows the reading of pre-trained models. ** .pth** (models generated via  the **PyTorch** framework) and the generating of new models. It also permits the comparison of three methods, GAN, RestNet and Bicubic interpolation.*\
The checkpoint_srgan.pth model was trained using the MS COCO database from 120K images (people, dog, landscape ...) 

* **Code n°2:** \
Code developed by Alex J. Champandard available on Github: (https://github.com/alexjc/neural-enhance#1-examples--usage) \
*Allows the reading of pre-trained models. ** .pth.bz2** (models generated via  the **Lasagne** framework) and the generating of new models*\
We are not aware of the Data bank of images used to generate the provided pre-trained models.

* **Code n°3:** \
*Allows the reading of pre-trained models. ** .pb** (models generated via  the **tensorflow** framework)*\
It is possible to train a new ESPCN model using the following works: https://github.com/fannymonori/TF-ESPCN. Though I did not go into the details of this possibility. 

## Remarks

1. Not owning a powerful enough PC, it was impossible for me to traim my own models using the provided codes.

2. All calculation were made using the CPU.

3. In order to not overload the memory (RAM) of my PC, only low resolution images were used for the tests (ali.jpg and lowres.jpg).

## Results

Le test a été réalisé avec une image de résolution médiocre (ali.png). Les résultats obtenus sont visibles dans la section résultats de chaque code. 

Au vu des précisions données pour chaque code, celui donnant le meilleur résultat pour notre étude (SR sur une image de Mohammed Ali) est le n°1 qui permet en plus de comparer facilement les méthodes. Cela s’explique très probablement par le fait que le modèle a été appris avec des images ayant une certaine cohérence avec notre sujet. Cependant pour générer un modèle vraiment spécifique à nos attentes (améliorer la qualité d’une image de Mohammed Ali) il faudrait que notre modèle puisse apprendre en se servant de base de données d’image de portrait de célébrités disponible sur DockerHub par exemple. On peut observer le résultat suivant :  

![alt text](https://github.com/Idelcads/IMKI_Technical_test/blob/main/Images_readme/result_code1.bmp)

Le code n°2 permet d’obtenir des résultats qui ne sont pour le moment pas exploitable au vu des images obtenus. Cependant on peut reconnaitre que la fonction d’agrandissement est bien respectée. Le problème vient probablement d’une mauvaise lecture de l’image après traitement ou d’une mauvaise conversion. 

![alt text](https://github.com/Idelcads/IMKI_Technical_test/blob/main/Images_readme/result_code2.png)

Le code n°3 est très simple est permet simplement de charger un modèle pour le tester. Au vu du modèle utilisé pour le test le résultat reste cependant correct. 

**Concerning the learning of a new model or the possibily to develop our own application the code n°1 seems to be the best option. Furthermore if we need to modify the architecture of the generetor or the discriminator we can easily start from the file `models.py`**

---

# Project-2

On s’intéresse désormais à la possibilité d’appliquer la méthode de Super Résolution à une vidéo et non plus à une image unique. Une publication [4] « SR_for_video.pdf » traite du sujet.

Il est intéressant de noter pour le cas d’une image, le GAN ne reçoit en entré que l’image Low Résolution visible sur l’image suivante.

![alt text](https://github.com/Idelcads/IMKI_Technical_test/blob/main/Images_readme/1.png)

Pour le cas d’une vidéo, le GAN reçoit en entré plusieurs images Low resolution t-1, t, t+1 tiré de la vidéo, afin de générer une nouvelle image t High resolution comme visible dans l’exemple suivant.

![alt text](https://github.com/Idelcads/IMKI_Technical_test/blob/main/Images_readme/2.png)

---

# Tutorial to better understand GAN generation

https://www.youtube.com/watch?v=5RYETbFFQ7s \
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html \
https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4 \

\
References

[1] Zhilao W., Jian Chen., Steven C.H. Hoi, Fellow, IEEE \
Deep Learning for Image Super-resolution: A Survey \
Scientific publication available at : https://arxiv.org/abs/1902.06068 \
[2] Christian Ledig, Lucas Theis, Ferenc Husz´ar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi \
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network \
Scientific publication available at : https://arxiv.org/abs/1609.04802 \
[3] Christopher Thomas \
Web article available at : https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5 \
[4] Santiago López-Tapia, Alice Lucas, Rafael Molina and Aggelos K. Katsaggelos. \
A Single Video Super-Resolution GAN for Multiple Downsampling Operators based on Pseudo-Inverse Image Formation Models \
Scientific publication available at : https://arxiv.org/ftp/arxiv/papers/1907/1907.01399.pdf \
