# COVID - Mask Detection

Projet d'intelligence artificiel en 5A ILC à l'ESIREM

## Table of contents
* [Objectif](#objectif)
* [Technology](#technology)
* [Setup](#setup)
* [Rapport](#rapport)
* [Auteurs](#auteurs)
* [Ecole](#Ecole)


## Objectif

L’objectif de ce projet est de permettre aux étudiants de voir l’ensemble de la chaine 
permettant de concevoir une application à base de Machine Learning (incluant les méthodes de Deep-Learning présentées lors des cours).


## Technology
Dans ce projet, on a utilisé les technologies suivante:
* Tensorflow
* OpenCV
* Matplotlib
* Keras

On peut installer les requerements avec la commande suivante:

```
$ pip install -r requirements.txt
```

## Setup
Pour demarer le projet, on peut soit refaire un entrainement en executant la commande suivante en lui passant en paramètre le nom du dataset ou son chemin d'accès:
```
$ python3 train.py "Dataset" 
```
Il est à noter aussi qu'il est préferable d'avoir des images .jpg et deux dossier dans le dataset:
* un dossier images 
* un dossier annotations

Au lieu de faire un entrainement, on peut executer directement le modèle qui se trouve dans le dossier Model.

Ainsi on peut:

* soit on démarre un flux vidéo avec le nom du model en paramètre
```
$ python3 detect_mask_video.py "the_model" 
```

* Ou faire afficher les images provenant du dataset qu'on lui passera en paramètre avec le modèle:
```
$ python3 detect_mask_image.py "the_model" "Dataset"
```

Le fichier facemaskrecognitionv2.ipynb correspond au train avec google colab

## Rapport
Le rapport concernat ce projet est contenu dans le fichier [MARONE_Rapport_Projet_DetecteurDeMasque](https://github.com/bruaba/mask_detection/blob/main/MARONE_Rapport_Projet_DetecteurDeMasque.pdf)

## Auteurs
Cheikh Ahmet Tidiane Chérif MARONE 
* maronho16@gmail.com 
* https://bitbucket.org/bruaba/

Elio Khater
* eliokhater@gmail.com




## Ecole
* ESIREM https://esirem.u-bourgogne.fr/
* Année: 2021
* Avec Olivier Brousse olivier.brousse@yumain.fr

