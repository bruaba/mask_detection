# COVID - Mask Detection

Projet d'intelligence artificiel en 5A ILC à l'ESIREM

## Table of contents
* [Présentation](#présentation)
* [Technology](#technology)
* [Setup](#setup)
* [Rapport](#rapport)
* [Auteurs](#auteurs)
* [Ecole](#Ecole)


## Présentation

L’objectif de ce projet est de permettre aux étudiants de voir l’ensemble de la chaine permettant  de  concevoir  une  application  à  base  de  Machine  Learning  (incluant  les méthodes de Deep-Learning présentées lors des cours). 
Le projet porte donc sur les 3 axes principaux suivants:
* 1. Conception  d’une  base  d’images  annotées  destinée  à  l’entrainement  des modèles/algorithmes de Machine Learning.
* 2.Sélection  et  entrainement  d’un  algorithme/modèle  adapté  à  l’application choisie.
* 3.Mise en œuvre temps réel de cet algorithme pour présentation des résultats obtenus.

Elle permet de détecter 3 classes d’objets:
* 1.Label "without_mask": Visages sans masque
* 2.Label "mask_weared_incorrect": Visages avec un masque mal porté
* 3.Label "with_mask":Visages avec un masque (bien porté)


## Technologie
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
Le rapport concernat ce projet est contenu dans le fichier [Rapport](https://github.com/bruaba/mask_detection/blob/main/Rapport.pdf)

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

