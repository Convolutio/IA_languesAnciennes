Lire en [Français](#lintelligence-artificielle-dans-la-linguistique-historique).\
Read in [English](#artificial-intelligence-in-historical-linguistics).

____

# L'intelligence artificielle dans la linguistique historique

Ce dépôt héberge le mémoire, l'article scientifique et les programmes éventuellement développés.

## Le mémoire

Ce document répond à la problématique : **quel est le potientiel de l'intelligence artificielle dans la linguistique historique ?**\
Il suivra le plan suivant :

1. **La linguistique historique et l’intelligence artificielle**
2. **Les contributions de l’IA dans la linguistique historique**
3. **Utilisation de l'IA pour la reconstruction d'une proto-langue**

## Article scientifique

Ce papier s'intéresse à la **reconstruction d'une proto-langue par approche neuronale**, en particulier, la reconstruction du proto-latin à partir de ses langues descendantes (français, espagnol, portugais, italien, roumain). Ici, l'impact des propriétés du modèle de langue utile dans une approche non supervisée est étudié.

## Code

Dans ce dossier, toutes les implémentations Python abordées dans les documents écrits précédents sont jointes. (L'objectif était de mettre en pratique l'expérience établie dans l'article scientifique.)\
Il se découpe en deux sous-dossiers : **[Fine-tuned](Code/Fine-tuned/)** et **[Unsupervised_reconstruction](Code/Unsupervised_reconstruction/)**.

*Tous les entrainements ont été effectués à partir de la base de données postée par [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

### Fine-tuned

Dans ce sous-dossier, deux modèles pré-entrainés ([mBART](https://huggingface.co/docs/transformers/model_doc/mbart) et [mT5](https://huggingface.co/docs/transformers/model_doc/mt5)) ont été *affinés* pour la tâche de reconstruction.\
Pour pouvoir les utiliser, il est nécessaire d'installer les librairies Python : `torch` et `transformers`.

### Unsupervised_reconstruction

Ce sous-dossier contient notre implémentation du papier ["Neural Unsupervised Reconstruction of Protolanguage Word Forms"](https://arxiv.org/abs/2211.08684) de Andre He, Nicholas Tomlin, Dan Klein, ainsi que celle de nos expérimentations. La librairie Python `torch` a également été utilisée.

___

# Artificial intelligence in historical linguistics

This repository hosts the dissertation, the scientific article and any programmes developed.

## The dissertation

This document answers the following question: **What is the potential of artificial intelligence in historical linguistics?**\
It will follow the following plan:

1. **Historical linguistics and artificial intelligence**.
2. **AI's contributions to historical linguistics**.
3. **The use of AI for the reconstruction of a proto-language**.

## Scientific paper

This paper focuses on the **reconstruction of a proto-language using a neural approach**, in particular, the reconstruction of proto-Latin from its descendant languages (French, Spanish, Portuguese, Italian, Romanian). Here, the impact of the properties of the language model useful in an unsupervised approach is studied.

## Code

In this folder, all the Python implementations discussed in the previous written documents are attached. (The aim was to put into practice the experiment established in the scientific article.)\
It is divided into two sub-folders: **[Fine-tuned](Code/Fine-tuned/)** and **[Unsupervised_reconstruction](Code/Unsupervised_reconstruction/)**.

*All training was carried out using the database posted by [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

### Fine-tuned

In this sub-folder, two pre-trained models ([mBART](https://huggingface.co/docs/transformers/model_doc/mbart) and [mT5](https://huggingface.co/docs/transformers/model_doc/mt5)) were fine-tuned for the reconstruction task.\
To use them, install the Python libraries: `torch` and `transformers`.

### Unsupervised_reconstruction

This sub-folder contains our implementation of the paper ["Neural Unsupervised Reconstruction of Protolanguage Word Forms"](https://arxiv.org/abs/2211.08684) by Andre He, Nicholas Tomlin, Dan Klein, as well as that of our experiments. The `torch` Python library was also used.
