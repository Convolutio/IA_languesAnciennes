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
