# L'intelligence artificielle dans la linguistique historique

Ce dépôt héberge le mémoire, l'article scientifique et les programmes éventuellement développés.

## Le mémoire

Ce document répond à la problématique : **quel est le potientiel de l'intelligence artificielle dans la linguistique historique ?**<br>
Il suivra le plan suivant :
1. **La linguistique historique et l’intelligence artificielle**
2. **Les contributions de l’IA dans la linguistique historique**
3. **Utilisation de l'IA pour la reconstruction d'une proto-langue**


## Article scientifique

Ce papier étudiera la **reconstruction d'une proto-langue par approche neuronale**, en particulier, la reconstruction du du proto-latin à partir de ses langues descendantes (français, espagnol, portugais, italien).

## Code

Dans ce dossier, vous trouverez toutes les implémentations Python abordées dans les documents écrits précédents. (L'ojectif était de mettre en pratique la théorie étudiée dans l'article scientifique.) 
Il se découpe en deux sous-dossiers : **fine-tuned** et **PaperImplementation**.

*Tous les entrainements ont été effectués sur la base de données postée par [Shauli-Ravfogel](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL).*

### Fine-tuned :

Dans ce sous-dossier, deux modèles pré-entrainés ([mBART](https://huggingface.co/docs/transformers/model_doc/mbart) et [mT5](https://huggingface.co/docs/transformers/model_doc/mt5)) ont été *affinés* pour la tâche de reconstruction.<br>
Pour pouvoir les utiliser, il est nécessaire d'installer les librairies Python : `torch` et `transformers`. 

### PaperImplementation :

Ce sous-dossier contient notre implémentation du papier ["Neural Unsupervised Reconstruction of Protolanguage Word Forms"](https://arxiv.org/abs/2211.08684) de Andre He, Nicholas Tomlin, Dan Klein.<br>
Pour pouvoir lancer le fichier python, il est nécessaire d'installer la librairie `torch`. 