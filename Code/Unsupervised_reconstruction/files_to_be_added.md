# ➕Fichiers à ajouter pour entraîner les modèles

Pour des raisons de droits d'auteur, certains fichiers ne peuvent pas être publiés sur le dépôt distant.

## Programmes dynamiques

Dans le dossier [`./Source`](Source/), un script `dynamicPrograms.py`, développé par Andre He (*University of California*, Berkeley ; il n'est pas exclu que des auteurs supplémentaires soient à l'origine de ce fichier), doit être ajouté. Ce script peut être partagé sur demande (à travers la levée d'une *issue* par exemple), mais nous prions dans ce cas de ne pas le rendre public.

## Données de reconstruction

Un dossier `./recons_data` doit aussi être ajouté, contenant 4 listes d'échantillons produites par les chercheurs à partir d'itérations d'espérance-maximisations sur un modèle de reconstruction probabiliste ([2013](https://aclanthology.org/N09-1008/)). Les 3 premières listes servent alors aux itérations initiales des espérance-maximisations pour notre modèle de reconstruction, afin que les modèles d'édition neuronaux soient entraînés quelques fois au départ. La 4ème liste sert pour le premier échantillonage.\
Ces listes sont associées à un jeu de données (cognats romans et bons ancêtres latins pour l'évaluation) relevant de la propriété de [Alina Maria Ciobanu et Liviu P. Dinu](http://www.lrec-conf.org/proceedings/lrec2014/pdf/175_Paper.pdf). Celui-ci diffère par rapport à la révision de [Meloni et al.](https://aclanthology.org/2021.naacl-main.353/) qui est consultable [ici](https://github.com/shauli-ravfogel/Latin-Reconstruction-NAACL). Les détails de ces modifications sont précisés dans l'[article](https://arxiv.org/pdf/2211.08684.pdf) sur lequel nous avons basé notre implémentation.

Le dossier `./recons_data` a l'architecture suivante :

```cmd
|___recons_data/
    |__iteration3_1.txt # listes pour les itérations
    |__iteration3_2.txt
    |__iteration3_3.txt
    |__iteration3_4.txt
    |___data/
         |__Latin_ipa.txt    
         |__French_ipa.txt
         |__Italian_ipa.txt
         |__Roumanian_ipa.txt
         |__Spanish_ipa.txt
         |__Portuguese_ipa.txt
         |__Latin.txt # version orthographique
         |__French.txt
         |__Italian.txt
         |__Roumanian.txt
         |__Spanish.txt
         |__Portuguese.txt    
```

Ces ressources nous ont été fournies par Andre He et nous le remercions encore pour ce don.