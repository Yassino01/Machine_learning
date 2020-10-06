## Traitement automatique du langage

* Appréhender les modèles de classification et de fouille de texte (détection de sentiments, …)
* Identifier la sémantique des éléments du texte (extraction de thèmes, représentations latentes et contextuelles)


### Dans ce projet:
ON est confrontés à :

* Réaliser une classification des avis des internautes selon les
sentiments qu’ils portent sur les films, sachant qu'on dispose d’une base de données textuelle labellisée divisée en deux classes de sentiments ( positifs et négatifs)).

* Entraîner un modèle à reconnaître si des phrases proviennent d’un discour
prononcé par Chirac ou par Mitterrand.

### Problématiques :

* Pre-processing des données (stemming,tokenization,lettres minuscules, stopwords)
   * Comment transformer en bag-of-words ?

* Gestion du déséquilibre des classes dans les données.
   * Comment adapter l’entrainement ?
   * Comment bien choisir la métrique ?
   
* Evaluations des diffirents models utilisés suivant un paramétrage de pré-processing et en utilisant plusieurs métrics (precision, recall,f1_macro)
