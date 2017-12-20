# Projet de traitement des images
Camille Saulnier, Simon Beaulieu, Clément Fournier

## Description du projet

Nous avons commencé par repérer tous les rectangles dans l’image grâce à une fonction de traitement disponible sur le git https://github.com/alyssaq/opencv.
A partir de tous les rectangles, nous avons gardé uniquement ceux dont la largeur correspondait à la largeur des carrés à découper. 
Puis, nous avons ordonné les points à l’intérieur des vecteurs de points puis ordonné les vecteurs entre eux. 
Grâce à ce traitement, nous avons déterminé les carrés qui se chevauchaient en établissant une distance minimale entre deux points de carrés différents. 
Nous avons trié chaque carré par ligne.
Nous avons déterminé, à l’aide des coordonnées de chaque ligne, la zone contenant l’image à identifier puis nous l’avons identifié grâce à la fonction matchtemplate. Nous avions établi une base de données avec les symboles existants et nous établissons à quelle image le symbole à reconnaître ressemble le plus. Puis à quelle taille l’image correspond (grâce à la même fonction).
Ensuite, pour chaque carré de chaque ligne, nous enregistrons la sous image avec le nom : iconeID_scripterNumber_pageNumber_row_column.png


### Instructions pour cloner
Aller dans le dossier generated
```
cd generated
```

Lancer la commande cmake configurant le projet pour Visual Studio 2017 (en mode compatibilité 2015) :
```
cmake -G "Visual Studio 15 2017 Win64" -T "v140" .. -DOpenCV_DIR="C:\Program Files\OpenCV_3.3.1\build"
```

Ouvrir le projet dans Visual Studio (cliquer sur opencv_test.sln)

Choisir "Aucune mise à niveau" lors de l'ouverture du projet sous Visual Studio
Penser à sélectionner ce projet opencv_test comme projet de démarrage
Adapter le chemin vers l'image de test dans le main
Compiler et tester
