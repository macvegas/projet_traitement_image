
# Aller dans le dossier generated
cd generated

# Lancer la commande cmake configurant le projet pour Visual Studio 2017 (en mode compatibilité 2015) :
cmake -G "Visual Studio 15 2017 Win64" -T "v140" .. -DOpenCV_DIR="C:\Program Files\OpenCV_3.3.1\build"

# Ouvrir le projet dans Visual Studio :
cliquer sur opencv_test.sln

Choisir "Aucune mise à niveau" lors de l'ouverture du projet sous Visual Studio
Penser à sélectionner ce projet opencv_test comme projet de démarrage
Adapter le chemin vers l'image de test dans le main
Compiler et tester

