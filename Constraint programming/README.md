### Sudoku solver algo:

Le probl`eme du Sudoku sous forme de CSP s’´ecrit :

> X = {81 cases identifi´ees par (ligne,colonne)}

> D = {0 (non initialis´ee), {1, 2, . . . , 9} (initialis´ee)}

> C = {Toutes diff´erentes(ligne), Toutes diff´erentes(colonne),
Toutes diff´erentes(cellule)}

Voici l’algorithme pour r´esoudre un Sudoku. Il utilise un simple
retour arri`ere et un filtrage (consistance de noeud).

Tant que toutes les cases ne sont pas remplies :

> P = Valeurs possibles de la case `a remplir courante (filtrage)

> Si P est vide alors on revient `a la case remplie pr´ec´edement

> Sinon on renseigne la case courante avec une valeur de P et
on passe `a la case suivante

L’´etape de filtrage sur la consistance de noeud est tr`es simple :
P = {1, . . . , 9} - (toutes les valeurs initialis´ees dans la mˆeme ligne,
colonne et cellule que la case courante).
