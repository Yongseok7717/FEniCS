# When we run this code by using chmod u+x ./main_Figure1.sh and ./main_Figure1.sh on terminal, 
# we have numerical convergent order graphs with resepct to internal variables and degree of polynomial 

#!/bin/bash
chmod u+x ./CG_P1.py
chmod u+x ./CG_P2.py
chmod u+x ./graph.py

# to get numerical results for linear polynomials
./CG_P1.py -k 1 -i 1 -I 10 -j 1 -J 10
./CG_P2.py -k 1 -i 1 -I 10 -j 1 -J 10

# to get numerical results for quadratic polynomials
./CG_P1.py -k 2 -i 1 -I 10 -j 1 -J 10
./CG_P2.py -k 2 -i 1 -I 10 -j 1 -J 10

# generate graphs as shown in Figure 1
./graph.py