# When we run this code by using chmod u+x ./main_Table.sh and ./main_Table.sh on terminal, 
# we have numerical error tables with fixed timesteps or spatial meshes, respectively

#!/bin/bash
chmod u+x ./Table1.py
chmod u+x ./Table2.py


# generate a table as shown in Table 1
./Table1.py|& tee -a Table_1.txt



# generate a table as shown in Table 2
./Table2.py|& tee -a Table_2.txt