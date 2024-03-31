from matplotlib import pyplot as plt
import numpy as np  
import re


fname = "../NASBase/log_test_supernet_256_wL.txt"


all_data = []

with open(fname) as infile:
    copy = False
    for line in infile:
        if  "Supernet Validation" in line.strip():
            #print(line)
            m = re.search('best\_acc:(.+?)\\n', line)
            if m:
                found = m.group(1).strip()
                print(found.strip())
                all_data.append(float(found))



plt.plot(all_data)
plt.show()