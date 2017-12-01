#prime numbers
#python3 problem7.py <numberPrime>

import sys

numberPrime=int(sys.argv[1])

num=3
count=1

while count<numberPrime:
    for x in range(2, num):
        if num%x==0:
            break
        elif x==num-1:
            count+=1
    num+=1
print(num-1)
