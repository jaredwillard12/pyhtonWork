def closest2n(a):
    last=0
    while 2**last<=a:
        last+=1
    return 2**(last-1)

horiz=0
vert=0

num=input("Number: ")
num=int(num)

i = closest2n(num)
perm = i

best=0

while True:
    count=1
    perm=i
    while i>=1 and horiz+i<=num and vert+1<=num:
        horiz+=i
        vert+=1
        a="{} and {}: h={} v={}".format(i, count, horiz, vert)
        print(a)
        i/=2
        count+=1
    i=2
    while horiz+1<=num and vert+i<=num:
        horiz+=1
        vert+=i
        a="{} and {}: h={} v={}".format(i, count, horiz, vert)
        print(a)
        i*=2
        count+=1
    print(count)
    i = perm/2
    horiz=0
    vert=0
    print("NOW" + str(i))
    if count>best:
        best=count
    if i==1:
        break
i=1
horiz=0
vert=0
count=1
while horiz+1<=num and vert+i<=num:
    horiz+=1
    vert+=i
    a="{} and {}: h={} v={}".format(i, count, horiz, vert)
    print(a)
    i*=2
    count+=1
if count>best:
    best=count
print(best)
