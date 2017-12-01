class RSA:
    def __init__(self,p=2819204254813,q=20820252664147,e=17):
        self.p=p
        self.q=q
        self.n=p*q
        self.k=(p-1)*(q-1)  #phi(n)
        self.e=e

    def euclideanMethodGCD(self):   # e*d=1 (mod phi(n))
        k = self.k
        e = self.e
        t = int(k/e)
        r = k-t*e
        while True:     # k = t*e + r
            print(str(k) + " = " + str(t) + "*" + str(e) + " + " + str(r))
            oldk = k
            olde = e
            oldt = t
            oldr = r

            k = olde
            e = oldr
            t = int(k/e)
            r = k-t*e
            if r==0:
                print(str(k) + " = " + str(t) + "*" + str(e) + " + " + str(r))
                return e

    def inverseMod(self):
        t=0
        r=self.k
        newt=1
        newr=self.e
        while newr != 0:
            q = r/newr
            temp=t
            t=newt
            newt=temp-int(q*newt)
            temp=r
            r=newr
            newr=temp-int(q*newr)
        if r>1:
            return "not possible"
        if t<0:
            t=t+n
        return t


j = RSA(257, 337)
i = j.inverseMod()
print(i)
