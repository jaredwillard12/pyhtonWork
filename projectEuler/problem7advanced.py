#prime numbers
#python3 problem7.py <numberPrime>

def isPrimeSquareRootRule(num):
    import numpy as np
    if num==2:
        return True
    elif num%2==0:
        return False
    else:
        numSqrt=int(np.sqrt(num))
        for i in range(1, numSqrt+1):
            if i!=1 and num%i==0:
                return False
            elif i==numSqrt:
                return True

def isPrimeFermatsLittleTheorem(p, a=2):
    math=a**(p-1)-1
    if math%p==0 or p==2:
        return True
    else:
        return False

def isPrimeStrongPRP(p, a=2):   #solving n-1=2^(s)*d where n is odd
    temp=p-1
    s=1
    while temp/2%2==0:
        s+=1
        temp/=2
    d=int((p-1)/(2**s))
    if pow(a,d,p)-1==0:     # a^d=1 (mod n)
        return True
    for j in range(s):
        if pow(a,d*(2**(j)),p)==p-1:    # (a^d)^(2^r)=-1 (mod n)
            return True
    return False

def provingPrimalityWithSPRP(p):
    if p<170584961 and isPrimeStrongPRP(p,350) and isPrimeStrongPRP(p,3958281543):
        return True
    elif p<75792980677 and isPrimeStrongPRP(p,2) and isPrimeStrongPRP(p,379215) and isPrimeStrongPRP(p,457083754):
        return True
    elif p<21652684502221 and isPrimeStrongPRP(p,2) and isPrimeStrongPRP(p,1215) and isPrimeStrongPRP(p,34862) and isPrimeStrongPRP(p,574237825):
        return True
    else:
        return False


def fromCommandLine():
    import sys
    numberPrime=int(sys.argv[1])

    num=1
    count=0

    while count<numberPrime:
        num+=1
        if isPrimeSquareRootRule(num):
            count+=1
    print(num)


if __name__ == "__main__":
    fromCommandLine()
