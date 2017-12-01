def fib():
    n=100
    a, b = 0, 1
    while b<n:
        print b,
        a,b = b, a+b

if __name__=='__main__':
    fib()
