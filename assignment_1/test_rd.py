# quick and dirty way. Dunno where the compiled shared library is stored, but
# it won't recompile unless changes were made in *.pyx module
import pyximport
pyximport.install()
import rd_fast

if __name__ == '__main__':
    rd_fast.set_seed()
    print("Gauss:", [rd_fast.gauss() for _ in range(10)])
    print("U(0, 1):", [rd_fast.rand() for _ in range(10)])
    print("Randint:", [rd_fast.randint(2) for _ in range(10)])
    print("Exponential", [rd_fast.exponential() for _ in range(10)])
    print("Beta (2, 3):", [rd_fast.beta(2, 3) for _ in range(10)])
    rd_fast.test()
