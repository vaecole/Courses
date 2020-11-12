"""
dynamic_programming: fibonacci: O(2^n) by legacy method, divide and conquer
"""

import sys


class Fibonacci:
    def __init__(self):
        self.mem = dict()

    def fab(self, n):
        if n == 2:
            return 2
        if n == 1:
            return 1
        if self.mem.__contains__(n):
            return self.mem.get(n)
        self.mem[n] = self.fab(n - 1) + self.fab(n - 2)
        return self.mem[n]


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    fib = Fibonacci()
    print(fib.fab(3200))
