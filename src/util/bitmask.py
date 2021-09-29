import numpy as np


class BitMask():
    def __init__(self, girth: int, random=True):
        if random:
            self.bits = [bool(np.random.randint(0, 2))
                         for _ in range(girth)]
        else:
            self.bits = np.zeros(shape=(1, girth), dtype=bool)

    def __str__(self) -> str:
        return str([int(x) for x in self.bits])

    def __len__(self):
        return len(self.bits)

    # iteratively add one to the bitmask
    def add_one(self):
        for i in range(len(self.bits) - 1, -1, -1):
            if not self.bits[i]:
                self.bits[i] = True
                break
            else:
                self.bits[i] = False

    # iteratively subtract one to the bitmask
    def subtract_one(self):
        rightmost = -1
        for i in range(len(self.bits)):
            if self.bits[i] == True:
                rightmost = i
        for i in range(rightmost, len(self.bits)):
            self.bits[i] = bool(self.bits[i] ^ 1)
