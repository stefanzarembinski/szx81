import random
import math

class Sinus:
    def __init__(self, noise=0.03, stop=10000, start=0, step=1):
        self.noise = noise
        self.start = start
        self.step = step
        self.stop = stop 

    def __getitem__(self, i):
        # import pdb; pdb.set_trace()
        if isinstance(i, slice):
            if i.start is not None:
                self.start = i.start
            if i.step is not None:
                self.step = i.step
            if i.stop is not None:
                self.stop = i.stop
            return Sinus(self.noise, self.stop, self.start, self.step)
        if i < 0:
            i = self.length + i
        
        k = ((i + self.start) // self.step) * self.step
        if k < self.start or k >= self.step:
            raise IndexError(f'index is {k} while it has to be within limits [{self.start}, {self.stop}[')
        return .5 * math.sin(k * .03 * self.step) \
                + random.uniform(-self.noise, self.noise) + .5
    
    def __len__(self):
        return self.stop - self.start
    
def main():
    
    # sin = Sinus(noise=0, len=20)[:3]
    sin = Sinus(noise=0, stop=20)
    sin = sin[:18]
    print(f'''
len(sin): {len(sin)}
sin[10] {sin[10]}
'''
        )
    try:
        sin[21]
    except Exception as ex:
        print(ex)
    sin[0]


if __name__ == "__main__":
    main()