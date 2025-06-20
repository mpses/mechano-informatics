import numpy as np

class Patterns:
    PATTERNS = [
        np.array([[+1, +1, +1, +1, +1],
                  [+1, -1, -1, -1, +1],
                  [+1, -1, -1, -1, +1],
                  [+1, -1, -1, -1, +1],
                  [+1, +1, +1, +1, +1]]),
        np.array([[-1, +1, -1, +1, -1],
                  [+1, -1, +1, -1, +1],
                  [-1, +1, -1, +1, -1],
                  [+1, -1, +1, -1, +1],
                  [-1, +1, -1, +1, -1]]),
        np.array([[-1, -1, +1, -1, -1],
                  [-1, -1, +1, -1, -1],
                  [+1, +1, +1, +1, +1],
                  [-1, -1, +1, -1, -1],
                  [-1, -1, +1, -1, -1]]),
        np.array([[+1, +1, +1, +1, +1],
                  [-1, -1, +1, -1, -1],
                  [-1, -1, +1, -1, -1],
                  [-1, -1, +1, -1, -1],
                  [-1, -1, +1, -1, -1]]),
        np.array([[-1, +1, -1, -1, -1],
                  [-1, +1, -1, -1, -1],
                  [-1, +1, -1, -1, -1],
                  [-1, +1, -1, -1, -1],
                  [-1, +1, -1, -1, -1]]),
        np.array([[-1, +1, +1, +1, -1],
                  [+1, -1, -1, -1, +1],
                  [+1, -1, -1, -1, +1],
                  [+1, -1, -1, -1, +1],
                  [-1, +1, +1, +1, -1]]),
    ]
