import snap
import sys
sys.path.append("..") # Adds higher directory to python modules path

from models import hybrid



g = snap.GenRndGnm(snap.PNGraph, 10, 10)
h = hybrid.HybridModel(g)
scores = h.generate_scores()

print(scores)

