"""
  ----------------------------------------------------------------------------
  "THE BEER-WARE LICENSE" (Revision 42):
  <antbob@users.noreply.github.com> wrote this file. As long as you retain
  this notice you can do whatever you want with this stuff. If we meet
  some day, and you think this stuff is worth it, you can buy me a beer in
  return Anton Bobrov
  ----------------------------------------------------------------------------
"""

import argparse
import pycuber as pc
from pycuber.solver import CFOPSolver

parser = argparse.ArgumentParser()
parser.add_argument("size", help="number of sets", type=int)
parser.add_argument("file", help="where to write")
args = parser.parse_args()

faces = ["L", "R", "U", "D", "B", "F"]
colors = {"[r]": "r",
          "[y]": "y",
          "[g]": "g",
          "[w]": "w",
          "[o]": "o",
          "[b]": "b",
          "[u]": "u"}

def cubeAsArray(acube):
    cubeArray = []
    for face in faces:
        face = acube.get_face(face)
        for x in [0,1,2]:
            for y in [0,1,2]:
                cubeArray.append(colors[str(face[x][y])])
    return cubeArray

def solutionAsArray(asolution):
    solutionArray = []
    for step in asolution:
        solutionArray.append(str(step))
    return solutionArray

setfile = open(args.file, 'w')

for i in range(args.size):
  cube = pc.Cube()
  alg = pc.Formula()
  random_alg = alg.random()
  cube(random_alg)
  cube_array = cubeAsArray(cube)
  solver = CFOPSolver(cube)
  solution = solver.solve(suppress_progress_messages=True)
  solution_array = solutionAsArray(solution.optimise())
  setfile.write(' '.join(cube_array) + '\n')
  setfile.write(' '.join(solution_array) + '\n')

setfile.close()
