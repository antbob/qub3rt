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
parser.add_argument("infile", help="input sets file")
parser.add_argument("outfile", help="output sequences file")
args = parser.parse_args()

faces = ["L", "U", "F", "D", "R", "B"]
colors = {"[r]": "0",
          "[y]": "1",
          "[g]": "2",
          "[w]": "3",
          "[o]": "4",
          "[b]": "5"}
kolors = {"r": "0",
          "y": "1",
          "g": "2",
          "w": "3",
          "o": "4",
          "b": "5"}

solved_cube = pc.Cube()

def cubeAsArray(acube):
    cubeArray = []
    for face in faces:
        face = acube.get_face(face)
        for x in [0,1,2]:
            for y in [0,1,2]:
                cubeArray.append(colors[str(face[x][y])])
    return cubeArray

setfile = open(args.infile, 'r')
seqfile = open(args.outfile, 'w')

for setline in setfile:
  setline = setline[0:17] + ' ' + \
            setline[36:53] + ' ' + \
            setline[90:107] + ' ' + \
            setline[54:71] + ' ' + \
            setline[18:35] + ' ' + \
            setline[72:89]
  cube_line = setline.split()
  cube_array = []
  for color in cube_line:
    cube_array.append(kolors[color])
  cube = pc.Cube(pc.array_to_cubies(cube_array))
  try:
    setline = next(setfile)
  except StopIteration:
    break
  solution = setline.split()
  for step in solution:
    seqfile.write(' '.join(cubeAsArray(cube)) + '\n')
    seqfile.write(step + '\n')
    cube.perform_step(step)
  if cube == solved_cube:
    seqfile.write(' '.join(cubeAsArray(cube)) + '\n')
    seqfile.write('#' + '\n')
  else:
    raise Exception()

seqfile.close()
setfile.close()
