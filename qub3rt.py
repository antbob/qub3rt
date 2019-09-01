# This should run on Python 2, like ev3dev side, to match rpyc service.
"""
  ----------------------------------------------------------------------------
  "THE BEER-WARE LICENSE" (Revision 42):
  <antbob@users.noreply.github.com> wrote this file. As long as you retain
  this notice you can do whatever you want with this stuff. If we meet
  some day, and you think this stuff is worth it, you can buy me a beer in
  return Anton Bobrov
  ----------------------------------------------------------------------------
"""

import sys
import rpyc
import argparse
import subprocess
import numpy as np

RPYCPORT = 18861
LOGNAME = 'QUB3RT'
PYCMD = 'python3 '
MODELCMD = 'model.py '

# Square colors map, must match TF model.
COLORS = {"L": "0", # red
          "U": "1", # yellow
          "F": "2", # green
          "D": "3", # white
          "R": "4", # orange
          "B": "5"} # blue

# Translate colors and sides order. Cube representation on
# ev3dev side differs from TF model PyCuber representation.
def translate_cube(cube_scan):
  pycube = []
  for square in cube_scan:
    pycube.append(COLORS[square])

  translate_cube = []
  translate_cube.extend(pycube[36:45])
  translate_cube.extend(pycube[0:9])
  translate_cube.extend(pycube[18:27])
  translate_cube.extend(pycube[27:36])
  translate_cube.extend(pycube[9:18])
  translate_cube.extend(pycube[45:54])

  return translate_cube

parser = argparse.ArgumentParser()
parser.add_argument("host", help="ev3dev host")
args = parser.parse_args()

try:
  print '\n' + LOGNAME + ':connect:' + args.host
  conn = rpyc.connect(args.host, RPYCPORT, keepalive=True)
except Exception as e:
  sys.exit(e)

conn._config['sync_request_timeout'] = None

print ('\n' + LOGNAME + ':scan:' +
    'place the cube on ev3 platform,' +
    ' middle yellow square facing up,' +
    ' middle red square facing out')

try:
  print '\n' + LOGNAME + ':scan:' + 'start'

  cube_scan = conn.root.scan_cube()

  if cube_scan is None:
    sys.exit("scan failed")

  print '\n' + LOGNAME + ':scan:' + str(cube_scan)

  model_cube = translate_cube(cube_scan)

  print '\n' + LOGNAME + ':model:' + str(model_cube)
  print '\n' + LOGNAME + ':model:' + 'start' + '\n'

  model_output = subprocess.check_output(PYCMD + MODELCMD +
      '--cube ' + ''.join(model_cube), shell=True)

  print model_output

  # The last line of the model output are actions.
  actions = model_output.splitlines()[-1].split(' ')

  if actions is None:
    sys.exit("subprocess failed")

  print '\n' + LOGNAME + ':actions:' + str(actions)
  print '\n' + LOGNAME + ':actions:' + 'start'

  conn.root.resolve_cube(actions)

  print '\n' + LOGNAME + ':cube:' + 'done' + '\n'
except Exception as e:
  sys.exit(e)
finally:
  conn.close()
