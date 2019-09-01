"""
  ----------------------------------------------------------------------------
  "THE BEER-WARE LICENSE" (Revision 42):
  <antbob@users.noreply.github.com> wrote this file. As long as you retain
  this notice you can do whatever you want with this stuff. If we meet
  some day, and you think this stuff is worth it, you can buy me a beer in
  return Anton Bobrov
  ----------------------------------------------------------------------------
"""

import os
import sys
import glob
import shutil
import argparse
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import pycuber as pc
from pycuber.solver import CFOPSolver

# TF model constants.
TRAIN_STEPS = 100000
VALID_STEPS = 10000
EPOCHS = 3

# 18 possible steps plus no step when solved.
STEPS = ["#", "F", "R", "U", "L", "B", "D",
         "F'", "R'", "U'", "L'", "B'", "D'",
         "F2", "R2", "U2", "L2", "B2", "D2"]

# Cube faces, order matters.
FACES = ["L", "U", "F", "D", "R", "B"]

# Square colors map.
COLORS = {"[r]": "0",
          "[y]": "1",
          "[g]": "2",
          "[w]": "3",
          "[o]": "4",
          "[b]": "5"}

# Feature column names. TF sorts feature columns by name (deterministic)
# which makes a mess with 1..54 naming when debugging input/output so
# name them accordingly to prevent sorting from changing the order.
COLUMNS = {"1": "a1", "2": "a2", "3": "a3", "4": "a4", "5": "a5",
           "6": "a6", "7": "a7", "8": "a8", "9": "a9",
           "10": "b1", "11": "b2", "12": "b3", "13": "b4", "14": "b5",
           "15": "b6", "16": "b7", "17": "b8", "18": "b9",
           "19": "c1", "20": "c2", "21": "c3", "22": "c4", "23": "c5",
           "24": "c6", "25": "c7", "26": "c8", "27": "c9",
           "28": "d1", "29": "d2", "30": "d3", "31": "d4", "32": "d5",
           "33": "d6", "34": "d7", "35": "d8", "36": "d9",
           "37": "e1", "38": "e2", "39": "e3", "40": "e4", "41": "e5",
           "42": "e6", "43": "e7", "44": "e8", "45": "e9",
           "46": "f1", "47": "f2", "48": "f3", "49": "f4", "50": "f5",
           "51": "f6", "52": "f7", "53": "f8", "54": "f9"}

# Cube in solved state.
SOLVED_CUBE = pc.Cube()

# Arguments parsing goes first.
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cube",
  help="Cube to predict a solution for, " +
       "should be a numerical sequence of cube colors " +
       "with sides as " + ''.join(FACES) + "/RYGWOB " +
       "000000000111111111222222222333333333444444444555555555",
  default=None, required=False)
args = parser.parse_args()

# Cube object to array.
def cubeAsArray(acube):
    cubeArray = []
    for face in FACES:
        face = acube.get_face(face)
        for x in [0,1,2]:
            for y in [0,1,2]:
                cubeArray.append(COLORS[str(face[x][y])])
    return cubeArray

# Features dict will have an array of values for each square as
# they evolve from the initial cube state to solved cube state.
def build_features_dict(features_dict, features_array):
  for lindex in range(1,55):
    features_dict.update({COLUMNS[str(lindex)]:
        np.array(features_array[lindex - 1])})
    lindex = lindex + 1

# Features array will have an array of values for each square as
# they evolve from the initial cube state to solved cube state.
def build_features_array(features_array, cube_array):
  nsquare = 0
  for square in cube_array:
    features_array[nsquare].append(int(square))
    nsquare = nsquare + 1

# Return an initialized features array.
def init_features_array():
  features_array = []
  for _ in range(54):
    features_array.append([])
  return features_array

# This will yield a batch of cube states and steps which
# will span from initial cube state to solved cube state.
def generate_input_arrays(cubes_list, steps_list):
  while True:
    gindex = 0
    steps_array = []
    features_array = init_features_array()
    features_dict = dict()
    for cube in cubes_list:
      build_features_array(features_array, list(cube))
      steps_array.append(STEPS.index(steps_list[gindex]))
      if steps_list[gindex] == '#':
        build_features_dict(features_dict, features_array)
        yield (features_dict, np.array(steps_array))
        features_array = init_features_array()
        features_dict = dict()
        steps_array = []
      gindex = gindex + 1

# Yields the current cube state and the next step, dynamically.
def generate_eval_arrays(cube_array, solution):
  cube = pc.Cube(pc.array_to_cubies(cube_array))
  features_array = init_features_array()
  features_dict = dict()
  for step in solution:
    build_features_array(features_array, cube_array)
    build_features_dict(features_dict, features_array)
    yield (features_dict, np.array([STEPS.index(step)]))
    if step == '#' and cube == SOLVED_CUBE:
      break
    cube.perform_step(step)
    cube_array = cubeAsArray(cube)
    features_array = init_features_array()
    features_dict = dict()

# Read the entire dataset and transform it for fit.
# This isnt exactly optimal or efficient but since
# the dataset was evolving so was this code step by
# step. Needs to be optimized for larger datasets.
def process_training_data():
  dataset_dir = 'dataset/'
  file_pattern = 'training.seq.*'
  dataset_files = []

  for path in glob.glob(os.path.join(dataset_dir, file_pattern)):
    dataset_files.append(path)

  # Might be a good idea to shuffle them randomly but sort in order now.
  dataset_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

  cubes = [[]]
  steps = [[]]
  aindex = 0
  nindex = 0

  print('# Processing training dataset')

  for setfilename in dataset_files:
    setfile = open(setfilename, 'r')

    for setline in setfile:
      if nindex != aindex:
        cubes.append([])
        steps.append([])
      aindex = nindex
      next_cube = setline.replace(' ', '').strip('\n')
      next_step = next(setfile).strip('\n')
      if next_step == '#':
        nindex = aindex + 1
      cubes[aindex].append(next_cube)
      steps[aindex].append(next_step)

    setfile.close()

  # Convert and split to training/validation frames.
  df = pd.DataFrame({'cubes': cubes, 'steps': steps})
  train_frame = df.head(TRAIN_STEPS)
  validate_frame = df.tail(VALID_STEPS)

  # Transform again in order to feed the generator function more easily.
  cubes_train_list = np.concatenate(train_frame['cubes'].values.tolist())
  steps_train_list = np.concatenate(train_frame['steps'].values.tolist())
  cubes_validate_list = np.concatenate(validate_frame['cubes'].values.tolist())
  steps_validate_list = np.concatenate(validate_frame['steps'].values.tolist())

  print('\n  processed:', len(df.index), 'batches')

  return (cubes_train_list, steps_train_list,
          cubes_validate_list, steps_validate_list)

# Helper method for debugging inputs.
def print_layer_input(x, layer_name):
  tf.print(x.shape)
  tf.print(layer_name, x, summarize=-1)
  return x

# Helper method to get top N predictions, sorted order.
def get_top_predictions(t_predictions, ntop=len(STEPS)):
  topp = []
  _, indices = tf.math.top_k(
      t_predictions, k=ntop, sorted=True)
  for index in indices.numpy():
    topp.append(STEPS[index])
  return topp

# Each square on the cube is represented by its own feature column.
feature_columns = []
for column in range(1,55):
  feature_columns.append(
      tf.feature_column.numeric_column(COLUMNS[str(column)]))

# Include the epoch in checkpoint filename (`str.format`)
checkpoint_path = "tf_ckpts/cp-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights.
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=EPOCHS)

# Here goes the model.
model = tf.keras.Sequential()
model.add(tf.keras.layers.DenseFeatures(feature_columns, name='input'))

#model.add(tf.keras.layers.Lambda(print_layer_input,
#    arguments={'layer_name': 'Feature columns'}))

model.add(tf.keras.layers.Dense(1024, name='relu-1', activation='relu'))
model.add(tf.keras.layers.Dense(2048, name='relu-2', activation='relu'))
model.add(tf.keras.layers.Dense(1024, name='relu-3', activation='relu'))
model.add(tf.keras.layers.Dense(19, name='softmax', activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard stuff here.
tf_log_dir = 'logs/fit/'
# Remove any old logs.
shutil.rmtree(tf_log_dir, ignore_errors=True)
tf_log_dir = tf_log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Histogram computation every epoch.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_log_dir,
    histogram_freq=1)

# Restore weights from the latest checkpoint if its available.
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint is None:

  # Load and process the dataset.
  (training_features, training_targets, validation_features,
   validation_targets) = process_training_data()

  print('\n# Fit model on training data\n')

  history = model.fit_generator(
                      generator=generate_input_arrays(
                          training_features, training_targets),
                      steps_per_epoch=TRAIN_STEPS,
                      epochs=EPOCHS,
                      callbacks=[cp_callback, tensorboard_callback],
                      # We pass some validation for
                      # monitoring validation loss and metrics
                      # at the end of each epoch
                      validation_data=generate_input_arrays(
                          validation_features, validation_targets),
                      validation_steps=VALID_STEPS)

  # The returned "history" object holds a record
  # of the loss values and metric values during training
  print('\nhistory dict:', history.history)

  print('\n# Evaluate on test data')

  eval_cube = pc.Cube()
  alg = pc.Formula()
  random_alg = alg.random()
  eval_cube(random_alg)
  eval_cube_array = cubeAsArray(eval_cube)
  solver = CFOPSolver(eval_cube)
  eval_solution = solver.solve(suppress_progress_messages=True)
  eval_solution = eval_solution.optimise()
  eval_solution = str(eval_solution).split()
  eval_solution.append('#')

  results = model.evaluate_generator(generate_eval_arrays(
                                     eval_cube_array, eval_solution),
                                     steps=len(eval_solution))
  print('test loss, test acc:', results)

  print('\n# Solved cube sanity check')

  solved_cube = pc.Cube()
  solved_cube_array = cubeAsArray(solved_cube)
  solved_features_array = init_features_array()
  solved_features_dict = dict()
  build_features_array(solved_features_array, solved_cube_array)
  build_features_dict(solved_features_dict, solved_features_array)

  predictions = model.predict(solved_features_dict)
  predictions = tf.reshape(predictions, [-1])
  predictions_max = tf.math.argmax(input=predictions)
  predictions_next_step = tf.keras.backend.eval(predictions_max)
  predicted_next_step = STEPS[int(predictions_next_step)]
  print('\n  Next step predicted:', predicted_next_step)
  if predicted_next_step != '#':
    sys.exit("sanity check failed")

else:
  # Load the previously saved weights
  model.load_weights(latest_checkpoint)
  print('\n# Fit restored from checkpoint\n')

# At this point the model is either done
# training or restored from checkpoint.

predict_cube = None

# Get scrambled cube.
if args.cube is None:
  print('\n  Generating random cube\n')
  predict_cube = pc.Cube()
  alg = pc.Formula()
  random_alg = alg.random()
  predict_cube(random_alg)
else:
  predict_cube = pc.Cube(pc.array_to_cubies(list(args.cube)))

predict_cube_array = cubeAsArray(predict_cube)

# PyCuber can verify that a cube is valid (read solvable)
# however it can let some things slip eg color mismatches.
try:
  print('\n  Verify cube\n')
  py_cube = pc.Cube(pc.array_to_cubies(predict_cube_array))
  py_solver = CFOPSolver(py_cube)
  py_solver.solve(suppress_progress_messages=True)
except ValueError as ve:
  sys.exit(ve)

print('\n  Initial cube:\n')
print(repr(predict_cube))

predicted_cube_arrays = [predict_cube_array]
predicted_solution = []
predictions_index = 0

print('\n# Generate predictions')

# While model accuracy is pretty good it is not perfect so it will
# trip from time to time. The logic here is to detect when it does
# and backtrack allowing it to try lesser scored predictions next.
# Cap max iterations just in case an infinite loop happens somehow.
for nstep in range(0,1000):

  predict_features_array = init_features_array()
  predict_features_dict = dict()
  build_features_array(predict_features_array, predict_cube_array)
  build_features_dict(predict_features_dict, predict_features_array)

  predictions = model.predict(predict_features_dict)
  predictions = tf.reshape(predictions, [-1])
  top_predictions = get_top_predictions(predictions)

  # Should not happen but if it does, end gracefully.
  if predictions_index > (len(top_predictions) - 1):
    print('\n  Predictions exhausted, cannot continue\n')
    break

  predicted_next_step = top_predictions[predictions_index]

  print('\n  Predicted step:', predicted_next_step)
  print('  Top predictions:', top_predictions)

  if predicted_next_step == '#':
    print('\n  Last step predicted\n')
    if predict_cube != SOLVED_CUBE:
      predictions_index = predictions_index + 1
      print('\n  Cube not solved, step back\n')
      continue

  if predict_cube == SOLVED_CUBE:
    predicted_solution.append(predicted_next_step)
    print('\n  Cube is solved, stop predicting\n')
    break

  predict_cube.perform_step(predicted_next_step)
  new_cube_array = cubeAsArray(predict_cube)

  step_back = False
  for carray in predicted_cube_arrays:
    if carray == new_cube_array:
      predict_cube = pc.Cube(pc.array_to_cubies(predict_cube_array))
      predictions_index = predictions_index + 1
      step_back = True
      print('\n  Loop detected, step back\n')
      break

  if not step_back:
    predictions_index = 0
    predict_cube_array = new_cube_array
    predicted_solution.append(predicted_next_step)
    predicted_cube_arrays.append(predict_cube_array)

print('\n  Finished cube:\n')
print(repr(predict_cube))

# This could happen when cube colors are mismatched,
# which PyCuber verification wont be able to catch.
if predict_cube != SOLVED_CUBE:
  sys.exit('cube failure')

print('\n  Predicted solution:', predicted_solution)
print('\n  Predicted solution steps:',
    len(predicted_solution[0:-1]), '\n')

# Clean print predicted solution so it can be parsed easily if needed.
# This should be the last line that goes to stdout, for scripted use.
print(' '.join(predicted_solution[0:-1]), file=sys.stdout, flush=True)

sys.exit(0)
