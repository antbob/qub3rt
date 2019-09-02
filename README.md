qub3rt
======

qub3rt is TensorFlow model that can solve Rubik's cube. It can be used either
standalone to solve virtual cube or real cube when connected to Lego EV3
MindCub3r robot.

TL;DR
=====

Just try running model.py and go from there. If you are messing with this
then chances are you have most if not all of dependencies already in place.

Standalone use
==============

Dependencies
------------

- tensorflow 2.0.0b1
- numpy 1.16.4
- pandas 0.24.2
- pycuber devel

later versions are most likely to work of course but they havent been tested.
(Note that this comment would generally apply to other dependencies from here)

tensorflow, numpy and pandas can be simply installed with pip as usual.

pycuber however has to be cloned from 'devel' branch at
https://github.com/adrianliaw/PyCuber/tree/devel
and the simplest way would be to just clone it into the same directory where
qub3rt lives and update your PYTHONPATH environment variable with its location.

Dataset
-------

The model comes with checkpoint files containing model weights ie it comes
pre-trained. This is just to save time and energy however should you wish
to train the model from scratch the dataset can be found separately at
https://www.kaggle.com/antbob/rubiks-cube-cfop-solutions
Furthermore the original scripts to generate the training set can be found
in this repository so a different dataset to train on can be generated eg
a larger dataset or perhaps a dataset with a different solving method or
even mixture of solving methods would be interesting to try. Either way
multiple instances of provided scripts can run in parallel for speed.

The existing dataset has 100000 cube solving sequences to train on and on an
average PC without any GPU accelerators would take just a few hours to train
from scratch.

The training set has been generated using PyCuber which implements CFOP
solving method. There are better solving algorithms that can yield more
efficient solutions eg 20 steps max however they are more CPU intensive
and would take much longer to generate a training set of that size plus
CFOP method is what a human would normally use to solve a cube.

Running
-------

```
$ python3 model.py --help
usage: model.py [-h] [-c CUBE]

optional arguments:
  -h, --help            show this help message and exit
  -c CUBE, --cube CUBE  Cube to predict a solution for, should be a numerical
                        sequence of cube colors with sides as LUFDRB/RYGWOB
                        000000000111111111222222222333333333444444444555555555
$
```

Invoking the model with no cube will make it generate a randomly scrambled one.
Invoking for specific cube doublecheck cube sides order and colors are correct.

On average it predicts about 70 something steps solutions which is similar to
PyCuber method from the training set however sometimes it would trip and yield
100 plus something step solutions for some scrambles.

MindCub3r robot
===============

The model can be used with MindCub3r robot to solve real cubes. Schematics for
MindCub3r robot can be found at
https://mindcuber.com/mindcub3r/mindcub3r.html
Either home or educational set should do, whatever you have. Note that either
infrared or ultrasonic sensors can be used, the software should work for both
but if you dont have either strictly its not required and with little effort
you can hack ev3dev_examples (see below) to workaround.

The robot requires some patience on your side. It is flaky which is not that
surprising given its made of Lego parts not precision machined components.
It often has problems scanning and detecting cube colors correctly and every
now and then its mechanical motor arm would fail to flip a stuck cube meaning
you have to interrupt it and start over.

To avoid alot of these problems I recommend you use a decent cube. Those made
by GAN (like Rubik's official speedcube) are pretty good. They rarely get
stuck and their colors seem to be at good enough color distance for basic Lego
color sensor to pick up correctly most of the time.

EV3 Dependencies
----------------

- ev3dev-stretch
- ev3dev_examples
- rpyc 4.1.1

ev3dev-stretch can be found at
https://www.ev3dev.org/downloads/
along with installation and setup instructions. You gonna need to setup
networking as well. Wi-Fi dongle compatible with ev3dev is ideal but you
can also set it up via Bluetooth or tethered via USB. RTFM at ev3dev.org

The original ev3dev_examples currently does not work at all so it has to be
cloned off my private 'tf-rpyc' branch at
https://github.com/antbob/ev3dev_examples/tree/tf-rpyc
which has all the changes required to make this work along with RPyC support.
Note that ev3dev_examples requires numpy and colormath which can be simply
installed with pip on ev3 brick. See ev3dev_examples README for details.

rpyc can be simply installed with pip on ev3 brick.

PC Dependencies
---------------

In addition to installing all dependencies to get a standalone model going
you would also need to install the following on PC side for robot control

- python 2.7.16
- rpyc 4.1.1

Python 2 is required because that is what the ev3 brick runs and RPyC does
not seem to allow mixing major Python versions when running RPyC service so
install Python 2 however you want and then use pip2 to simply install rpyc.

EV3 Running
-----------

Open shell to ev3 brick and

```
$ cd ev3dev_examples/python/
$ python rubiks_rpyc.py
```

This will start RPyC service the model can connect to and use. Its awfully
quiet so given that ev3 brick has worse hardware than your mobile phone
allow it a few moments to start before trying to connect from PC side.

2 different log files can be found in the same directory as the script

- rubiks_rpyc.py.err.log
- rubiks.log

The 1st one can be used to monitor RPyC service as its name suggest and
the 2nd one can be used to monitor cube scanning and executing solution.

PC Running
----------

Once RPyC service is up and running on ev3 brick then invoke on PC side

```
python2 qub3rt.py ev3_brick_ip_or_hostname
```

This one is chatty enough so you can monitor progress. Note that color
matching on ev3 brick is rather slow so be patient and remember that
you can monitor progress on the brick side via aforementioned log files.

Well, congratulations on making it this far and hope you got stocked
when you finally made this thing work just as I did :)
Peace out!
