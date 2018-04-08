#!/usr/bin/env python

from os import walk
from os.path import join
from sys import argv
from subprocess import Popen, PIPE
from tqdm import tqdm

path_in = argv[1]
postfix = '.bag'
command = 'rosbag play -q -r 0.5 --clock "{}"'

try:
    filenames = sorted([join(r, n) for r, _, fs in walk(path_in) for n in fs if n.endswith(postfix)])
    print('Found {} logfiles.\n'.format(len(filenames)))

    for f in tqdm(filenames):
            Popen([command.format(f)], shell=True, stdout=PIPE).wait()
except KeyboardInterrupt:
    print('Shutting down!')
