#!/usr/bin/env python

from argparse import ArgumentParser
import numpy as np
import rospy
from hyper_msgs.msg import hyper_image
from scipy.io import savemat
from threading import Thread


def process_msg(filename, msg):
        cube = np.array(msg.data).reshape((msg.rows, msg.cols, msg.bands))
        savemat(filename, {'data': cube})
        print('Saved: "{}"'.format(filename))


class ImageExtractor:
    def __init__(self, topic, output, offset):
        self.image_sub = rospy.Subscriber(topic, hyper_image, self.callback, queue_size=200)
        self.output = output
        self.counter = offset

    def callback(self, msg):
        filename = '{}_{:05}.mat'.format(self.output, self.counter)
        self.counter += 1
        Thread(target=process_msg, args=(filename, msg)).start()


def main():
    try:
        command_help = 'This node extracts all hyper images from a given topic.'
        parser = ArgumentParser(description=command_help)
        parser.add_argument('-i', '--input', help='input topic [default=\'.\']', default='/vis_cam/spectral_data/corrected')
        parser.add_argument('-o', '--output', help='output path [default=\'./img\']', default='./img')
        parser.add_argument('-c', '--offset', help='delta offset for output names [default=0]', default=0)

        args = vars(parser.parse_args())
        topic = args['input']
        output = args['output']
        offset = int(args['offset'])

        print('input topic:    {}\noutput path:    {}\ncounter offset: {}'.format(topic, output, offset))

        ic = ImageExtractor(topic, output, offset)
        rospy.init_node('image_extractor', anonymous=True)

        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
