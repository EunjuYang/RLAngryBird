"""
    Writer : Eunjoo Yang
    E-mail : yejyang@kaist.ac.kr

    This file is for agent training
    reference from ddpg-aigym git https://github.com/stevenpjg/ddpg-aigym.git
"""
import numpy as np
from ddpg import DDPG

episodes = 10000
is_batch_norm = False # batch normalization switch

def main():

    for i in xrange(episodes):
        print "=== Starting episode no:",i,"===\n"


if __name__ == '__main__':
    main()
