"""
    define RLAgent class
    Title : RLAgent.py
    Writer      : Eunjoo Yang
    Last Date   : 2017/05/15
"""
from jpype import *
from math import *
import os.path
import numpy as np
import random
import tensorflow as tf
from ddpg import DDPG
from ou_noise import OUNoise

# JPype Setting
jarpath = os.path.join(os.path.abspath('../../'))
startJVM(getDefaultJVMPath(), "-Djava.ext.dirs=%s" % jarpath)
TrajectoryPkg = JPackage('ab').planner
DemoOtherPkg = JPackage('ab').demo.other
VisionPkg = JPackage('ab').vision
AwtPkg = JPackage('java').awt
UtilPkg = JPackage('java').util

# Load Package
# java util
Point = AwtPkg.Point
Rectangle = AwtPkg.Rectangle
BufferedImage = AwtPkg.image.BufferedImage
ArrayList = UtilPkg.ArrayList
List = UtilPkg.List
# AI Bird
TrajectoryPlanner = TrajectoryPkg.TrajectoryPlanner # Class
ClientActionRobot = DemoOtherPkg.ClientActionRobot # Class
ClientActionRobotJava = DemoOtherPkg.ClientActionRobotJava # Class
ABObject = VisionPkg.ABObject # Class
GameState = VisionPkg.GameStateExtractor # Class
Vision = VisionPkg.Vision # Class

# specify parameters here:
episodes = 10000
is_batch_norm = False  # batch normalization switch
TRAIN_STEP = 30

class RLAgent:


    def __init__(self,ip = "127.0.0.1",id = 28888):
        """
        constructor for RLAgent class
        :param ip: default = 127.0.0.1
        """
        self.ar = ClientActionRobotJava(ip)
        self.tp = TrajectoryPlanner() # generate instance for Trajectory Planner
        self.firstShot = True
        self.solved = []
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id
        self.width = 840
        self.height = 480
        # raw image size(states) = [width, height, 3 (channel)]
        self.num_states = [self.height, self.width, 3]
        #action space : [distance(0~)pixel, angle(0~90), taptime(0~5000)ms]
        self.num_actions = 3
        self.action_space_high=[90, 75, 50]
        self.action_space_low = [0, 0, 0]
        self.noise_mean = [20, -20, 0]
        self.noise_sigma = [10, 30, 20]
        self.ddpg = DDPG(self.num_states, self.num_actions,self.action_space_high, self.action_space_low, is_batch_norm)

    def getNextLevel(self):
        """
        get Next Level
        :return: level
        """

        level = 0
        unsolved = False

        for i in range(len(self.solved)):

            if self.solved[i] == 0:
                unsolved = True
                level = i + 1
                if level <= self.currentLevel and self.currentLevel < len(self.solved):
                    continue
                else:
                    return level
        if unsolved:
            return level
        level = (self.currentLevel + 1)% len(self.solved)
        if level == 0:
            level = len(self.solved)

        return level


    def checkMyScore(self):
        scores = self.ar.checkMyScore()
        level = 1
        for i in scores:
            print " level %d %d"%(level, i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1


    def getScreenBuffer(self,buffer,width = 840, height = 480):
        """
        getScreenBuffer from screen image buffer
        :param width:
        :param height:
        :param buffer:
        :return:
        """
        print "## getScreenBuffer"
        returnBuffer = np.zeros((height,width,3))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j,i)
                returnBuffer[i,j,0] = RGB & 0x0000ff
                returnBuffer[i,j,1] = RGB & 0x00ff00
                returnBuffer[i,j,2] = RGB & 0xff0000

        print "## returnScreenBuffer"
        return returnBuffer

    def shoot(self, action):
        """
        shoot by given action
        :return: GameState
        """

        screenshot = self.ar.doScreenShot()
        vision = Vision(screenshot)
        sling = vision.findSlingshotMBR()

        state = self.ar.checkState()
        while sling == None and self.ar.checkState() == state.PLAYING:
            print "no slingshot detected Please remove pop up or zoom out"
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

        pigs = vision.findPigsMBR()
        state = self.ar.checkState()

        # if there is a sling, then play, otherwise skip
        if sling != None:

            # If there are pigs we pick up a pig randomly and shoot it
            if len(pigs) != 0:

                refPoint = self.tp.getReferencePoint(sling)
                print "## Ref Point"
                print refPoint

                #get shoot information from ddpg , tapTime is millsecond
                releaseDistance= action[0] # r
                releaseAngle = action[1] * 100 # theta
                tapTime = action[2] # tap time

                #Point releasePoint
                print "## Release Distance", (releaseDistance)
                print "## Release Angle", (releaseAngle)/100

                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                _sling = vision.findSlingshotMBR()

                if _sling != None:

                    scale_diff = (sling.width - _sling.width) ** 2 + (sling.height - _sling.height) ** 2

                    if scale_diff < 25:
                        self.ar.shoot(int(refPoint.x), int(refPoint.y), int(releaseDistance), int(releaseAngle), 0, int(tapTime), True)
                        print "## shoot!"
                        state = self.ar.checkState()

                        if state == state.PLAYING:
                            self.firstShot = False
                    else:
                        print "Sclae is changed, can not execute the shot, will re-segment the image"
                else:
                    print "no sling detected, cannot execute the shot, will re-segment the image"

        return state

    def ddpg_run(self):
        """
            run ddpg algorithm with raw pixel status
            :return:
        """

        info = self.ar.configure(ClientActionRobot.intToByteArray(self.id))
        self.solved = np.zeros(info[2])
        self.checkMyScore()
        print "## currentLevel %d" % self.currentLevel

        #ddpg
        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        exploration_noise = OUNoise(self.num_actions, self.noise_mean, self.noise_sigma)#mean, sigma
        counter = 1
        reward_per_episode = 0
        total_reward = 0
        print "Number of States:", self.num_states
        print "Number of Actions:", self.num_actions
        # saving reward:
        reward_st = np.array([0])

        #Learning for Same level
        for i in xrange(episodes): #1 episode is for 1stage

            self.currentLevel = self.getNextLevel()
            if self.currentLevel < 4:
                self.ar.loadLevel(self.currentLevel)
            else:
                self.currentLevel = 1
                self.ar.loadLevel(self.currentLevel)

            prevscore = 0
            reward_per_episode = 0
            steps = 0
            print "==== Starting episode no:", i, "====", "\n"

            # One episode
            while True:

                #get the input(screenshot)
                screenshot = self.ar.doScreenShot()
                x = self.getScreenBuffer(screenshot,self.width, self.height)
                action = self.ddpg.evaluate_actor(np.reshape(x, [1, self.num_states[0], self.num_states[1], self.num_states[2]]))
                print "## Action From Network!!"
                print action
                action = action[0]
                noise = exploration_noise.noise()
                action = action + noise  # Select action according to current policy and exploration noise
                print action
                action[0] = action[0] if action[0] > self.action_space_low[0] else -action[0]
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                action[1] = action[1] if action[1] > self.action_space_low[1] else -action[1]
                action[1] = action[1] if action[1] < self.action_space_high[1] else self.action_space_high[1]
                action[2] = action[2] if action[2] > self.action_space_low[2] else -action[2]
                action[2] = action[2] if action[2] < self.action_space_high[2] else self.action_space_high[2]
                print "## Action at step", steps, " :", action, "\n"
                state = self.shoot(action)

                if state == state.WON or state == state.LOST:
                    # episode ends
                    print "## End Game"

                    screenshot = self.ar.doScreenShot()
                    observation = self.getScreenBuffer(screenshot,self.width, self.height)

                    if state == state.WON:
                        score = self.ar.getScoreEndGame(screenshot)
                        reward = (score - prevscore) /1000.0
                    else:
                        reward = 0.00

                    self.currentLevel = self.currentLevel # self.currentLevel = self.getNextLevle()
                    self.firstShot = True
                    done = True
                    # add s_t,s_t+1,action,reward to experience memory
                    print "######### SCORE" , score
                    print "######### REWARD" , reward
                    print "### add experience", action, reward, done
                    self.ddpg.add_experience(x, observation, action, reward, done)
                    # train critic and actor network
                    if counter > TRAIN_STEP :
                        self.ddpg.train()
                    reward_per_episode += reward
                    counter += 1
                    steps += 1

                    # check if episode ends:
                    print 'EPISODE: ', i, ' Steps: ', steps, ' Total Reward: ', reward_per_episode
                    print "Printing reward to file"
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    reward_st = np.append(reward_st, reward_per_episode)
                    np.savetxt('episode_reward.txt', reward_st, newline="\n")
                    print '\n\n'

                    #RL_Bird:reward should be updated to all actions in the stage.
                    break;

                elif state == state.PLAYING:
                    # PLAING in a episode
                    # get the input(screenshot)
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    sling = vision.findSlingshotMBR()

                    while sling == None and self.ar.checkState() == state.PLAYING:
                        print "no slingshot detected Please remove pop up or zoom out"
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()

                    # get s_t+1
                    observation = self.getScreenBuffer(screenshot,self.width, self.height)
                    # add s_t,s_t+1,action,reward to experience memory
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    prevscore = score
                    done = False
                    self.ddpg.add_experience(x, observation, action, reward, done)
                    print "### add experience", action, reward, done
                    # train critic and actor network
                    if counter > TRAIN_STEP :
                        self.ddpg.train()
                    reward_per_episode += reward
                    counter += 1
                    steps += 1

                elif state == state.LEVEL_SELECTION:
                    print "unexpected level selection page, go to the last current level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.MAIN_MENU:
                    print"unexpected main menu page, reload the level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.EPISODE_MENU:
                    print "unexpected episode menu page, reload the level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)

                #########################

        total_reward += reward_per_episode
        print "Average reward per episode {}".format(total_reward / episodes)


if __name__ == '__main__':

    Bird = RLAgent()
    Bird.ddpg_run()
    shutdownJVM()