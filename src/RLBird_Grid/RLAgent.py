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
TRAIN_STEP = 5
episodes = 10000
is_batch_norm = True  # batch normalization switch
BATCH_SIZE = 5 #should be smaller than TRAIN_STEP

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
        self.num_states = 400
        #action space : [distance(0~300)pixel, angle(0~90), taptime(0~5000)ms]
        self.num_actions = 3
        self.action_space_high=[300, 90, 5000]
        self.action_space_low = [0, 0, 0]
        self.noise_mean = [100, 30, 1500]
        self.noise_sigma = [70, 20, 1000]
        self.ddpg = DDPG(self.num_states, self.num_actions,self.action_space_high, self.action_space_low, is_batch_norm, BATCH_SIZE)

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
        returnBuffer = np.zeros((3,height,width))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j,i)
                returnBuffer[0,i,j] = RGB & 0x0000ff
                returnBuffer[1,i,j] = RGB & 0x00ff00
                returnBuffer[2,i,j] = RGB & 0xff0000

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

        while sling == None and self.ar.checkState() == GameState.PLAYING:
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
                        state = self.ar.checkState()

                        if state == state.PLAYING:
                            #screenshot = self.ar.doScreenShot()
                            #vision = Vision(screenshot)
                            #traj = vision.findTrajPoints()
                            #self.tp.adjustTrajectory(traj, sling, releasePoint)
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
        self.currentLevel = self.getNextLevel()
        print "## currentLevel %d" % self.currentLevel


        #ddpg
        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        exploration_noise = OUNoise(self.num_actions, self.noise_mean, self.noise_sigma)#mean, sigma
        counter = 0
        reward_per_episode = 0
        total_reward = 0
        print "Number of States:", self.num_states
        print "Number of Actions:", self.num_actions
        # saving reward:
        reward_st = np.array([0])

        #Learning for Same level
        for i in xrange(episodes): #1 episode is for 1stage

            self.ar.loadLevel(self.currentLevel)
            print "==== Starting episode no:", i, "====", "\n"
            #get the input(screenshot)
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

            while sling == None and self.ar.checkState() == GameState.PLAYING:
                print "no slingshot detected Please remove pop up or zoom out"
                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            observation = self.FeatureExtractor(vision)

            prevscore = 0
            reward_per_episode = 0
            steps = 0

            # One episode
            while True:
                x = observation
                action = self.ddpg.evaluate_actor(np.reshape(x, [1, self.num_states]))
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
                print "Action at step", steps, " :", action, "\n"
                state = self.shoot(action)

                if state == state.WON or state == state.LOST:
                    # episode ends
                    print "## End Game"
                    ####
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    observation = self.FeatureExtractor(vision)

                    if state == state.WON:
                        score = self.ar.getScoreEndGame(screenshot)
                        reward = (score - prevscore) / 1000.0
                    else:
                        reward = 0.00


                    self.currentLevel = self.currentLevel # self.currentLevel = self.getNextLevel()
                    self.firstShot = True
                    done = True
                    # add s_t,s_t+1,action,reward to experience memory
                    print "######### SCORE" , score
                    print "######### REWARD" , reward
                    print "### add experience", action, reward, done
                    self.ddpg.add_experience(x, observation, action, reward, done)
                    # train critic and actor network
                    if counter > TRAIN_STEP:
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

                    while sling == None and self.ar.checkState() == GameState.PLAYING:
                        print "no slingshot detected Please remove pop up or zoom out"
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()

                    # get s_t+1
                    vision = Vision(screenshot)
                    observation = self.FeatureExtractor(vision)
                    # add s_t,s_t+1,action,reward to experience memory
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    print "#### instant reward", reward
                    prevscore = score
                    done = False
                    self.ddpg.add_experience(x, observation, action, reward, done)
                    print "### add experience", action, reward, done
                    # train critic and actor network
                    if counter > TRAIN_STEP:
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

    def FeatureExtractor(self, vision):
        """
        get Observation (State for DDPG) by given vision
        :return: observation
        """
        #1.(1 kinds of Grid (in Grid if there exist, Nothing : 0, only Pigs : 1, only Obstacles : 2, Pigs and obstacles : 3)
        #Divide by 20X20 tiles . width(440~840), height(240~640). index = (width-440)/20 + ((height-240)/20) *20
        observation = np.zeros((400) , dtype=np.float32)

        pigs = vision.findPigsMBR()
        for i in xrange(pigs.size()):
            temp_object = pigs.get(i)
            #Calculate center position of object
            center_x = temp_object.x + temp_object.width/2
            center_y = temp_object.y + temp_object.height / 2

            #print "Pig Position : x ", center_x, ", y ", center_y
            if center_x-440 > 0 and center_y-240 > 0 :
                observation[(int)((center_y-440)/20)*20+ (int)((center_x-240)/20)] = 1

        blocks = vision.findBlocksMBR()
        for i in xrange(blocks.size()):
            temp_object = blocks.get(i)
            # Calculate center position of object
            center_x = temp_object.x + temp_object.width / 2
            center_y = temp_object.y + temp_object.height / 2

            #print "Obstacle Position : x ", center_x, ", y ", center_y
            if center_x - 440 > 0 and center_y - 240 > 0:
                if observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] == 0: #no pig
                    observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] = 2
                else: # pig exist
                    observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] = 3

        return observation


if __name__ == '__main__':

    Bird = RLAgent()
    Bird.ddpg_run()
    shutdownJVM()