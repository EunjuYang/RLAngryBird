"""
    define RLAgent class
    Title : RLAgent.py
    Writer      : Eunjoo Yang, Wangyu Han
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
import time
from collections import deque


import sys, threading

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
episodes = 150
is_batch_norm = True  # batch normalization switch
TRAIN_STEP = 30
BATCH_SIZE = 20 #should be smaller than TRAIN_STEP

ALL_WIN_LEVEL = 20 #Game is over when all win from level 1 to level of this value
TARGET_SCORE = 900000
TEST_FLAG = True
START_WITH_TEST = False

SMALL_NOISE_LEVEL = [1,2,3,4,6,7,8,11,12,13,14,16,17,19,20]
#TRAIN_LEVEL = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
TRAIN_LEVEL = [18,9,10,11,12,13,14,15,16,17,18,19,20,21]

class RLAgent:
    def __init__(self,ip = "127.0.0.1",id = 28888):
        """
        constructor for RLAgent class
        :param ip: default = 127.0.0.1
        """
        if is_batch_norm:
            f = open("train_loss_bn.txt", 'a')
        else:
            f = open("train_loss.txt", 'a')
        data = "Batch size : %d "% BATCH_SIZE
        f.write(data)
        f.close()

        self.ar = ClientActionRobotJava(ip)
        self.tp = TrajectoryPlanner() # generate instance forne Trajectory Planner
        self.firstShot = True
        self.solved = []
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id
        self.width = 840
        self.height = 480
        # raw image size(states) = [width, height, 3 (channel)]
        self.num_states = 402
        #action space : [distance(0~45)pixel, angle(0~90), taptime(0~5000)ms]
        self.num_actions = 3
        self.action_space_high=[45, 45, 1500]
        self.action_space_low = [40, 10, 1000]
        self.noise_mean = [0, 0, 0]
        self.noise_sigma = [2, 10, 200]
        self.noise_sigma_small = [1, 1, 0]
        self.ddpg = DDPG(self.num_states, self.num_actions,self.action_space_high, self.action_space_low, is_batch_norm, BATCH_SIZE)
        self.final_observation = 0
        self.Final_observation_error = False
        self.temp_replay_memory = deque()

        self.total_numbirds = [3,5,4,4,4, 4,4,4,4,5, 4,4,4,4,4, 5,3,5,4,5, 8]
        self.steps = 0
        self.test_flag = START_WITH_TEST
        self.all_win_flag = True
        self.test_totalscore = 0.0

        self.TEST_START_FLAG = False
        t1 = threading.Thread(target=self.CLI)
        t1.start()
        self.TEST_START_LEVEL = 1
        self.TRAIN_START_FLAG = False
        self.TRAIN_START_LEVEL = TRAIN_LEVEL[0]
        self.TEST_AFTER_EVERY_EPISODE = False
        self.TEMP_TEST_AFTER_EVERY_EPISODE = False


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
        state = self.ar.checkState()

        while sling == None and state == state.PLAYING:
            print "no slingshot detected Please remove pop up or zoom out"
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

        pigs = vision.findPigsMBR()
        state = self.ar.checkState()

        # if there is a sling, then play, otherwise skip
        if sling != None:
            refPoint = self.tp.getReferencePoint(sling)

            #get shoot information from ddpg , tapTime is millsecond
            releaseDistance= action[0] # r
            releaseAngle = action[1] * 100 # theta
            tapTime = action[2] # tap time

            #Point releasePoint
            print "## Release Distance", (releaseDistance)
            print "## Release Angle", (releaseAngle)/100


            self.ar.fullyZoomOut()
            prev_screenshot = self.ar.doScreenShot()
            prev_vision = Vision(prev_screenshot)
            _sling = prev_vision.findSlingshotMBR()

            if self.currentLevel < 10:
                Prev_num_birds = (prev_vision.findBirdsMBR()).size()

                if Prev_num_birds == 0:
                    time.sleep(4)
                    state = self.ar.checkState()
                    return state;

            if _sling != None:
                scale_diff = (sling.width - _sling.width) ** 2 + (sling.height - _sling.height) ** 2
                if scale_diff < 25:
                    ###temporary
                    if self.test_flag:
                        self.ar.shoot(int(refPoint.x), int(refPoint.y), int(releaseDistance), int(releaseAngle), 0, int(tapTime), True)
                        time.sleep(0.5)
                        state = self.ar.checkState()
                    elif self.currentLevel > 9:
                        self.ar.fastshoot(int(refPoint.x), int(refPoint.y),
                                          -int(releaseDistance * cos(radians(releaseAngle / 100))),
                                          int(releaseDistance * sin(radians(releaseAngle / 100))), 0, int(tapTime),
                                          False)
                        shoot_time = time.time()
                        temp_observation = 0
                        time.sleep(3)
                        while time.time()-shoot_time < 15 :
                            screenshot = self.ar.doScreenShot()
                            state = self.ar.checkState()
                            if state == state.WON or state == state.LOST:
                                break
                            vision = Vision(screenshot)
                            temp_observation = self.FeatureExtractor(vision)

                        if state == state.WON or state == state.LOST:
                            self.final_observation = temp_observation
                            self.Final_observation_error = False
                            time.sleep(2.5)  # wait for score is ready
                    else:
                        shoot_time = time.time()
                        self.ar.fastshoot(int(refPoint.x), int(refPoint.y), -int(releaseDistance*cos(radians(releaseAngle/100))),int(releaseDistance*sin(radians(releaseAngle/100))), 0, int(tapTime), False)
                        time.sleep(1)  # wait for score is ready
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()
                        vision = Vision(screenshot)
                        Curr_num_birds = (vision.findBirdsMBR()).size()
                        Curr_num_pigs = (vision.findPigsMBR()).size()

                        final_observation_flag = False
                        while Curr_num_birds != Prev_num_birds-1:
                            state = self.ar.checkState()
                            if state == state.WON or state == state.LOST:
                                print "break!"
                                break
                            self.ar.fullyZoomOut()
                            screenshot = self.ar.doScreenShot()
                            vision = Vision(screenshot)
                            Curr_num_birds = (vision.findBirdsMBR()).size()
                            Curr_num_pigs = (vision.findPigsMBR()).size()
                            #time out
                            if time.time()-shoot_time > 30:
                                break
                            #Update final_observation only if there is no pig, and bird number is decreased.
                            if Curr_num_pigs == 0 or Curr_num_birds==0:
                                if Curr_num_birds == Prev_num_birds-1:
                                    print "Real Final observation made"
                                    self.final_observation = self.FeatureExtractor(vision)
                                    self.final_observation[400] = Prev_num_birds-1
                                    final_observation_flag = True
                                    self.Final_observation_error = False
                                if Curr_num_birds == Prev_num_birds and final_observation_flag == False:
                                    print "Temp Final observation made"
                                    self.final_observation = self.FeatureExtractor(vision)
                                    self.final_observation[400] = Prev_num_birds-1
                                    final_observation_flag = True
                                    self.Final_observation_error = False

                        if Curr_num_pigs == 0 or Curr_num_birds == 0:
                            if final_observation_flag == False:
                                print "Final observation Capture error. Don't save this experience"
                                self.Final_observation_error = True
                            while state != state.WON and state != state.LOST:
                                state = self.ar.checkState()
                                if time.time() - shoot_time > 30:
                                    break
                            time.sleep(2.5)#wait for score is ready

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
        #self.checkMyScore()
        TRAIN_LEVEL_index = 0
        if TEST_FLAG and START_WITH_TEST:
            self.currentLevel = 1
        else:
            self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]

        #ddpg
        # Randomly initialize critic,actor,target critic, target actor network  and replay buffer
        exploration_noise = OUNoise(self.num_actions, self.noise_mean, self.noise_sigma)#mean, sigma
        exploration_noise_small = OUNoise(self.num_actions, self.noise_mean, self.noise_sigma_small)  # mean, sigma
        counter = 0
        reward_per_episode = 0
        total_reward = 0
        print "Number of States:", self.num_states
        print "Number of Actions:", self.num_actions

        #Learning for Same level
        for current_episode in xrange(episodes): #1 episode is for 1stage
            if counter > TRAIN_STEP:
                self.TEST_AFTER_EVERY_EPISODE = self.TEMP_TEST_AFTER_EVERY_EPISODE
            self.ar.loadLevel(self.currentLevel)
            print "==== Starting episode no:", current_episode, "====", "\n"
            #get the input(screenshot)
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()
            state = self.ar.checkState()
            while sling == None and state == state.PLAYING:
                print "no slingshot detected Please remove pop up or zoom out"
                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                sling = vision.findSlingshotMBR()

            self.steps = 0 ######IMPORTANT
            observation = self.FeatureExtractor(vision)

            prevscore = 0
            reward_per_episode = 0
            self.final_observation = 0
            self.Final_observation_error = True

            # One episode
            while True:
                if self.test_flag:
                    print "\n###############################################################"
                    print "############                 TEST                  ############"
                    print "###############################################################\n"
                else:
                    print "\n###############################################################"
                    print "############                 TRAIN                 ############"
                    print "###############################################################\n"
                print "## currentLevel %d\n" % self.currentLevel

                x = observation
                action = self.ddpg.evaluate_actor(np.reshape(x, [1, self.num_states]))
                print "## Action From Network!!"
                print action
                action = action[0]
                noise = exploration_noise.noise()
                for temp_level in SMALL_NOISE_LEVEL:
                    if self.currentLevel == temp_level:
                        noise = exploration_noise_small.noise()
                        break
                if self.test_flag == False:
                    action = action + noise  # Select action according to current policy and exploration noise
                print action
                action[0] = action[0] if action[0] > self.action_space_low[0] else 2*self.action_space_low[0]-action[0]
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                action[1] = action[1] if action[1] > self.action_space_low[1] else 2*self.action_space_low[1]-action[1]
                action[1] = action[1] if action[1] < self.action_space_high[1] else self.action_space_high[1]
                action[2] = action[2] if action[2] > self.action_space_low[2] else 2*self.action_space_low[2]-action[2]
                action[2] = action[2] if action[2] < self.action_space_high[2] else self.action_space_high[2]
                print "Action at step", self.steps, " :", action, "\n"

                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                prev_num_birds = (vision.findBirdsMBR()).size()

                state = self.shoot(action)

                if state == state.WON or state == state.LOST:
                    # episode ends
                    print "## End Game"
                    ####
                    screenshot = self.ar.doScreenShot()

                    if state == state.WON:
                        score = self.ar.getScoreEndGame(screenshot)
                        reward = (score - prevscore) / 1000.0
                    else:
                        reward = -100.00

                    # TEST Result write
                    if self.test_flag and state == state.WON:
                        self.test_totalscore += score
                        f = open("Test_Result.txt", 'a')
                        data = "%d Win %d\n" % (self.currentLevel, score)
                        f.write(data)
                        f.close()


                    #Propagate WIN/LOSE reward to all steps
                    if self.steps != 0:
                        temp_reward = reward * 0.5
                        temp = self.temp_replay_memory.pop()
                        temp_x = temp[0]
                        temp_x = np.reshape(temp_x, [self.num_states])
                        temp_observation = temp[1]
                        temp_observation = np.reshape(temp_observation, [self.num_states])
                        temp_action = temp[2]
                        temp_action = np.reshape(temp_action, [self.num_actions])
                        while True:
                            try:
                                if self.test_flag == False:
                                    self.ddpg.add_experience(temp_x, temp_observation, temp_action, temp[3] + temp_reward, temp[4])
                                temp = self.temp_replay_memory.pop()
                                temp_x = temp[0]
                                temp_x = np.reshape(temp_x, [self.num_states])
                                temp_observation = temp[1]
                                temp_observation = np.reshape(temp_observation, [self.num_states])
                                temp_action = temp[2]
                                temp_action = np.reshape(temp_action, [self.num_actions])
                                temp_reward = temp_reward * 0.5
                            except:
                                break


                    # Move to next level at the end of episode
                    if self.test_flag:  # TEST
                        # In TEST, All win until LEVEL 'ALL_WIN_LEVEL' Then stop game
                        if self.currentLevel == ALL_WIN_LEVEL:
                            f = open("Test_Result.txt", 'a')
                            data = "All Win. Stop Game\nTotal Score : %d\n" % (self.test_totalscore)
                            f.write(data)
                            f.close()
                            self.ddpg.save_parameters()  # save parameters
                            return -1  # END GAME
                        self.currentLevel = self.getNextLevel()
                    else:  # Training
                        if TRAIN_LEVEL_index == len(TRAIN_LEVEL) - 1 or self.TEST_START_FLAG or self.TEST_AFTER_EVERY_EPISODE:  # Last Level
                            if TEST_FLAG:  # TEST START!
                                if self.TEST_START_FLAG:
                                    self.currentLevel = self.TEST_START_LEVEL
                                else:
                                    self.currentLevel = 1
                                self.TEST_START_FLAG = False
                                self.ddpg.save_parameters()  # save parameters
                                self.ddpg.close_sessions()
                                self.ddpg.restore_parameters()  # restore parameters
                                self.test_flag = True
                                self.all_win_flag = True
                                self.test_totalscore = 0.0
                                f = open("Test_Result.txt", 'a')
                                data = "Test at episode: %d \n" % current_episode
                                f.write(data)
                                f.close()
                            else:  # NO TEST, START FROM FIRST TRAINING LEVEL
                                self.currentLevel = TRAIN_LEVEL[0]
                        else:
                            TRAIN_LEVEL_index += 1
                            self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]



                    # TEST, When lose stop test
                    if self.test_flag and (self.currentLevel != 1 and self.currentLevel != self.TEST_START_LEVEL):

                        if state == state.LOST or self.TRAIN_START_FLAG:
                            f = open("Test_Result.txt", 'a')
                            if self.TEST_START_FLAG:
                                f.write("Stop By User. ")
                            data = "Total Score : %d\n" % (self.test_totalscore)
                            f.write(data)

                            f.close()
                            if TARGET_SCORE < self.test_totalscore:
                                f = open("Test_Result.txt", 'a')
                                data = "Exceed Target Score. Stop Game\nTotal Score : %d\n" % (self.test_totalscore)
                                f.write(data)
                                f.close()
                                self.ddpg.save_parameters()  # save parameters
                                return -1  # END GAME

                            self.test_flag = False

                            if self.TEST_AFTER_EVERY_EPISODE:
                                TRAIN_LEVEL_index += 1
                                if TRAIN_LEVEL_index == len(TRAIN_LEVEL):
                                    TRAIN_LEVEL_index = 0
                            else:
                                TRAIN_LEVEL_index = 0
                            if self.TEST_START_FLAG or self.TEST_AFTER_EVERY_EPISODE: ###TEMP
                                self.currentLevel = self.TRAIN_START_LEVEL
                            else:
                                self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]  # TRAINING START!
                            self.TRAIN_START_FLAG = False


                    self.firstShot = True
                    done = True
                    # add s_t,s_t+1,action,reward to experience memory
                    print "######### SCORE" , score
                    print "######### REWARD" , reward
                    if self.Final_observation_error == False:
                        print "### add experience", action, reward, done
                        if self.test_flag == False:
                            self.ddpg.add_experience(x, observation, action, reward, done)
                    elif self.test_flag:
                        print ""
                    else :
                        print "### Not add experience by Capture ERROR"
                    # train critic and actor network
                    if counter > TRAIN_STEP:
                        if self.test_flag == False:
                            self.ddpg.train()
                    reward_per_episode += reward
                    if self.test_flag == False:
                        counter += 1
                    self.steps += 1

                    # check if episode ends:
                    print 'EPISODE: ', current_episode, ' Steps: ', self.steps, ' Total Reward: ', reward_per_episode
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    exploration_noise_small.reset()
                    print '\n\n'

                    #RL_Bird:reward should be updated to all actions in the stage.
                    break

                elif state == state.PLAYING:
                    # PLAING in a episode
                    # get the input(screenshot)
                    self.ar.fullyZoomOut()
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    sling = vision.findSlingshotMBR()
                    state = self.ar.checkState()
                    while sling == None and state == state.PLAYING:
                        print "no slingshot detected Please remove pop up or zoom out"
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()
                        vision = Vision(screenshot)
                        sling = vision.findSlingshotMBR()


                    self.steps += 1
                    # get s_t+1
                    vision = Vision(screenshot)
                    print "#Next state is captured"
                    observation = self.FeatureExtractor(vision)
                    observation[400] = prev_num_birds-1
                    # add s_t,s_t+1,action,reward to experience memory
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    if reward == 0.0:
                        reward = -100



                    print "#### instant reward", reward
                    prevscore = score
                    done = False

                    self.temp_replay_memory.append((x, observation, action, reward, done))
                    if self.test_flag == False:
                        print "### add experience", action, reward, done

                    # train critic and actor network
                    if counter > TRAIN_STEP:
                        if self.test_flag == False:
                            self.ddpg.train()
                    reward_per_episode += reward
                    if self.test_flag == False:
                        counter += 1
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    exploration_noise_small.reset()


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
        #1.(1)1 kinds of Grid(400 dimension (in Grid if there exist, Nothing : 0, only Pigs : 10, only Obstacles : -10, Pigs and obstacles : 5)
        # 2) number of remaining birds (1 dimension)
        #Divide by 20X20 tiles . width(440~840), height(240~640). index = (width-440)/20 + ((height-240)/20) *20
        observation = np.zeros((self.num_states) , dtype=np.float32)

        pigs = vision.findPigsMBR()
        for i in xrange(pigs.size()):
            temp_object = pigs.get(i)
            #Calculate center position of object
            center_x = temp_object.x + temp_object.width/2
            center_y = temp_object.y + temp_object.height / 2

            #print "Pig Position : x ", center_x, ", y ", center_y
            if center_x-440 > 0 and center_y-240 > 0 :
                observation[(int)((center_y-440)/20)*20+ (int)((center_x-240)/20)] = 10
        #print "number of remaining pigs : ", pigs.size()
        blocks = vision.findBlocksMBR()
        for i in xrange(blocks.size()):
            temp_object = blocks.get(i)
            # Calculate center position of object
            center_x = temp_object.x + temp_object.width / 2
            center_y = temp_object.y + temp_object.height / 2

            #print "Obstacle Position : x ", center_x, ", y ", center_y
            if center_x - 440 > 0 and center_y - 240 > 0:
                if observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] == 0: #no pig
                    observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] = -10
                else: # pig exist
                    observation[(int)((center_y-440) / 20) * 20 + (int)((center_x-240) / 20)] = 5

        #number_birds = (vision.findBirdsMBR()).size()
        #print "number of remaining birds : ", number_birds
        #observation[400]=number_birds
        observation[400]=self.total_numbirds[self.currentLevel-1] - self.steps

        observation[401]=self.currentLevel
        # print "Current level : ", self.currentLevel

        return observation

    def CLI(self):
        while 1:
            line = sys.stdin.readline()
            try:
                if line.split(' ')[0] == "set" and line.split(' ')[1] == "test":
                    setlevel = int(line.split(' ')[2])
                    print "Set test start level to %d" %(setlevel)
                    self.TEST_START_LEVEL = setlevel

                if line.split(' ')[0] == "set" and line.split(' ')[1] == "train":
                    setlevel = int(line.split(' ')[2])
                    print "Set train start level to %d" %(TRAIN_LEVEL[setlevel])
                    self.TRAIN_START_LEVEL = TRAIN_LEVEL[setlevel]

                if line == "test\n":
                    self.TEST_START_FLAG = True
                    print "Test start flag is set"
                if line == "train\n":
                    self.TEST_START_FLAG = False
                    self.TRAIN_START_FLAG = True
                    print "Train start flag is set"
                if line == "TEST_EVERY\n":
                    print "Test start after training one episode"
                    self.TEMP_TEST_AFTER_EVERY_EPISODE = True
                if line == "TEST_CYCLE\n":
                    print "Test start after training cycle"
                    self.TEMP_TEST_AFTER_EVERY_EPISODE = False
            except:
                print "ERROR!"



if __name__ == '__main__':

    Bird = RLAgent()
    Bird.ddpg_run()
    shutdownJVM()