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
import time

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
TRAIN_STEP = 40
episodes = 1000
is_batch_norm = True  # batch normalization switch
BATCH_SIZE = 40 #should be smaller than TRAIN_STEP

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
        self.num_states = 802
        #action space : [distance(0~45)pixel, angle(0~90), taptime(0~5000)ms]
        self.num_actions = 3
        self.action_space_high=[45, 45, 2000]
        self.action_space_low = [40, 10, 500]
        #self.noise_mean = [100, 30, 1500]
        #self.noise_sigma = [70, 20, 1000]
        self.noise_mean = [0, 0, 0]
        self.noise_sigma = [2, 15, 0]
        self.ddpg = DDPG(self.num_states, self.num_actions,self.action_space_high, self.action_space_low, is_batch_norm, BATCH_SIZE)
        self.final_observation = 0
        self.Final_observation_error = False


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
                prev_screenshot = self.ar.doScreenShot()
                prev_vision = Vision(prev_screenshot)
                _sling = prev_vision.findSlingshotMBR()
                Prev_num_birds = (prev_vision.findBirdsMBR()).size()
                if _sling != None:
                    scale_diff = (sling.width - _sling.width) ** 2 + (sling.height - _sling.height) ** 2
                    if scale_diff < 25:
                        ###temporary
                        #self.ar.shoot(int(refPoint.x), int(refPoint.y), int(releaseDistance), int(releaseAngle), 0, int(tapTime), True)

                        shoot_time = time.time()
                        self.ar.fastshoot(int(refPoint.x), int(refPoint.y), -int(releaseDistance*cos(radians(releaseAngle/100))),int(releaseDistance*sin(radians(releaseAngle/100))), 0, int(tapTime), False)

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
                            if time.time()-shoot_time > 20:
                                break
                            #Update final_observation only if there is no pig, and bird number is decreased.
                            if Curr_num_pigs == 0 or Curr_num_birds==0:
                                if Curr_num_birds == Prev_num_birds-1:
                                    print "Real Final observation made"
                                    self.final_observation = self.getState(vision)
                                    self.final_observation[800] = Prev_num_birds-1
                                    final_observation_flag = True
                                    self.Final_observation_error = False
                                if Curr_num_birds == Prev_num_birds and final_observation_flag == False:
                                    print "Temp Final observation made"
                                    self.final_observation = self.getState(vision)
                                    self.final_observation[800] = Prev_num_birds-1
                                    final_observation_flag = True
                                    self.Final_observation_error = False

                        if Curr_num_pigs == 0 or Curr_num_birds == 0:
                            if final_observation_flag == False:
                                print "Final observation Capture error. Don't save this experience"
                                self.Final_observation_error = True
                            while state != state.WON and state != state.LOST:
                                state = self.ar.checkState()
                                if time.time() - shoot_time > 20:
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

        Stage_maximum_actions = 1

        #Learning for Same level
        for current_episode in xrange(episodes): #1 episode is for 1stage

            self.ar.loadLevel(self.currentLevel)
            print "==== Starting episode no:", current_episode, "====", "\n"
            #get the input(screenshot)
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

            observation = self.getState(vision)

            prevscore = 0
            reward_per_episode = 0
            steps = 0
            self.final_observation = 0
            self.Final_observation_error = True

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
                action[0] = action[0] if action[0] > self.action_space_low[0] else 2*self.action_space_low[0]-action[0]
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                action[1] = action[1] if action[1] > self.action_space_low[1] else 2*self.action_space_low[1]-action[1]
                action[1] = action[1] if action[1] < self.action_space_high[1] else self.action_space_high[1]
                action[2] = action[2] if action[2] > self.action_space_low[2] else 2*self.action_space_low[2]-action[2]
                action[2] = action[2] if action[2] < self.action_space_high[2] else self.action_space_high[2]
                print "Action at step", steps, " :", action, "\n"

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

                    #Move to next level in every 5 steps
                    if current_episode%5 == 4:
                        if self.currentLevel == 5:
                            self.currentLevel = 1
                        else:
                            self.currentLevel = self.getNextLevel()
                    else:
                        self.currentLevel = self.currentLevel # self.currentLevel = self.getNextLevel()

                    self.firstShot = True
                    done = True
                    # add s_t,s_t+1,action,reward to experience memory
                    print "######### SCORE" , score
                    print "######### REWARD" , reward
                    if self.Final_observation_error == False:
                        print "### add experience", action, reward, done
                        self.ddpg.add_experience(x, self.final_observation, action, reward, done)
                    else :
                        print "### Not add experience by Capture ERROR"
                    # train critic and actor network
                    if counter > TRAIN_STEP:
                        self.ddpg.train()
                    reward_per_episode += reward
                    counter += 1
                    steps += 1

                    # check if episode ends:
                    print 'EPISODE: ', current_episode, ' Steps: ', steps, ' Total Reward: ', reward_per_episode
                    #print "Printing reward to file"
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    #reward_st = np.append(reward_st, reward_per_episode)
                    #np.savetxt('episode_reward.txt', reward_st, newline="\n")
                    print '\n\n'

                    #RL_Bird:reward should be updated to all actions in the stage.
                    break

                elif state == state.PLAYING:
                    # PLAING in a episode
                    # get the input(screenshot)
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

                    # get s_t+1
                    vision = Vision(screenshot)
                    print "#Next state is captured"
                    observation = self.getState(vision)
                    observation[800] = prev_num_birds-1
                    # add s_t,s_t+1,action,reward to experience memory
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    if reward == 0.0:
                        reward = -100
                    #?????
                    #if (Stage_maximum_actions - 1 == steps):
                    #    reward = -10


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
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    #?????
                    #if (Stage_maximum_actions == steps):
                    #    print "### Count is over. restart game"
                    #    break;

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

    def getState(self, vision):
        """
        getState from vision
        :return:tate
        """
        #1.(1)1 kinds of Grid(400 dimension (in Grid if there exist, Nothing : 0, only Pigs : 10, only Obstacles : -10, Pigs and obstacles : 5)
        # 2) number of remaining birds (1 dimension)
        #Divide by 20X20 tiles . width(440~840), height(240~640). index = (width-440)/20 + ((height-240)/20) *20

        tile_x = 440
        tile_y = 240
        tile_size = 20

        state = np.zeros((self.num_states) , dtype=np.float32)

        pigs = vision.findPigsMBR()
        for i in xrange(pigs.size()):
            temp_object = pigs.get(i)
            #Calculate center position of object
            center_x = temp_object.x + temp_object.width/2
            center_y = temp_object.y + temp_object.height / 2

            #print "Pig Position : x ", center_x, ", y ", center_y
            idx = (int)((center_y - tile_y) / tile_size) * tile_size + (int)(center_x - tile_x) / tile_size
            if center_x-440 > 0 and center_y-240 > 0 :
                state[idx] = 5

        blocks = vision.findBlocksMBR()
        for i in xrange(blocks.size()):
            temp_object = blocks.get(i)
            # Calculate center position of object
            center_x = temp_object.x + temp_object.width / 2
            center_y = temp_object.y + temp_object.height / 2
            left_x = temp_object.x
            left_y = temp_object.y
            right_x = temp_object.x + temp_object.width
            right_y = temp_object.y + temp_object.height

            #print "Obstacle Position : x ", center_x, ", y ", center_y
            idx1 = 400 + (int)((center_y - tile_y) / tile_size) * tile_size + (int)(center_x - tile_x) / tile_size
            idx2 = 400 + (int)((left_y - tile_y) / tile_size) * tile_size + (int)(left_x - tile_x) / tile_size
            idx3 = 400 + (int)((left_y - tile_y) / tile_size) * tile_size + (int)(right_x - tile_x) / tile_size
            idx4 = 400 + (int)((right_y - tile_y) / tile_size) * tile_size + (int)(left_x - tile_x) / tile_size
            idx5 = 400 + (int)((right_y - tile_y) / tile_size) * tile_size + (int)(right_x - tile_x) / tile_size
            if center_x - 440 > 0 and center_y - 240 > 0:
                state[idx1] = +1
            if idx2 < 800 and idx2 >= 0:
                state[idx2] = +1
            if idx3 < 800 and idx3 >= 0:
                state[idx3] = +1
            if idx4 < 800 and idx4 >= 0:
                state[idx4] = +1
            if idx5 < 800 and idx5 >= 0:
                state[idx5] = +1

        number_birds = (vision.findBirdsMBR()).size()
        state[800] = number_birds
        current_level = self.getNextLevel()-1

        return state


if __name__ == '__main__':

    Bird = RLAgent()
    Bird.ddpg_run()
    shutdownJVM()
