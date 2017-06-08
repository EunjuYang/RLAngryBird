"""
    define RLAgent class for low dimension state
    Title : RLAgent.py
    Writer      : Eunjoo Yang
    Last Date   : 2017/06/03
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
TRAIN_STEP = 100
episodes = 1000
is_batch_norm = True  # batch normalization switch
BATCH_SIZE = 100 #should be smaller than TRAIN_STEP

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
        self.solved = []
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id
        self.width = 840
        self.height = 480
        self.BATCH_SIZE = BATCH_SIZE
        # pig state [0 ~ 399] obstacle state [0 ~ 400] + level + remained birds
        self.num_states = 400 + 400 + 1
        #action space : [distance(0~45)pixel, angle(0~90), taptime(0~5000)ms]
        self.num_actions = 3
        self.action_space_high=[45, 75, 2000]
        self.action_space_low = [40, 10, 500]
        self.noise_mean = [0, -20, 0]
        self.noise_sigma = [10, 30, 0]
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
        #print "## getScreenBuffer"
        returnBuffer = np.zeros((3,height,width))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j,i)
                returnBuffer[0,i,j] = RGB & 0x0000ff
                returnBuffer[1,i,j] = RGB & 0x00ff00
                returnBuffer[2,i,j] = RGB & 0xff0000
        #print "## returnScreenBuffer"
        return returnBuffer

    def getStateBuffer(self,buffer):

        x_offset = 440
        y_offset = 240
        height = 480 - y_offset
        width = 840 - x_offset

        returnBuffer = np.zeros((3, height, width))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j + x_offset, i + y_offset)
                returnBuffer[0,i,j] = RGB & 0x0000ff
                returnBuffer[1,i,j] = RGB & 0x00ff00
                returnBuffer[2,i,j] = RGB & 0xff0000
        return returnBuffer



    def shoot(self, action):
        """
        shoot by given action
        :return: GameState
        """

        self.ar.fullyZoomOut()
        screenshot = self.ar.doScreenShot()
        vision = Vision(screenshot)
        sling = vision.findSlingshotMBR()
        state = self.ar.checkState()

        while sling == None:
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

        pigs = vision.findPigsMBR()
        state = self.ar.checkState()
        refPoint = self.tp.getReferencePoint(sling)

        while refPoint == None:
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()
            refPoint = self.tp.getReferencePoint(sling)

        # If the episode doesn't end
        if sling != 0:

            refPoint = self.tp.getReferencePoint(sling)
            r = action[0]
            theta = action[1]
            tapTime = action[2]

            screen_beforeshoot = self.ar.doScreenShot()
            vision_beforeshoot = Vision(screen_beforeshoot)
            _NoBirds = (vision_beforeshoot.findBirdsMBR()).size()
            checkScreen = self.getStateBuffer(screen_beforeshoot)
            checkstate = self.getState(vision_beforeshoot)

            dx = -int(r * cos(radians(theta)))
            dy = int(r * sin(radians(theta)))

            # do fast shoot
            self.ar.fastshoot(int(refPoint.x), int(refPoint.y), dx, dy, 0, int(tapTime), False)

            shoot_time = time.time()
            time.sleep(1.3)#wait for score is ready

            screen_aftershoot = self.ar.doScreenShot()
            vision_aftershoot = Vision(screen_aftershoot)
            NoBirds = (vision_aftershoot.findBirdsMBR()).size()
            _checkScreen = self.getStateBuffer(screen_aftershoot)
            _checkstate = self.getState(vision_aftershoot)

            # Until End of One Step
            while True:

                time.sleep(5.7)#wait for score is ready
                #print NoBirds, _NoBirds
                state = self.ar.checkState()
                # End of episode
                if state  == state.WON or state == state.LOST:
                    print "End of Episode State is returned"
                    self.final_observation = self.getState(vision_aftershoot)
                    break

                screen_aftershoot = self.ar.doScreenShot()
                vision_aftershoot = Vision(screen_aftershoot)
                NoPigs = (vision_aftershoot.findPigsMBR()).size()
                if NoBirds < _NoBirds and ((np.array_equal(checkScreen, _checkScreen) == True) or (np.array_equal(checkstate, _checkstate) == True)) and ( NoPigs != 0):
                    self.final_observation = self.getState(vision_aftershoot)
                    break

                screen_aftershoot = self.ar.doScreenShot()
                vision_aftershoot = Vision(screen_aftershoot)
                NoBirds = (vision_aftershoot.findBirdsMBR()).size()
                checkScreen = _checkScreen
                checkstate = _checkstate
                _checkScreen = self.getStateBuffer(screen_aftershoot)
                _checkstate = self.getState(vision_aftershoot)
                if time.time()-shoot_time > 40:
                    print "Time out finish to shoot"
                    break

            #print NoBirds, _NoBirds
            state = self.ar.checkState()
            self.final_observation = self.getState(vision_aftershoot)

        time.sleep(8.0)#wait for score is ready
        #print NoBirds, _NoBirds
        state = self.ar.checkState()
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

            observation = self.getState(vision)

            prevscore = 0
            reward_per_episode = 0
            steps = 0
            self.final_observation = 0
            self.Final_observation_error = True

            # One episode
            while True:
                print "################################ Onestep ##############################"

                while sling == None and state == state.PLAYING:
                    print "no slingshot detected Please remove pop up or zoom out"
                    self.ar.fullyZoomOut()
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    sling = vision.findSlingshotMBR()

                x = self.getState(vision)
                action = (self.ddpg.evaluate_actor(np.reshape(x, [1, self.num_states])))[0]
                print "## Action From Network!!"
                print action
                print "## Action with Noise!!"
                noise = exploration_noise.noise()
                action = action + noise  # Select action according to current policy and exploration noise
                print action
                action[0] = action[0] if action[0] > self.action_space_low[0] else 2*self.action_space_low[0]-action[0]
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                action[1] = action[1] if action[1] > self.action_space_low[1] else 2*self.action_space_low[1]-action[1]
                action[1] = action[1] if action[1] < self.action_space_high[1] else self.action_space_high[1]
                action[2] = action[2] if action[2] > self.action_space_low[2] else 2*self.action_space_low[2]-action[2]
                action[2] = action[2] if action[2] < self.action_space_high[2] else self.action_space_high[2]
                print "## Action to shoot!!"
                print "Action at step", steps, " :", action, "\n"

                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                _NoBirds = (vision.findBirdsMBR()).size()

                afterstate = self.shoot(action)

                if afterstate == state.WON or afterstate == state.LOST:

                    # episode ends
                    print "## End Game"
                    time.sleep(1)#wait for score is ready
                    screenshot = self.ar.doScreenShot()
                    score = self.ar.getScoreEndGame(screenshot)
                    # Reward
                    if afterstate == state.WON:
                        reward = (score - prevscore) / 1000.0
                    elif afterstate == state.LOST:
                        reward = -10.00

                    #Move to next level in every 5 steps
                    if current_episode % 3 == 2:
                        self.currentLevel = self.currentLevel % 22 + 1
                    else:
                        self.currentLevel = self.currentLevel # self.currentLevel = self.getNextLevel()

                    done = True
                    print "######### SCORE" , score
                    print "######### PREVSCORE" , prevscore
                    print "######### REWARD" , reward
                    print "### add experience", action, reward, done
                    # add s_t,s_t+1,action,reward to experience memory
                    self.ddpg.add_experience(x, self.final_observation, action, reward, done)
                    reward_per_episode += reward
                    counter += 1
                    steps += 1
                    # check if episode ends:
                    print 'EPISODE: ', current_episode, ' Steps: ', steps, ' Total Reward: ', reward_per_episode
                    # train critic and actor network
                    if counter > TRAIN_STEP:
                        self.ddpg.train()
                        #self.BATCH_SIZE = self.BATCH_SIZE + 1
                        #self.ddpg.set_batchsize(self.BATCH_SIZE)
                    exploration_noise.reset()  # reinitializing random noise for action exploration
                    print '\n\n'
                    break

                elif afterstate == state.PLAYING:
                    # PLAING in a episode
                    # get the input(screenshot)
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    print "#Next state is captured"
                    observation = self.getState(vision)
                    # add s_t,s_t+1,action,reward to experience memory
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    reward = -10.00 if reward == 0 else reward
                    print "#### instant reward", reward
                    prevscore = score
                    done = False

                    self.ddpg.add_experience(x, observation, action, reward, done)
                    print "### add experience", action, reward, done
                    # train critic and actor network
                    if counter > TRAIN_STEP:
                        self.ddpg.train()
                        self.BATCH_SIZE = self.BATCH_SIZE + 1
                        self.ddpg.set_batchsize(self.BATCH_SIZE)
                    reward_per_episode += reward
                    counter += 1
                    steps += 1
                    exploration_noise.reset()  # reinitializing random noise for action exploration
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