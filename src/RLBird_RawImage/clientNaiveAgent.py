"""
    define NaiveAgent class
    Title : RLAgent.py
    Writer      : Eunjoo Yang
    Last Date   : 2017/05/15
"""
from jpype import *
from math import *
import os.path
import numpy as np
import random

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

Birds = {'RedBird':4, 'YellowBird':5, 'BlueBird':6, 'WhiteBird':8}

class NaiveAgent:


    def __init__(self,ip = "127.0.0.1",id = 28888):
        """
        constructor for RLAgent class
        :param ip: default = 127.0.0.1
        """
        self.ar = ClientActionRobotJava(ip)
        self.tp = TrajectoryPlanner() # generate instance for Trajectory Planner
        self.prevTarget = None
        self.firstShot = True
        self.solved = []
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id

    def getNextLevel(self):

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
        print "## MyScore: %d" % scores[self.currentLevel - 1]
        level = 1
        for i in scores:
            #print " level %d %d"%(level, i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1

    def run(self):
        """
        run function
        :return:
        """

        info = self.ar.configure(ClientActionRobot.intToByteArray(self.id))
        self.solved = np.zeros(info[2])
        self.checkMyScore()
        self.currentLevel = self.getNextLevel()
        print "currentLevel %d"%self.currentLevel
        self.ar.loadLevel(self.currentLevel)

        while True:
            state = self.solve()
            if state == state.WON :
                self.checkMyScore()
                self.currentLevel = self.getNextLevel()
                self.ar.loadLevel(self.currentLevel)
                scores = self.ar.checkScore()
                #print "Global best score:"
                #for i in scores:
                    #print " level %d : %f" %(i+1, scores[i])

                self.tp = TrajectoryPlanner()
                self.firstShot = True

            elif state == state.LOST:
                self.failedCounter += 1
                if self.failedCounter > 3:
                    self.failedCounter = 0
                    self.currentLevel = self.getNextLevel()
                    self.ar.loadLevel(self.currentLevel)
                else:
                    print "restart"
                    self.ar.restartLevel
            elif state == state.LEVEL_SELECTION:
                print "unexpected level selection page, go to the last current level: %d" % self.currentLevel
                self.ar.loadLevel(self.currentLevel)
            elif state == state.MAIN_MENU:
                print"unexpected main menu page, reload the level: %d" % self.currentLevel
                self.ar.loadLevel(self.currentLevel)
            elif state == state.EPISODE_MENU:
                print "unexpected episode menu page, reload the level: %d" % self.currentLevel
                self.ar.loadLevel(self.currentLevel)




    def solve(self):
        """
        solve function
        :return: GameState
        """

        screenshot = self.ar.doScreenShot()
        vision = Vision(screenshot)
        sling = vision.findSlingshotMBR()
        state = self.ar.checkState()

        while sling == None and state == state.PLAYING :
            print "no slingshot detected Please remove pop up or zoom out"

            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

        pigs = vision.findPigsMBR()
        state = self.ar.checkState()

        # if there is a sling, then play, otherwise skip
        if sling != None:

            print "### There is a sling"
            # If there are pigs we pick up a pig randomly and shoot it
            if len(pigs) != 0:
                print "### There is a pig"
                pig = pigs[(int)(random.random() % len(pigs))]
                _tpt = pig.getCenter()

                # if the target is very close to before, randomly choose a point near it
                if (self.prevTarget != None and self.distance(self.prevTarget, _tpt) < 10):
                    _angle = random.random() * pi * 2
                    _tpt.setLocation(_tpt.x + int(cos(_angle) * 10),  _tpt.y + int(sin(_angle) * 10))
                    print "## Randomly chaning to " , _tpt

                self.prevTarget = Point(int(_tpt.x), int(_tpt.y))

                # estimate the trajectory
                pts = self.tp.estimateLaunchPoint(sling, _tpt)

                # do a high shot when entering a level to find an accurate velocity
                if self.firstShot and len(pts) > 1:
                    releasePoint = pts[1]
                else:
                    if len(pts) == 1:
                        releasePoint = pts[0]
                    else:
                        if len(pts) == 2:

                            print "## first shot"
                            # randomly choose between the trajectories, with a 1 in 6 chance of choosing the high one
                            if int(random.random() % 6) == 0:
                                releasePoint = pts[1]
                            else:
                                releasePoint = pts[0]

                print "## reference point"
                refPoint = self.tp.getReferencePoint(sling)

                tapTime = 0
                if releasePoint != None :

                    releaseAngle = self.tp.getReleaseAngle(sling, releasePoint)
                    print "## Release Point" , releasePoint
                    print "## Release Angle" , degrees(releaseAngle)

                    tapInterval = 0

                    Bird = self.ar.getBirdTypeOnSling()
                    if Bird == Birds['RedBird']:
                        tapInterval = 0
                    elif Bird == Birds['YellowBird']:
                        tapInterval = 65 + int(random.random() % 25)
                    elif Bird == Birds['WhiteBird']:
                        tapInterval = 50 + int(random.random() % 20)
                    elif Bird == Birds['BlueBird']:
                        tapInterval = 65 + int(random.random() % 20)
                    else:
                        tapInterval = 60

                    tapTime = self.tp.getTapTime(sling, releasePoint, _tpt, tapInterval)

                else:
                    print "No Release Point Found"
                    return self.ar.checkState()

                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                _sling = vision.findSlingshotMBR()

                if _sling != None :

                    scale_diff = (sling.width - _sling.width)**2 + (sling.height - _sling.height)**2

                    if scale_diff < 25:
                        dx = int(releasePoint.getX() - refPoint.x)
                        dy = int(releasePoint.getY() - refPoint.y)

                        if dx < 0:
                            self.ar.shoot(int(refPoint.x), int(refPoint.y), int(dx), int(dy), 0, int(tapTime), False)
                            state = self.ar.checkState()
                            self.ar.fullyZoomOut()
                            screenshot = self.ar.doScreenShot()
                            InGameScore = self.ar.getInGameScore(screenshot)
                            print "## In game Score %d " % InGameScore

                            if state == state.PLAYING:
                                screenshot = self.ar.doScreenShot()
                                vision = Vision(screenshot)
                                traj = vision.findTrajPoints()
                                self.tp.adjustTrajectory(traj, sling, releasePoint)
                                self.firstShot = False
                            else:
                                self.checkMyScore()

                    else:
                        print "Sclae is changed, can not execute the shot, will re-segment the image"
                else:
                    print "no sling detected, cannot execute the shot, will re-segment the image"


        return state

    def distance(self, p1, p2):
        return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)






if __name__ == '__main__':

    Bird = NaiveAgent()
    Bird.run()
    shutdownJVM()


