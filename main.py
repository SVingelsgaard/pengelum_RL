from doctest import master
from operator import length_hint
from pickle import TRUE
import string
import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.clock import Clock
from kivy.config import Config
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.properties import NumericProperty
from kivy.properties import ListProperty, StringProperty
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.graphics import Line

import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard

import gym
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent



model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = [1, 4]),
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=3, activation=tf.nn.softmax)#maby not right...
        ])  

class Env(gym.Env):
    def reset():
        sim.reset()
    def step():
        sim.mafs()
 



class WindowManager(ScreenManager):
    pass
class StartScreen(Screen):
    pass
class MainScreen(Screen):
    pass
class Envirement(FloatLayout):
    g = NumericProperty(-29.81/10)#m/s^2
    M = NumericProperty(10)# how manny pixels in a meter
class Output(Label):
    text = StringProperty("0.0")
class Pengelum(Image):
    #for graphics   
    xPos = NumericProperty(0)#px
    angleDegrees = NumericProperty(0)
    #for math shit
    theta = NumericProperty(.99*np.pi)
    L = NumericProperty(22.5)#length of pengelum. In meters on screen(.06). 325px(to the center of the blub)
    rotVel = NumericProperty(0.0)

class Graph(BoxLayout):
    pass

class GUI(App):
    
    def on_start(self): #variables
        #system variables
        self.setCYCLETIME = 0.02
        self.readCYCLETIME = 0
        self.runTime = 0
        self.env = self.root.get_screen('mainScreen').ids.env

        #program variables
        self.slider = self.root.get_screen('mainScreen').ids.slider
        self.sliderVel = 0
        self.sliderLast = 0
        self.pengelum = self.root.get_screen('mainScreen').ids.pengelum

        self.output = self.root.get_screen('mainScreen').ids.output

        self.graph = self.root.get_screen('mainScreen').ids.graph

        self.graphLen = 3*60#sampels/frames in the plot

        self.errorLast = 0#for not fuck up

        self.integralError = 0#allso for not fuckup
        self.autoMod = False
        self.plotGrap = False
        self.done = False
        self.time = 0
        self.timeLast = 0
        self.mafsTime = 0

        #graph variables
        self.y = []#self.graphLen * [None]
        self.x = []#self.graphLen * [None]
        self.y2 = []

        #ML
        #ai class. maby idfk shit
        
    
        self.state = np.array([[0,0,0,0]], dtype = np.float32)
        self.action = np.array([False,False,False], dtype = np.bool_)
        self.reward = 0


        '''q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=(100,))'''


        self.right = 0.0
        self.left = 0.0

    #continus cycle
    def cycle (self, readCYCLETIME):
        #scuffed time shit
        self.readCYCLETIME = readCYCLETIME
        if self.runTime != 0 and self.runTime < .03:
            time.sleep(1)

        self.digitalControll()

        if self.resetEnv:
            self.reset()



        if self.plotGrap:
            self.updateGraph()
        if self.autoMod:
            self.keyboardControll()
        
        #ML data + mafs atm
        #self.step()

        #graph
        self.x.append(self.runTime)
        self.y2.append(self.pengelum.theta/np.pi)
        self.y.append(self.slider.value/484)

        #update grapics
        self.pengelum.angleDegrees = float(np.degrees(self.pengelum.theta))
        self.pengelum.xPos = self.slider.value

        #runtime
        self.runTime += self.readCYCLETIME 

        if (self.error > .4) or (self.error < -.4):
            self.done = True
        if self.done:
            self.reset()
            self.done = False


    def mafs(self):
        self.time = time.time()#set time to actual time

        if self.timeLast == 0:#on first cycle
            self.timeLast = self.time
            self.mafsTime = self.setCYCLETIME
        else:
            self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
            self.timeLast = self.time#uptdate last time

        
        self.sliderVel = -float((self.slider.value - self.sliderLast)*self.mafsTime)#slider vel
        self.sliderLast = self.slider.value#update last slider val

        self.sliderResult = (self.sliderVel/10) * np.cos(self.pengelum.theta)#how much te slider vel will affect theta

        self.output.text = str((self.slider.value/10))#output. whatever

        self.pengelum.rotVel += (float(((self.env.g/self.pengelum.L) * np.sin(self.pengelum.theta))-(self.pengelum.rotVel * .3)))*self.mafsTime#angular vel

        self.pengelum.theta += self.pengelum.rotVel + float(self.sliderResult)#set angle. belive slider result shoud be here. prollyu not 100%right. but feels realistic
    
    
    def step(self):
        #self.mafs()
        self.reward = 0
        self.states = np.array([self.slider.value, self.sliderVel, self.pengelum.theta, self.pengelum.rotVel])#state of the sim
        

        self.error = ((((self.pengelum.theta/np.pi)/2) % 1)-.5)*-2#calc error
        #reward
        if (self.error < .2) and (self.error > -.2):
            self.reward += 1
            #possible break



    def reset(self):
        self.pengelum.theta = .99 * np.pi
        self.pengelum.rotVel = 0
        self.slider.value = 0
        self.sliderLast = 0
        self.sliderResult = 0
        self.left = 0
        self.right = 0
        self.resetEnv = False

    def digitalControll(self):
        if self.right != 0:
            self.slider.value += float(self.right) *1000* self.readCYCLETIME#1500 for float 0-1 val. 
            self.right = False
        if self.left != 0:
            self.slider.value -= float(self.left) *1000* self.readCYCLETIME
            self.left = False

        #clamp
        if self.slider.value > 484:
            self.slider.value = 484
        elif self.slider.value < -484:
            self.slider.value = -484
    def keyboardControll(self):
        if keyboard.is_pressed("right arrow"):
            self.right = 1
        else:
            self.right = 0
        if keyboard.is_pressed("left arrow"):
            self.left = 1
        else:
            self.left = 0
    def autoMode(self):
        if self.autoMod:
            self.autoMod = False
        else:
            self.autoMod = True
    def updateGraph(self):
        plt.clf()
        plt.title("pengelum angle")
        plt.xlabel("t")
        plt.ylabel("angle/sliderval")
        #plt.ylim((25,250))

        self.graph.clear_widgets()
        self.x = self.x[-self.graphLen:]
        self.y = self.y[-self.graphLen:]
        self.y2 = self.y2[-self.graphLen:]
        plt.plot(self.x, self.y, 'k')
        plt.plot(self.x, self.y2, 'r')
        plt.grid()
        self.graph.add_widget(FigureCanvasKivyAgg(plt.gcf()))
    def plotGraph(self):
        if self.plotGrap:
            self.plotGrap = False
            self.graph.remove_widget(FigureCanvasKivyAgg(plt.gcf()))
        else:
            self.plotGrap = True
    #runns cycle
    def runApp(self):
        Clock.schedule_interval(self.cycle, self.setCYCLETIME)
    #runs myApp(graphics)
    def build(self):
        return Builder.load_file("frontend/main.kv")
#runs program and cycle
if __name__ == '__main__':
    sim = GUI()
    sim.run()