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
import random

import gym
from gym.spaces import Discrete, Box

import tensorflow as tf
import keras

#from tf_agents.agents.dqn import dqn_agent
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#random test shit
from copy import deepcopy

class Env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(sim.action.shape),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(sim.state.shape),
            dtype=np.float32
        )

    def reset(self):
        sim.reset()
        return sim.state 
    def step(self, action):
        sim.action = action
        sim.step()
        
        info = {}

        return sim.state, sim.reward, sim.done, info
    def render(self):
        pass

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
    theta = NumericProperty(0)
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
        self.envirement = self.root.get_screen('mainScreen').ids.env

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
        self.error = 0
        self.episodes = 0
        self.score = 0 

        #graph variables
        self.y = []#self.graphLen * [None]
        self.x = []#self.graphLen * [None]
        self.y2 = []

        #ML
        #ai class. maby idfk shit
        self.state = np.array([0,0,0,0], dtype = np.float32)
        self.action = np.array([0,0], dtype = np.float32)
        self.reward = 0
        self.right = False
        self.left = False

        #reset sim
        self.reset()

        #create env. gym class i think
        self.env = Env()

        #create model
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(1,4,)),#wrong shape or sum
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(units=24, activation=tf.nn.relu),
                tf.keras.layers.Dense(2, activation=tf.nn.softmax)#maby not right...
                ]) 

        #create agent
        self.dqn = DQNAgent(
              model=self.model, 
              memory=SequentialMemory(limit=50000, window_length=1), 
              policy=BoltzmannQPolicy(), 

              nb_actions=2, 
              nb_steps_warmup=10,
              target_model_update=1e-2
              )
        #compile agent
        self.dqn.compile(tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

    #continus cycle
    def cycle (self, readCYCLETIME):
        #scuffed time shit  
        self.readCYCLETIME = readCYCLETIME
        if self.runTime != 0 and self.runTime < .03:
            time.sleep(1)

        """if not self.done:
                #env.render()
            action = self.env.action_space.sample()
            self.env.step(action)
            

            #print(f'Episode:{self.episodes} Score:{self.score}')"""
        

        #runtime
        self.runTime += self.readCYCLETIME 

    def plot(self):
        if self.plotGrap:
            self.updateGraph()
        if self.autoMod:
            self.keyboardControll()

        #graph
        self.x.append(self.runTime)
        self.y2.append(self.pengelum.theta/np.pi)
        self.y.append(self.slider.value/484)

        #update grapics
        self.pengelum.angleDegrees = float(np.degrees(self.pengelum.theta))
        self.pengelum.xPos = self.slider.value
            
    def step(self):
        #reset reward
        self.reward = 0

        try:
            self.right, self.left = self.action #agent action
        except:
            self.right, self.left = 0.0, 0.0
            print("action failed")
        
        self.mafs()
        
        self.state = np.array([self.slider.value, self.sliderVel, self.error, self.pengelum.rotVel])#state of the sim

        #reward
        if (self.error < .2) and (self.error > -.2):
            self.reward += 1
        self.score += self.reward
        
        #check done
        if self.episodes >= 5:
            self.done = True
            
        #done
        if self.done:
            pass

        self.output.text = f"episode nr {self.episodes+1}"#output. whatever
        

        if (self.error > .4) or (self.error < -.4):
            self.reset()
            self.episodes += 1
        
        #update grapichs. mayby jalla
        self.pengelum.angleDegrees = float(np.degrees(self.pengelum.theta))
        self.pengelum.xPos = self.slider.value

    def mafs(self):
        self.time = time.time()#set time to actual time

        if self.timeLast == 0:#on first cycle
            self.timeLast = self.time-.2
            
        self.mafsTime = self.time - self.timeLast #calc mafstime. basically cycletime
        self.timeLast = self.time#uptdate last time
        
        
        self.digitalControll()#digital control

        
        
        self.sliderVel = -float((self.slider.value - self.sliderLast)*self.mafsTime)#slider vel
        self.sliderLast = self.slider.value#update last slider val

        self.sliderResult = (self.sliderVel/10) * np.cos(self.pengelum.theta)#how much te slider vel will affect theta

        

        self.pengelum.rotVel += float((((self.envirement.g/self.pengelum.L) * np.sin(self.pengelum.theta))-(self.pengelum.rotVel * .3)))*self.mafsTime#angular vel
        self.pengelum.theta += self.pengelum.rotVel + float(self.sliderResult)#set angle. belive slider result shoud be here. prollyu not 100%right. but feels realistic
    
        self.error = ((((self.pengelum.theta/np.pi)/2) % 1)-.5)*-2#calc error
    



    def reset(self):
        self.pengelum.theta = ((.995+(random.randint(-10,10)/200))*np.pi)#.99 so does not get stuck
        self.pengelum.rotVel = 0
        self.slider.value = 0
        self.sliderLast = 0
        self.sliderResult = 0
        self.left = 0
        self.right = 0
        self.state = np.array([self.slider.value, self.sliderVel, self.error, self.pengelum.rotVel])#state of the sim

        #update grapichs. mayby jalla
        self.pengelum.angleDegrees = float(np.degrees(self.pengelum.theta))
        self.pengelum.xPos = self.slider.value
        

    def digitalControll(self):
        try:
            if self.right != 0:
                self.slider.value += float(self.right) *1000* self.mafsTime#1500 for float 0-1 val. 
                self.right = False
            if self.left != 0:
                self.slider.value -= float(self.left) *1000* self.mafsTime
                self.left = False
        except:
            pass
        

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

        #test

        
        self.dqn.fit(self.env, nb_steps=5000, visualize=0, verbose=1)#tror error skyldes at resetfunksjonen kalles opp og dermed ikke får hentet de riktige observation variablene.

    def stepButton(self):
        self.step()
    def resetButton(self):
        self.env.reset()    
        
        
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