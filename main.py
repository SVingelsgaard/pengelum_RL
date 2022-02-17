from doctest import master
from operator import length_hint
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

#import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


#window settings
"""Config.set('graphics', 'window_state', 'maximized')
Config.set('graphics', 'fullscreen', '1')"""


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
    yPos = NumericProperty(0)#px
    angleDegrees = NumericProperty(0)
    

    #for math shit
    theta = NumericProperty(.0*np.pi)
    L = NumericProperty(22.5)#length of pengelum. In meters on screen(.06). 325px(to the center of the blub)
    xx = NumericProperty(0.0)#distance of arc form 0deg to pengelum. In meters 
    rotAcc = NumericProperty(0.0)#gravety in the direction of rotation
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
        self.sliderAcc = 0
        self.sliderLast = 0
        self.pengelum = self.root.get_screen('mainScreen').ids.pengelum

        self.output = self.root.get_screen('mainScreen').ids.output

        self.graph = self.root.get_screen('mainScreen').ids.graph

        self.graphLen = 3*60#sampels/frames in the plot

        self.errorLast = 0#for not fuck up

        self.integralError = 0#allso for not fuckup
        self.autoMod = False
        self.plotGrap = False

        #graph variables
        self.y = []#self.graphLen * [None]
        self.x = []#self.graphLen * [None]
        self.y2 = []

    #continus cycle
    def cycle(self, readCYCLETIME):
        self.readCYCLETIME = readCYCLETIME
        if self.runTime != 0 and self.runTime < .03:
            time.sleep(1)


        self.mafs()
        if self.plotGrap:
            self.updateGraph()
        if self.autoMod:
            self.PID()


        #graph
        self.x.append(self.runTime)
        self.y2.append(self.pengelum.theta/np.pi)
        self.y.append(self.slider.value/484)
    
        #update grapics
        self.pengelum.angleDegrees = float(np.degrees(self.pengelum.theta))
        self.pengelum.xPos = self.slider.value

        #runtime
        self.runTime += readCYCLETIME 


    def mafs(self):
        self.pengelum.xx = self.pengelum.L * self.pengelum.theta#calc x. do not thik i need it

        self.sliderAcc = -float(((self.slider.value/10) - self.sliderLast)*self.readCYCLETIME)#slider acc
        if (np.cos(self.pengelum.theta)) > 0:
            self.output.text = "down" 
        else: 
            self.output.text = "up"
        self.sliderResult = self.sliderAcc * np.cos(self.pengelum.theta)#*self.pengelum.L

        self.output.text = str((self.slider.value/10))

        self.pengelum.rotAcc = float(((self.env.g/self.pengelum.L) * np.sin(self.pengelum.theta))-(self.pengelum.rotVel * .01))#rotvel * .01 = air resistance proportional to vel.

        self.pengelum.rotVel += self.pengelum.rotAcc#+ float(self.sliderResult*.5)

        self.sliderLast = self.slider.value/10

        self.pengelum.theta += self.pengelum.rotVel * self.readCYCLETIME+ float(self.sliderResult)


    def build_model(states, actions):
        model = Sequential()
        model.add(Flatten(input_shape=(1,states)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(actions, activation='linear'))
        return model

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
    GUI().run()