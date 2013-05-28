# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:27:03 2013

@author: aasensio
"""
import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from scipy.io.idl import readsav
import matplotlib.cm as cm
from numpy import amin, amax, gradient

import pylab
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Hinode')

        self.create_main_frame()
        self.on_draw(0,0,True)

    
    def on_draw(self, x, y, redraw):
        """ Redraws the figure
        """
        if (redraw == True):
            if (self.imageShown == "I"):
                self.axesLeft.imshow(self.hinodeI.stI[:,:,0], cmap = cm.Greys_r, 
                         interpolation='nearest', extent=[0,1,0,1], origin='lower')
            else:
                self.axesLeft.imshow(self.hinodeV.stV[:,:,20], cmap = cm.Greys_r, 
                         interpolation='nearest', extent=[0,1,0,1], origin='lower')
                
            self.canvasLeft.draw()
            self.axesRightI.clear()
            self.axesRightV.clear()            
            self.lineRightI, = self.axesRightI.plot(self.lambdaAxis['lambda'],self.hinodeI.stI[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            self.lineRightV, = self.axesRightV.plot(self.lambdaAxis['lambda'],self.hinodeV.stV[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            
            C = 4.6686e-13
            deltaLambda = self.lambdaAxis['lambda'][1] - self.lambdaAxis['lambda'][0]
            dIdl = 6301.0**2 * 3.0 * gradient(self.hinodeI.stI[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0], deltaLambda)
            self.lineRightWeak, = self.axesRightV.plot(self.lambdaAxis['lambda'], -C * dIdl * self.Bpar.bpar[x/4,y/4])
            self.axesRightI.set_ylim([0,1.1])
            self.axesRightV.set_ylim([-0.01,0.01])
            self.axesRightI.ticklabel_format(useOffset=False)
            self.axesRightV.ticklabel_format(useOffset=False)
            self.canvasRight.draw()
        else:
            self.lineRightI.set_ydata(self.hinodeI.stI[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            self.lineRightV.set_ydata(self.hinodeV.stV[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            C = 4.6686e-13
            deltaLambda = self.lambdaAxis['lambda'][1] - self.lambdaAxis['lambda'][0]
            dIdl = 6301.0**2 * 3.e0 * gradient(self.hinodeI.stI[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0], deltaLambda)
            self.lineRightWeak.set_ydata(-C * dIdl * self.Bpar.bpar[x/4,y/4])
            self.axesRightI.set_ylim([0,1.1])
            ymin = amin(self.hinodeV.stV[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            ymax = amax(self.hinodeV.stV[x/4,y/4,:] / self.hinodeI.stI[x/4,y/4,0])
            self.axesRightV.set_ylim([ymin,ymax])
            self.axesRightI.set_title("X: %d - Y: %d - B: %f"%(x,y,self.Bpar.bpar[x/4,y/4]))
            self.axesRightI.ticklabel_format(useOffset=False)
            self.axesRightV.ticklabel_format(useOffset=False)
            self.canvasRight.draw()       
        
                
    def on_move(self, event):
        """ On move
        """
        self.xpos = 512-int(event.xdata*512)
        self.ypos = 512-int(event.ydata*512)
        self.on_draw(self.xpos, self.ypos, False)
        
    def button_stokesI(self):
        self.imageShown = "I"
        self.on_draw(self.xpos, self.ypos, True)
        
    def button_stokesV(self):
        self.imageShown = "V"
        self.on_draw(self.xpos, self.ypos, True)
        
        
    
    def create_main_frame(self):
        
        self.main_frame = QWidget()
        
        fullLayout = QVBoxLayout()
        
        plotLayout = QHBoxLayout()
        
        self.dpi = 100
        self.xpos = 0
        self.ypos = 0
        self.imageShown = "I"
        
        self.leftPlot = QWidget()
        self.leftPlot.setFixedSize(512,512)
        self.figLeft = Figure((512/self.dpi, 512/self.dpi), dpi=self.dpi)        
        self.canvasLeft = FigureCanvas(self.figLeft)
        self.canvasLeft.setParent(self.leftPlot)
        self.axesLeft = self.figLeft.add_axes([0,0,1,1])
        self.canvasLeft.mpl_connect('motion_notify_event', self.on_move)
        plotLayout.addWidget(self.leftPlot)
        
        self.rightPlot = QWidget()
        self.rightPlot.setFixedSize(512,512)
        self.figRight = Figure((512/self.dpi, 512/self.dpi), dpi=self.dpi)        
        self.canvasRight = FigureCanvas(self.figRight)
        self.canvasRight.setParent(self.rightPlot)
        self.axesRightI = self.figRight.add_subplot(211)
        self.axesRightI.tick_params(labelsize=8)        
        self.axesRightV = self.figRight.add_subplot(212)
        self.axesRightV.tick_params(labelsize=8)
        self.axesRightV.ticklabel_format(useOffset=False)
        plotLayout.addWidget(self.canvasRight)
        
        fullLayout.addLayout(plotLayout)
        
        butLayout = QHBoxLayout()
        
        butI = QRadioButton("Intensity",self)
        butV = QRadioButton("Magnetogram",self)
        butI.setChecked(True)
        buttons = QButtonGroup()
        buttons.addButton(butI)
        buttons.addButton(butV)
        butLayout.addWidget(butI)
        butLayout.addWidget(butV)
        
        fullLayout.addLayout(butLayout)
        
        self.connect(butI, SIGNAL('clicked()'), self.button_stokesI)
        self.connect(butV, SIGNAL('clicked()'), self.button_stokesV)
                
        # Read Hinode data
        self.hinodeI = readsav('../data/stI_rebinned.idl')
        self.hinodeV = readsav('../data/stV_rebinned.idl')
        self.Bpar = readsav('../data/Bpar.idl')
        self.lambdaAxis = readsav('../data/wavelengthAxis.idl')
                
#        self.main_frame.addLayout(plotLayout)                
        self.main_frame.setLayout(fullLayout)
        self.setCentralWidget(self.main_frame)

def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()