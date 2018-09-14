'''@author: SuD
   This module is used for graphing the classification which we recieved after running sentiment analysis on the live stream of twitter data
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
# just a style - it's my preference
style.use("ggplot")
#initializing figure
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("twitterOutput.txt","r").read() # getting our data
    lines = pullData.split('\n') # splitting them into seperate lines
    # intializing
    xar = []
    yar = []
    x = 0
    y = 0
    for l in lines[:]:
        x += 1 #incrementing our x axis i.e. the position number on which tweet's classification we are
        # for countering bias because for positive people use less vocabulary but for negative they use more
        if "pos" in l:
            y += 3 # increasing for positive
        elif "neg" in l:
            y -= 1 # decreasing for negative
        xar.append(x)
        yar.append(y)
    ax1.clear() #clearing the figure
    ax1.plot(xar,yar)#plotting the values
ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()