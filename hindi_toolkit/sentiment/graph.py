import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline, interp1d

def plot_sentiment_curve(lines):
	# f = open("nboutput.txt","r")
	# canvas = plt.figure()
	# rect = canvas.patch
	# rect.set_facecolor('white')
	# lines = f.readlines()
	xl = [i+1 for i in range(len(lines))]
	yl = [int(i) for i in lines]
	x_sm = np.array(xl)
	y_sm = np.array(yl)
	xnew = np.linspace(x_sm.min(), x_sm.max(), 200)
	#xnew = np.linspace(x_sm.min(), x_sm.max(), 60)
	#print(xnew)
	#exit()
	# # y_smooth = spline(xl, yl, x_smooth)
	# sp1 = canvas.add_subplot(1,1,1, axisbg='w')
	#sp1.plot(x, y, 'red', linewidth=2)
	#----------------------------------------------
	# xnew = np.linspace(x_sm.min(),x_sm.max(),300)	
	# power_smooth = spline(x_sm,y_sm,xnew)
	# plt.plot(xnew,power_smooth)
	#-------------------------------------
	#xnew = np.linspace(0,24,24*10)
	ynew = interp1d(xl, yl, kind='linear')
	#print(ynew(xnew))
	#exit()
	#plt.plot(xnew,ynew(xnew))
	plt.plot(xl,yl)
	plt.ylim([-2,2])
	#sp1.plot(xl, yl, 'red', linewidth=2)
	plt.show()
	# canvas.draw()