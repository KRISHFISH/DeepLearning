from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict 
window= Tk()
l1= Label()
def My_Project():
    global l1
    Widget= cv
    x= window.winfo_rootx()+ Widget.winfo_x()
    y= window.winfo_rooty()+ Widget.winfo_y()
    x1= x+Widget.winfo_width()
    y1= y+Widget.winfo_height()
    img= ImageGrab.grab().crop((x,y,x1,y1)).resize((28,28))
    img= img.convert('L')
    x= np.asarray(img)
    vec= np.zeros((1,784))
    k=0
    for i in range(28):
        for j in range(28):
            vec[0][k]= x[i][j]
            k=k+1
    Theta1= np.loadtxt('Theta1.txt')
    Theta2= np.loadtxt('Theta2.txt')
    pred= predict(Theta1, Theta2, vec/255)
    l1= Label(window, text= "digit="+ str(pred[0]),font= ('Algerian', 20))
    l1.place(x=230, y=420)
last_x, last_y= None, None
def clear_widget():
    global cv, l1
    cv.delete("all")
    l1.destroy()
def event_activation(event):
    global last_x, last_y
    #x,y= event.x, event.y
    cv.bind('<B1-Motion>', draw_lines)
    last_x, last_y= event.x, event.y
def draw_lines(event):
    global last_x, last_y
    x, y= event.x, event.y
    cv.create_line((last_x, last_y, x,y), width=30, fill= "White", capstyle= ROUND, smooth= True, splinesteps= 12)
    last_x, last_y= x,y 
B1= Button(window, text= "1. Clear Canvas", font=("Algerian", 15), bg= "Orange", fg= "Black", command= clear_widget)
B1.place(x=120, y=370)
B2= Button(window, text= "2. Prediction", font=("Algerian", 15), bg= "Orange", fg= "Black", command= My_Project)
B2.place(x=320, y=370)
cv= Canvas(window, width= 350, height= 290, bg= "Black")
cv.place(x= 120, y=70)
cv.bind('<Button-1>', event_activation)
window.mainloop()