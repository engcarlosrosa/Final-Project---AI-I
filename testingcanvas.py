from tkinter import *

master = Tk()

w = Canvas(master, width=1440, height=900)
w.pack()


#w.create_line(0, 0, 800, 600, fill="green", dash=(40, 20))
#w.create_line(0, 600, 800, 0, fill="red", dash=(20, 10))

w.create_rectangle(0, 0, 1440, 900, fill="black")


root = Tk()
T = Text(root, height=200, width=300)
T.pack()
T.insert(END, "Just a text Widget\nin two lines\n")
mainloop()
