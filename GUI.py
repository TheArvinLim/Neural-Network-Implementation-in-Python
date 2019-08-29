from Neural_Network import*
import tkinter as tk
from PIL import Image
from PIL import ImageDraw
import pickle
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import mnist_loader

with open('BOOM.pkl', 'rb') as input:
    network = pickle.load(input)

def predict(network, x):
    prediction = network.feed_forward(x)
    print("I guess {}".format(str(np.argmax(prediction))))
    np.set_printoptions(precision=2)
    for num, val in enumerate(prediction):
        if val < 0.01:
            val = 0
        print("{}: {}%".format(num, val*100))
    #view_MNIST(x)
    return np.argmax(prediction)

def view_MNIST(image):
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
def prepare_image(filename):
    img = Image.open(filename)
    img = img.resize((28,28), Image.ANTIALIAS) #resize
    img = img.convert('L') #convert to greyscale
    img = np.array(img) #convert to numpy array
    img = 255 - img #invert black and white
    img = img.reshape((784,1)) #flatten array
    return img

class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Guess!",width=10,bg='white',command=self.guess)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)

        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def guess(self):
        filename = "temp.jpg"
        self.image.save(filename)
        x = prepare_image(filename)
        prediction = predict(network, x)
        answer['text'] = str(prediction)
    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=10,fill='black')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,0,0),width=10)

        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    answer = tk.Label(root)
    answer.pack()
    root.mainloop()


# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# for data in validation_data:
#     x, y = data[:]
#     prediction = predict(network, x)
#     print("Prediction: {}".format(prediction)) 
#     print("Actual: {}".format(y)) 
#     view_MNIST(x)
