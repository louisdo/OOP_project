from tkinter import *
from tkinter.ttk import Frame, Label, Entry
import tkinter.messagebox as msbox


class Design(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.inputBar()
        self.functionButton()

    def inputBar(self):
        self.parent.title("Result")
        self.pack(fill=BOTH, expand=True)

        frame = Frame(self)
        frame.pack(fill=BOTH, expand= True)

        lable = Label(frame, text="Input", width=6)
        lable.pack(side=LEFT, anchor= N, padx=5, pady=5)

        entry = Entry(frame)
        entry.pack(fill=BOTH, padx= 5, pady= 5, expand=True)

    def functionButton(self):
        frameButton = Frame(self, relief=RAISED, borderwidth=1)
        frameButton.pack(fill=BOTH, expand=True)

        self.pack(fill=BOTH, expand=True)
        closeButton = Button(self, text="Close", command=self.quit)
        closeButton.pack(side=RIGHT, padx=5, pady=5)

        okButton = Button(self, text="OK", command=self.showResult)
        okButton.pack(side=RIGHT)

    def showResult(self):
        msbox.showinfo(title= "RESULT", message= "DONE!")

if __name__ == '__main__':

    root = Tk()
    root.geometry("500x300+400+200")
    app = Design(root)
    root.mainloop()
