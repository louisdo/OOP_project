import tkinter.messagebox as msbox
from tkinter import *
from tkinter.ttk import Frame, Label, Entry


class Design(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.input_sentence = " "
        self.entry = 0
        self.parent = parent
        self.input_bar()
        self.function_button()

    def input_bar(self):
        self.parent.title("Result")
        self.pack(fill=BOTH, expand=True)

        frame = Frame(self)
        frame.pack(fill=BOTH, expand=True)

        label = Label(frame, text="Input", width=6)
        label.pack(side=LEFT, anchor=N, padx=5, pady=5)

        self.entry = Entry(frame)
        self.entry.pack(fill=BOTH, padx=5, pady=5, expand=True)

    def function_button(self):
        frame_button = Frame(self, relief=RAISED, borderwidth=1)
        frame_button.pack(fill=BOTH, expand=True)

        self.pack(fill=BOTH, expand=True)
        close_button = Button(self, text="Close", command=self.quit)
        close_button.pack(side=RIGHT, padx=5, pady=5)

        ok_button = Button(self, text="OK", command=self.show_result)
        ok_button.pack(side=RIGHT)

    def test_process(self, sentence):
        return "Processing sentence: " + sentence

    def test_function(self, function, sentence):
        return function(sentence)

    def show_result(self):
        self.input_sentence = self.entry.get()
        result = self.test_function(self.process, self.input_sentence)
        msbox.showinfo(title="RESULT", message=result)



root = Tk()
root.geometry("500x300+400+200")
app = Design(root)
root.mainloop()
