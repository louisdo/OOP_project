import tkinter.messagebox as msbox
from tkinter import *
from tkinter.ttk import Frame, Label, Entry


class Design(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.input_bar()
        self.function_button()

    def input_bar(self):
        self.parent.title("Result")
        self.pack(fill=BOTH, expand=True)

        frame = Frame(self)
        frame.pack(fill=BOTH, expand=True)

        lable = Label(frame, text="Input", width=6)
        lable.pack(side=LEFT, anchor=N, padx=5, pady=5)

        entry = Entry(frame)
        entry.pack(fill=BOTH, padx=5, pady=5, expand=True)

    def function_button(self):
        frame_button = Frame(self, relief=RAISED, borderwidth=1)
        frame_button.pack(fill=BOTH, expand=True)

        self.pack(fill=BOTH, expand=True)
        close_button = Button(self, text="Close", command=self.quit)
        close_button.pack(side=RIGHT, padx=5, pady=5)

        ok_button = Button(self, text="OK", command=self.show_result)
        ok_button.pack(side=RIGHT)

    def show_result(self):
        msbox.showinfo(title="RESULT", message="DONE!")


if __name__ == '__main__':
    root = Tk()
    root.geometry("500x300+400+200")
    app = Design(root)
    root.mainloop()
