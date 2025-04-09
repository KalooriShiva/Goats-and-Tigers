
from tkinter import *
from tkinter import messagebox
import os
import numpy as np
import sys
import time
from pullimekka import *
from functions import *
from PIL import ImageTk, Image
from random import randint, choice
from sys import platform
from rl_tiger import QLearningAgent
from rl_goat import GoatQLearningAgent  # Your goat RL class
agent_goat = GoatQLearningAgent()
agent_goat.load_model('goat_q_table.pkl')

agent = QLearningAgent()
agent.load_model()  # Loads from 'q_table.pkl'
# Board parts

numPosition = 23 
if platform == 'darwin':
    # Mac
    boardSize = 500
    fontSize = 15
    buttonStyle = 30
elif platform == 'win32' or platform == 'cygwin':
    # Windows
    boardSize = 500
    fontSize = 12
    buttonStyle = 0
elif  platform == 'linux':
    # Linux
    pass

boardSize = 500
Board().clearBoard()
possible_pos = ['a1','a2','a3','b0','b1','b2','b3','b4','c1','c2','c3','c4','d1','d2','d3','d4','e1','e2','e3','e4','f1','f2','f3']
root = os.path.abspath('images')
# Game modes:
class Main():

    def __init__(self,mode):

        tigerPhoto =  root + '/tiger.gif'
        goatPhoto = root + '/goat.gif'
            
        self.window = Tk()        
        self.window.title('Huligutta (Goats & Tigers)')
        self.window.geometry(str(boardSize) + 'x' + str(boardSize+20))
        self.window.resizable(0,0)
        # self.window.tk_focusFollowsMouse()
        # self.window.tk.call('tk', 'scaling', 17)
        self.tiger = PhotoImage(file=tigerPhoto)
        self.tiger = self.tiger.subsample(2,2)
        self.goat = PhotoImage(file=goatPhoto)
        self.goat = self.goat.subsample(2,2)
        self.canvas = Canvas(self.window,width=boardSize,height=boardSize)

        # For self.turn, Goat: False, Tiger: True
        self.turn = False
        self.move = False
        self.initialize_board()
        self.canvas.pack()
        
        self.location = ''
        self.turntext = StringVar()
        self.numGoats = StringVar()
        self.goatsEatentext = StringVar()
        self.selectedBtn = StringVar()
        self.goatEaten = 0
        self.goatCount = 0
        self.moveCount = 0
        self.moveCount_prev = 0
        self.turnDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.turntext).place(x=boardSize - 100,y= 25)
        self.selectedDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.selectedBtn).place(x=boardSize - 100,y = 50)
        self.goatDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.numGoats).place(x=10,y= 25)
        self.goatEatenDisp = Label(self.window, font=("Helvetica", fontSize), textvariable=self.goatsEatentext).place(x= 10,y= 50)
        self.numGoats.set("Number of goats: " + str(self.goatCount))
        self.goatsEatentext.set("Goats eaten: " + str(self.goatEaten))
        self.turnDisplay()
        self.selectedToggle()
        self.reportBugbtn = Button(self.window,text = 'Report Bug',command=lambda : self.report_bug()).place(x= 70,y = boardSize - 15,anchor=CENTER)

        # Buttons
        # self.btn1  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b0')).place(x=boardSize/2,y=boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn2  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a1')).place(x=boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn3  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a2')).place(x=boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn4  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('a3')).place(x=boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn5  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b1')).place(x=boardSize/2 - 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn6  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b2')).place(x=boardSize/2- 100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn7  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b3')).place(x=boardSize/2-135,y=boardSize/2+70,height=30,width=30,anchor=CENTER)
        # self.btn8  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('b4')).place(x=boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn9  = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c1')).place(x=boardSize/2 - 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn10 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c2')).place(x=boardSize/2 - 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn11 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c3')).place(x=boardSize/2 - 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn12 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('c4')).place(x= boardSize/2 - 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn13 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d1')).place(x=boardSize/2 + 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn14 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d2')).place(x=boardSize/2 + 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn15 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d3')).place(x=boardSize/2 + 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn16 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('d4')).place(x=boardSize/2 + 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        # self.btn17 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e1')).place(x=boardSize/2 + 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn18 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e2')).place(x=boardSize/2+100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn19 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e3')).place(x=boardSize/2 + 135,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        # self.btn20 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('e4')).place(x=boardSize - boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)        
        # self.btn21 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f1')).place(x=boardSize-boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        # self.btn22 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f2')).place(x=boardSize-boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        # self.btn23 = Button(self.window, bd=buttonStyle,command=lambda : self.button_position('f3')).place(x=boardSize-boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)

    def initialize_board(self):
        # Draws the board
        self.canvas.create_rectangle(boardSize/10,boardSize/2 - 70,boardSize-boardSize/10,boardSize/2+70)    
        self.canvas.create_line(boardSize/10,boardSize/2,boardSize-boardSize/10,boardSize/2)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize - boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/10,boardSize - boardSize/10,boardSize - boardSize/10,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/2 - 80,boardSize - boardSize/10)
        self.canvas.create_line(boardSize/2,boardSize/10, boardSize/2 + 80,boardSize - boardSize/10)

        printAndLog('========================')
        attempts = textCount('Attempts: ')
        printAndLog('Attempts: ' +  str(attempts))
        printAndLog('Game Mode: '+ mode)
        printAndLog('========================')

    def button_position(self,pos):
        # Moves pieces in clicks

        # print('###### Debug ######')
        # print('location:', self.location)
        # print('Position clicked: ', pos)

        # Handles how goat positions move
        self.moveCount_prev = self.moveCount

        if mode == 'pvp':
            self.pvpMode(pos)
            self.selectedToggle()
            self.turnDisplay()
        elif mode == 'goatPlayer':
            self.goatMode(pos)
        elif mode == 'tigerPlayer':
            self.tigerMode(pos)        

        if self.moveCount_prev != self.moveCount:
            self.update()
            self.collectData()
        self.window.mainloop()                

    def pvpMode(self,pos):

        # How goats move
        if self.turn == False:
            
            if Position(pos[0],pos[1]).content() == 'X':
                print('You must select any empty or goat positions')
            elif Position(pos[0],pos[1]).content() == () and self.move == False:
                if self.goatCount < 15:
                    Goat(pos).place()
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.turn = True
                else: 
                    print("exceeded goats amount")

            elif Position(pos[0],pos[1]).content() == 'O':
                if self.goatCount == 15:
                    if pos == self.location:
                        self.move = not self.move
                        # print('Debug: self.move ',self.move)                    
                # Select valid position to move
                    elif self.move == False: 
                        self.move = True
                        self.location = pos
                else:
                    print("Goats can only be moved if 15 goats are placed")
            else:
                
                if Goat(self.location).move(pos) == 1:
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.move = False
                    self.location = ''
                    self.turn = True
                    self.moveCount = self.moveCount + 1
            pass

        # How tigers move
        elif self.turn == True:
            # print('DEBUG: location',self.location)
            
            # print('DEBUG: possible moves',Piece(pos).possibleMoves())
            if Position(pos[0],pos[1]).content() != 'X' and self.move == False:
                print('You must select any current tiger positions') 

            elif Position(pos[0],pos[1]).content() == 'X':
                if pos == self.location:
                    self.move = not self.move
                elif self.location == '' or self.move == False or pos != self.location:
                    self.move = True
                    self.location = pos

            elif Position(pos[0],pos[1]).content() == () and self.move == True:
                # print('DEBUG: location',self.location)
                # print('DEBUG: secondAdjacent',Piece(self.location).secondAdjacent(pos))
                # print('DEBUG: Adjacent',Piece(self.location).adjacent(pos))
                # print('DEBUG: possiblemoves 2 ', Piece(self.location).possibleMoves())
                if pos in Position(self.location[0],self.location[1]).get_neighbors():
                    tigerMoveFlag = Tiger(self.location).move(pos)

                    if tigerMoveFlag == 1:
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        return

                # TODO: error on capture message when Tiger moves to corner (i.e Tiger move to e4)
                elif pos in Tiger(self.location).possibleMoves():
                    if Tiger(self.location).capture(pos) == 1:

                        self.goatEaten = self.goatEaten + 1 
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1

                else:
                    print('Error on capture')
                    self.move = False
                    self.location = ''
                    self.turn = True
        self.selectedToggle()

    def goatMode(self,pos):

        if self.turn == False:
            if Position(pos[0],pos[1]).content() == 'X':
                print('You must select any empty or goat positions')
            elif Position(pos[0],pos[1]).content() == () and self.move == False:
                if self.goatCount < 15:
                    Goat(pos).place()
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.turn = True
                else: 
                    print("exceeded goats amount")
            elif Position(pos[0],pos[1]).content() == 'O':
                if self.goatCount == 15:
                    if pos == self.location:
                        self.move = not self.move
                        # print('Debug: self.move ',self.move)
                # Select valid position to move
                    elif self.move == False: 
                        self.move = True
                        self.location = pos
                else:
                    print("Goats can only be moved if 15 goats are placed")
            else:
                if Goat(self.location).move(pos) == 1:
                    self.change_button(pos,self.goat)
                    self.window.update()
                    self.move = False
                    self.location = ''
                    self.turn = True
                    self.moveCount = self.moveCount + 1
        self.selectedToggle()
        time.sleep(0.5)
        # if self.turn == True:
        #     # Randomize tiger positions
        #     possibleCaptures = {}
        #     tigers = tigerPositions(Board().boardPositions)
        #     # print("DEBUG: tigers ", tigers)

        #     for tiger in tigers:
        #         # print('DEBUG: Position(tiger[0],tiger[1]).get_captures()', Position(tiger[0],tiger[1]).get_captures())
        #         if Position(tiger[0],tiger[1]).get_captures() != None:
        #             for capture in Position(tiger[0],tiger[1]).get_captures():
        #                 possibleCaptures[tiger] = capture

        #     # tigerPos = choice(tigers)
        #     # while len(Tiger(tigerPos).possibleMoves()) == 0:
        #     #     if len(tigers) != 0:
        #     #         tigers.remove(tigerPos)
        #     #         tigerPos = choice(tigers)
        #     #     else:
        #     #         return
            
        #     # # print('DEBUG Tiger possible moves ',Tiger(tigerPos).possibleMoves())
        #     # tigerChoice = choice(Tiger(tigerPos).possibleMoves())
        #     # # print('DEBUG Tiger choice' , tigerChoice)

        #     state = agent.get_state(Board())
        #     valid_actions = agent.get_valid_actions(tigers)

        #     if not valid_actions:
        #         return  # No valid moves

        #     action = agent.choose_action(state, valid_actions)
        #     tigerPos, tigerChoice = action
            
        #     if pos in tigerPos:
        #         del tigerChoice[tigerChoice.index(pos)]

        #     # print('###### Debug ######')
        #     # print('tiger to move:', tigerPos)
        #     # print('Possible moves: ', Tiger(tigerPos).possibleMoves())
        #     # print('Choice the tiger made: ',choice(Tiger(tigerPos).possibleMoves()))

        #     if len(possibleCaptures) != 0:
        #         tigerPos,where2go = choice(list(possibleCaptures.items()))
        #         # print('DEBUG possibleMOves', tigerPos,' ',where2go)
        #         # print('DEBUG TIGERCPOS', tigerPos)
        #         # print('DEBUG possibleCaptures[tigerPos] ', possibleCaptures[tigerPos])
        #         if Tiger(tigerPos).capture(where2go) == 1:

        #             self.goatEaten = self.goatEaten + 1
        #             self.move = False
        #             self.location = ''
        #             self.turn = False
        #             self.moveCount = self.moveCount + 1
        #     elif tigerChoice in Position(tigerPos[0],tigerPos[1]).get_neighbors():
        #         tigerMoveFlag = Tiger(tigerPos).move(tigerChoice) 

        #         if tigerMoveFlag == 1:
        #             self.move = False
        #             self.turn = False
        #             self.moveCount = self.moveCount + 1
        #             return

        #     else:
        #         print('Error on capture')
        #         self.move = False
        #         self.location = ''
        #         self.turn = True

        if self.turn == True:
            possibleCaptures = {}
            tigers = tigerPositions(Board().boardPositions)

            for tiger in tigers:
                captures = Position(tiger[0], tiger[1]).get_captures()
                if captures:
                    for capture in captures:
                        possibleCaptures[tiger] = capture

            state = agent.get_state(Board())
            valid_actions = agent.get_valid_actions(tigers)

            if not valid_actions:
                return  # No valid moves

            action = agent.choose_action(state, valid_actions)
            tigerPos, tigerChoice = action

            if len(possibleCaptures) != 0:
                tigerPos, where2go = choice(list(possibleCaptures.items()))
                if Tiger(tigerPos).capture(where2go) == 1:
                    self.goatEaten += 1
                    self.move = False
                    self.location = ''
                    self.turn = False
                    self.moveCount += 1
            elif tigerChoice in Position(tigerPos[0], tigerPos[1]).get_neighbors():
                tigerMoveFlag = Tiger(tigerPos).move(tigerChoice)
                if tigerMoveFlag == 1:
                    self.move = False
                    self.turn = False
                    self.moveCount += 1
                    return
            else:
                print('Error on capture')
                self.move = False
                self.location = ''
                self.turn = True


    def tigerMode(self,pos):

        if self.turn == True:
            # print('DEBUG: possible moves',Piece(pos).possibleMoves())
            if Position(pos[0],pos[1]).content() != 'X' and self.move == False:
                print('You must select any current tiger positions') 

            elif Position(pos[0],pos[1]).content() == 'X':
                if pos == self.location:
                    self.move = not self.move
                elif self.location == '' or self.move == False or pos != self.location:
                    self.move = True
                    self.location = pos

            elif Position(pos[0],pos[1]).content() == () and self.move == True:
                # print('DEBUG: location',self.location)
                # print('DEBUG: secondAdjacent',Piece(self.location).secondAdjacent(pos))
                # print('DEBUG: Adjacent',Piece(self.location).adjacent(pos))
                # print('DEBUG: possiblemoves 2 ', Piece(self.location).possibleMoves())
                if pos in Position(self.location[0],self.location[1]).get_neighbors():
                    tigerMoveFlag = Tiger(self.location).move(pos) 

                    if tigerMoveFlag == 1:
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1
                        
                elif pos in Piece(self.location).possibleMoves():
                    if Tiger(self.location).capture(pos) == 1:

                        self.goatEaten = self.goatEaten + 1 
                        self.move = False
                        self.location = ''
                        self.turn = False
                        self.moveCount = self.moveCount + 1

                else:
                    print('Error on capture')
                    self.move = False
                    self.location = ''
                    self.turn = True       
            self.selectedToggle()

        # if self.turn == False:
        #     time.sleep(.2)
        #    # Randomize goat positions
        #     emptyPos = emptyPositions(Board().boardPositions)
        #     goats = goatPositions(Board().boardPositions)
        #     # print("DEBUG: goats ", goats)

        #     if self.numGoats == 15:
        #         goatPos = choice(goats)

        #         while len(Goat(goatPos).possibleMoves()) == 0:
        #             if len(goats) != 0:
        #                 goats.remove(goatPos)
        #             else:
        #                 return

        #         goatChoice = choice(Goat(goatPos).possibleMoves())
        #         # print('DEBUG goatChoice ',goatChoice)

        #         if pos in goatPos:
        #             del goatChoice[goatChoice.index(pos)]

        #         # print('###### Debug ######')
        #         # print('goat to move:', goatPos)
        #         # print('Possible moves: ', Goat(goatPos).possibleMoves())
        #         # print('Choice the goat made: ',choice(Goat(goatPos).possibleMoves()))

        #         if goatChoice in Position(goatPos[0],goatPos[1]).get_neighbors():
        #             goatMoveFlag = Goat(goatPos).move(goatChoice) 

        #             if goatMoveFlag == 1:
        #                 self.move = False
        #                 self.turn = True
        #                 self.moveCount = self.moveCount + 1
        #                 return

        #         else:
        #             print('Error')
        #             self.move = False
        #             self.location = ''
        #             self.turn = False
        #     else:
        #         goatChoice = choice(emptyPos)
        #         Goat(goatChoice).place()
        #         self.move = False
        #         self.turn = True

        if self.turn == False:
            time.sleep(.2)

            board_obj = Board()
            empty_pos = emptyPositions(board_obj.boardPositions)
            goat_positions = goatPositions(board_obj.boardPositions)

            if self.numGoats == 15:
                # All goats placed, now move using Goat AI
                state = agent_goat.get_state(board_obj)
                valid_actions = agent_goat.get_valid_actions(goat_positions)

                if not valid_actions:
                    return  # No valid goat moves

                action = agent_goat.choose_action(state, valid_actions)
                goat_pos, goat_target = action

                if goat_target in Position(goat_pos[0], goat_pos[1]).get_neighbors():
                    goatMoveFlag = Goat(goat_pos).move(goat_target)
                    if goatMoveFlag == 1:
                        self.move = False
                        self.turn = True
                        self.moveCount += 1
                        return
                else:
                    print("Goat AI Error: Invalid target")
                    self.move = False
                    self.turn = False

            else:
                # Goat placement phase: randomly place until 15 goats are on board
                placement = random.choice(empty_pos)
                Goat(placement).place()
                self.move = False
                self.turn = True

                
    def update(self):
        # Updates the Screen
        numGoats = 0
        tigerPos = tigerPositions(Board().boardPositions)
        possibleMovesCount = 0
        for i in range(len(possible_pos)):
            if Position(possible_pos[i][0],possible_pos[i][1]).content() == ():
                self.change_button(possible_pos[i],'')
            elif Position(possible_pos[i][0],possible_pos[i][1]).content() == 'O':
                numGoats = numGoats + 1
                self.change_button(possible_pos[i],self.goat)
            else:
                self.change_button(possible_pos[i],self.tiger)
        
        for tiger in tigerPos:
            possibleMovesCount = possibleMovesCount + len(Piece(tiger).possibleMoves())

        self.goatCount = numGoats
        self.numGoats.set("Number of goats: " + str(numGoats))
        self.goatsEatentext.set("Goats eaten: " + str(self.goatEaten))
        self.turnDisplay()
        self.selectedToggle()
        self.window.update()
        # Endgame
        if possibleMovesCount == 0:
            printAndLog('Goat wins')
            messagebox.showinfo("Game Over","Goat wins") 
            return

        if self.goatEaten == 5:
            printAndLog('Tiger wins')
            messagebox.showinfo("Game Over", "Tiger wins") 
            return

    def collectData(self):
        printAndLog("Move: " + str(self.moveCount))
        printAndLog("--------------------")
        printAndLog("Goats: " + str(self.goatCount))
        tigers = tigerPositions(Board().boardPositions)
        printAndLog("Tigers positions: " + str(tigers))
        printAndLog(str(Board().boardPositions))
        editDistance = edit_distance(Board().boardPositions)
        printAndLog("Edit distance: " + str(editDistance))
        
    def selectedToggle(self):
        if self.move == True:
            self.selectedBtn.set('Selected:' + str(self.location))
        else:
            self.selectedBtn.set('')

    def turnDisplay(self):
        # Displays turn as text in the window
        if self.turn:
            self.turntext.set("Turn: Tiger")
        else:
            self.turntext.set("Turn: Goat")

    def change_button(self,pos,img):
        # change the images of the board pieces

        if pos == 'b0':
            self.btn1 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b0')).place(x=boardSize/2,y=boardSize/10,height=30,width=30,anchor=CENTER)
        if pos == 'a1':
            self.btn2 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a1')).place(x=boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'a2':
            self.btn3 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a2')).place(x=boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'a3':
            self.btn4 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('a3')).place(x=boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'b1':
            self.btn5 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b1')).place(x=boardSize/2 - 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)  
        if pos == 'b2':
            self.btn6 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b2')).place(x=boardSize/2- 100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'b3':
            self.btn7 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b3')).place(x=boardSize/2-135,y=boardSize/2+70,height=30,width=30,anchor=CENTER)
        if pos == 'b4':
            self.btn8 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('b4')).place(x=boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER) 
        if pos == 'c1':
            self.btn9 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c1')).place(x=boardSize/2 - 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'c2':
            self.btn10 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c2')).place(x=boardSize/2 - 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'c3':        
            self.btn11 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c3')).place(x=boardSize/2 - 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'c4':
            self.btn12 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('c4')).place(x= boardSize/2 - 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)
        if pos == 'd1':     
            self.btn13 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d1')).place(x=boardSize/2 + 25,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'd2':
            self.btn14 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d2')).place(x=boardSize/2 + 38,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'd3':      
            self.btn15 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d3')).place(x=boardSize/2 + 53,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'd4':
            self.btn16 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('d4')).place(x=boardSize/2 + 80,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)  
        if pos == 'e1':  
            self.btn17 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e1')).place(x=boardSize/2 + 65,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'e2':
            self.btn18 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e2')).place(x=boardSize/2+100,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'e3':
            self.btn19 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e3')).place(x=boardSize/2 + 135,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)
        if pos == 'e4':
            self.btn20 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('e4')).place(x=boardSize - boardSize/10,y=boardSize - boardSize/10,height=30,width=30,anchor=CENTER)        
        if pos == 'f1':
            self.btn21 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f1')).place(x=boardSize-boardSize/10,y=boardSize/2 - 70,height=30,width=30,anchor=CENTER)
        if pos == 'f2':
            self.btn22 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f2')).place(x=boardSize-boardSize/10,y=boardSize/2,height=30,width=30,anchor=CENTER)
        if pos == 'f3': 
            self.btn23 = Button(self.window, image=img, bd=buttonStyle,command=lambda : self.button_position('f3')).place(x=boardSize-boardSize/10,y=boardSize/2 + 70,height=30,width=30,anchor=CENTER)

    def report_bug(self):
        printAndLog('Bug reporting...')
        print('What to include for the report:')
        print('What happened?')
        print('What operating system you are running?')
        reason = input('Type the reason for the bug:')
        printAndLog('Bug reported \nReason: ' + reason)

    def start(self):
        # Sets up the game window
        # Tiger's initial positions

        # Uncomment to start official game 
        # Tiger('b0').place()
        # Tiger('c1').place()
        # Tiger('d1').place()

        # Uncomment to randomize initial positions
        numOpenPositions = random.randint(3,18)
        # numGoatsPlaced = numOpenPositions - 3
        moves = random.sample(possible_pos,k=numOpenPositions)
        randomTigerPositions = random.sample(moves,k=3)
        for tiger in randomTigerPositions:
            Tiger(str(tiger)).place()
            moves.remove(str(tiger))
        # for goat in moves:
        #     Goat(str(goat)).place()
        #########################################################

        if mode == 'tigerPlayer':
            goats = goatPositions(Board().boardPositions)
            empty = emptyPositions(Board().boardPositions)
            emptyPos = choice(empty)
            Goat(emptyPos).place()
            self.turn = True
        self.update()
        self.window.mainloop()        

if __name__ == "__main__":
    # Game mode:
        # Uncomment to choose the game mode

    # mode = 'pvp'          # Player vs. Player
    # mode = 'goatPlayer'   # You are Goat
    mode = 'tigerPlayer'  # You are tiger
    
    game = Main(mode)
    game.start()