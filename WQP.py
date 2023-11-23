from tkinter import *
from tkinter import messagebox,ttk
from tkinter import font
from turtle import onclick
from PIL import Image, ImageTk
import os
import requests
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
import random
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import numpy as np
import anfis as anf
import utils as utl
import pso as pso
import pandas as pd

class home_page:
    def __init__(self, root):
        self.root = root
        self.root.title("User Home Page")
        self.root.geometry('1350x700+0+0')
        self.root.config(bg='#08A3D2')
        self.root.focus_force()
        self.title_frame = Frame(self.root , bd=10 , relief=GROOVE , bg='light sea green').place(x=0 , y=0 ,height=85 , width=1350)
        title = Label(self.root , text='User Details Report' , compound=RIGHT ,
                                  font=('times new roman' , 20 , 'bold') , bg='light sea green' , fg='blue').place(x=600 , y=10 , height=60 , width=500)
        
        self.bg = ImageTk.PhotoImage(file="images/water_back.jpg")
    
        self.bg_image = Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)
         
        #---------------------------------Row1

        self.ph = Label(self.root, text='PH of Water', font=("times new roman", 15, "bold"), bg='LightSkyBlue', fg='black').place(x=350, y=100)
        self.txt_ph = Entry(self.root, font=('times new roman', 15), bg='white')        
        self.txt_ph.place(x=350, y=130, width=200)
        self.unitbx1 = Label(self.root,font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=550,y=130,width=50,height=25)

        hardness = Label(self.root, text='Hardness Of water', font=("times new roman", 15, "bold"), bg='LightSkyBlue', fg='black').place(x=820, y=100)
        self.txt_hardness = Entry(self.root, font=('times new roman', 15), bg='white')
        self.txt_hardness.place(x=820, y=130, width=200)
        self.unitbx2 = Label(self.root,text="mg/L",font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=1020,y=130,width=50,height=25)


        #---------------------------------Row2
        solids = Label(self.root, text='Solids Present in Water', font=("times new roman", 15, "bold"), bg='LightSkyBlue', fg='black').place(x=350, y=170)
        self.txt_solids = Entry(self.root, font=('times new roman', 15), bg='white')
        self.txt_solids.place(x=350, y=200, width=250)
        self.unitbx3 = Label(self.root,text="ppm",font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=550,y=200,width=50,height=25)


        sulphate = Label(self.root, text='Sulphate Present in Water', font=("times new roman", 15, "bold") , bg='LightSkyBlue', fg='black').place(x=820, y=170)
        self.txt_sulphate = Entry(self.root, font=('times new roman', 15), bg='white')
        self.txt_sulphate.place(x=820, y=200, width=250)
        self.unitbx4 = Label(self.root,text="mg\L",font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=1020,y=200,width=50,height=25)


        #---------------------------------Row3
        turbidity = Label(self.root, text='Turbidity Present in Water', font=("times new roman", 15, "bold"), bg='LightSkyBlue', fg='black').place(x=350, y=240)
        self.txt_turbidity = Entry(self.root, font=('times new roman', 15), bg='white')
        self.txt_turbidity.place(x=350, y=270 , width=250)
        self.unitbx5 = Label(self.root,text="NTU",font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=550,y=270,width=50,height=25)


        conductivity = Label(self.root, text='Conductivity Present in Water', font=("times new roman", 15, "bold") , bg='LightSkyBlue', fg='black').place(x=820, y=240)
        self.txt_conductivity = Entry(self.root, font=('times new roman', 15), bg='white')
        self.txt_conductivity.place(x=820, y=270, width=250)
        self.unitbx6 = Label(self.root,text="S\m",font=("times new roman", 15, "bold"), bg='white', fg='black').place(x=1020,y=270,width=50,height=25)


        val = Label(self.root, text='Please use similar formate for the input as a range 50-60', font=("times new roman", 15, "bold") , bg='LightSkyBlue', fg='black').place(x=450, y=380)

        #-----------------------------------Terms
        btn_login = Button(self.root, text="Submit",font=('times new roman', 15), bg='#d77337', fg='white', bd=0, cursor='hand2',command=self.process).place(x=220, y=480, width=130)

    def WQT_check(self):
        # # print(self.txt_ph)
        # if self.txt_ph.get() == "SELECT" or self.txt_hardness.get() == "SELECT" or self.txt_solids.get() == "SELECT" or self.txt_conductivity.get() == "SELECT" or self.txt_sulphate.get() == "SELECT" or self.txt_turbidity.get() == "SELECT" or self.txt_potability.get() == "SELECT" :
        #     messagebox.showerror("Error", "All fields are Required!!!", parent=self.root)
        # else:
            rph = self.txt_ph.get()
            rhardness = self.txt_hardness.get()
            rsolids = self.txt_solids.get()
            # rchloamin = self.txt_chloramin.get()
            rsulphate = self.txt_sulphate.get()
            rconductivity = self.txt_conductivity.get()
            rturbidity = self.txt_turbidity.get()
            # rpotability = self.txt_potability.get()
            
            numbers_ph = [float(n) for n in rph.split('-')]
            print(numbers_ph)
            numbers_hardness = [float(n) for n in rhardness.split('-')]
            numbers_solids = [float(n) for n in rsolids.split('-')]
            # numbers_cloamin = [float(n) for n in rchloamin.split('-')]
            numbers_sulphate = [float(n) for n in rsulphate.split('-')]
            numbers_conductivity = [float(n) for n in rconductivity.split('-')]
            numbers_turbidity = [float(n) for n in rturbidity.split('-')]
            # numbers_potability = [int(n) for n in rpotability.split('-')]
            outfile = open('User_test.csv', 'a', newline='')
            w = csv.writer(outfile)  # Need to write the user input to the .csv file.
            w.writerow(["ph", "hardness", "solids", "conductivity", "sulphate", "turbidity","year"])  # <-This is the portion that seems to fall apart.

            #Everything wrapped in a while True loop, you can change to any loop accordingly
            i = 0
            while i<=7:
                input_ph = random.uniform(numbers_ph[0],numbers_ph[1])
                input_hardness = random.uniform(numbers_hardness[0],numbers_hardness[1])
                input_solids = random.uniform(numbers_solids[0],numbers_solids[1])
                input_conductivity = random.uniform(numbers_conductivity[0],numbers_conductivity[1])
                input_sulphate = random.uniform(numbers_sulphate[0],numbers_sulphate[1])
                input_turbidity = random.uniform(numbers_turbidity[0],numbers_turbidity[1])
                # input_potability = random.randint(numbers_potability[0],numbers_potability[1])
                input_year = 2019
                w.writerow([input_ph, input_hardness, input_solids, input_conductivity, input_sulphate, input_turbidity, input_year])  # <-This is the portion that seems to fall apart.
                print("Samples UPDATED")
                i += 1
            
    def process(self):

        if self.txt_ph.get() == "" or self.txt_hardness.get() == "" or self.txt_solids.get() == "" or self.txt_conductivity.get() == "" or self.txt_sulphate.get() == "" or self.txt_turbidity.get() == "":
            messagebox.showerror("Error", "All fields are Required!!!", parent=self.root)
        else:
            self.WQT_check()
            # self.root = Frame(self.root, bg='light sea green').place(x=0 , y=0 ,height=700 , width=1350)

            self.WQI_Prediction()
            self.plot()
            self.WQT_index()

    def WQI_Prediction(self):
        self.frame_wqp = Frame(self.root, bg='white')
        self.frame_wqp.place(x=50, y=50, width=600, height=600)
        df_WQI = pd.read_csv('User_test.csv')
        df_WQI['ph']=pd.to_numeric(df_WQI['ph'],errors='coerce')
        df_WQI['hardness']=pd.to_numeric(df_WQI['hardness'],errors='coerce')
        df_WQI['solids']=pd.to_numeric(df_WQI['solids'],errors='coerce')
        df_WQI['conductivity']=pd.to_numeric(df_WQI['conductivity'],errors='coerce')
        df_WQI['sulphate']=pd.to_numeric(df_WQI['sulphate'],errors='coerce')
        df_WQI['turbidity']=pd.to_numeric(df_WQI['turbidity'],errors='coerce')
        df_WQI['year']=pd.to_numeric(df_WQI['year'],errors='coerce')

        start=2
        end=1777
        ph=df_WQI.iloc [start:end ,0].astype(np.float64)
        hardness=df_WQI.iloc [start:end ,1].astype(np.float64)
        solids=df_WQI.iloc [start:end ,2].astype(np.float64)
        value=0
        conductivity = df_WQI.iloc[ start:end,3].astype(np.float64)  
        turbidity = df_WQI.iloc [start:end ,5].astype(np.float64)
        sulphate  = df_WQI.iloc [start:end ,4].astype(np.float64)
        # year = df_WQI.iloc [start:end ,6].astype(np.int64)

        #calulation of Ph
        df_WQI['nph']=df_WQI.ph.apply(lambda x: (100 if (x>=14) or (x<=1)  
                                        else(80 if  (13>=x>=11) or (3.5>=x>=1) 
                                            else(60 if (11>=x>=8.5) or (4>=x>=3.5) 
                                                else(40 if (8.5>=x>=7.8) or (6.7>=x>=4)
                                                    else 1)))))
        #calulation of Ph
        df_WQI['whardness']=df_WQI.hardness.apply(lambda x:(100 if (x>180)  
                                        else(80 if  (x<=180 and x>=120) 
                                            else(60 if (x<=120 and x>=60)
                                                else(40 if (x<=60 and x>=17.1) 
                                                    else 1)))))
        #calulation of Ph
        df_WQI['wsolids']=df_WQI.solids.apply(lambda x: (100 if (x>1000)  
                                        else(80 if  (1000>=x>=900) 
                                            else(60 if (900>=x>=600) 
                                                else(40 if (600>=x>=300)
                                                    else 1)))))
        #calulation of Ph
        df_WQI['wconductivity']=df_WQI.conductivity.apply(lambda x:(100 if (x>800)  
                                        else(80 if  (x<=800 and x>=650) 
                                            else(60 if (x<=650 and x>=120)
                                                else(40 if (x<=120 and x>=50) 
                                                    else 1)))))
        df_WQI['wsulphate']=df_WQI.sulphate.apply(lambda x:(100 if (x>250)  
                                        else(80 if  (x<=250 and x>=160) 
                                            else(60 if (x<=160 and x>=80)
                                                else(40 if (x<=80 and x>0) 
                                                    else 1)))))
        df_WQI['wturbidity']=df_WQI.turbidity.apply(lambda x: (100 if (x>50)  
                                 else(80 if  (50>=x>=15) 
                                      else(60 if (15>=x>=7.2)
                                          else(40 if (7.2>=x>=2.5)
                                              else 1)))))
        print("--------------------------------------------------------")
        
        print(df_WQI.nph.mean())
        print(df_WQI.whardness.mean())
        print(df_WQI.wsolids.mean())
        print(df_WQI.wconductivity.mean())
        print(df_WQI.wsulphate.mean())
        print(df_WQI.wturbidity.mean())

        if df_WQI.nph.mean() == 100 or df_WQI.whardness.mean() == 100 or df_WQI.wsolids.mean() == 100 or df_WQI.wconductivity.mean() == 100 or df_WQI.wsulphate.mean() == 100 or df_WQI.wturbidity.mean() == 100:
            self.agg = 100
            df_WQI['wph']=df_WQI.nph 
            df_WQI['whardness']=df_WQI.whardness 
            df_WQI['wsolids']=df_WQI.wsolids
            df_WQI['wconductivity']=df_WQI.wconductivity
            df_WQI['wsulphate']=df_WQI.wsulphate
            df_WQI['wturbidity']=df_WQI.wturbidity
            # df_WQI['wqi']=df_WQI.wph+df_WQI.whardness+df_WQI.wsolids+df_WQI.wconductivity+df_WQI.wsulphate+df_WQI.wturbidity
        else:
            print("--------------------------------------------------------")
            df_WQI['wph']=df_WQI.nph * 0.165
            df_WQI['whardness']=df_WQI.whardness * 0.234
            df_WQI['wsolids']=df_WQI.wsolids * 0.23
            df_WQI['wconductivity']=df_WQI.wconductivity* 0.28
            df_WQI['wsulphate']=df_WQI.wsulphate* 0.16
            df_WQI['wturbidity']=df_WQI.wturbidity* 0.28
            df_WQI['wqi']=df_WQI.wph+df_WQI.whardness+df_WQI.wsolids+df_WQI.wconductivity+df_WQI.wsulphate+df_WQI.wturbidity
            ag=df_WQI['wqi'].mean()
            print("--------------------------------------------------------")
            self.agg = ag
            # self.agg = 200
        if self.agg>=0 and self.agg<=25:
            print("Excellent water quality")
            WQI = "Excellent water quality : "+str(float("{0:.2f}".format(self.agg)))+"\n Drinking, Irrigation and Industrial Purpose"
        elif self.agg>=26 and self.agg<=50:
            print("Good water quality")
            WQI = "Good water quality : "+str(float("{0:.2f}".format(self.agg)))+"\n Drinking, Irrigation and Industrial Purpose"
        elif self.agg>=51 and self.agg<=75:
            print("Poor water quality")
            WQI = "Poor water quality : "+str(float("{0:.2f}".format(self.agg)))+"\n Irrigation and Industrial Purpose"
        elif self.agg>=76 and self.agg<100:
            print("Very Poor water quality")
            WQI = "Very Poor water quality : "+str(float("{0:.2f}".format(self.agg)))+"\n For Irrigation Purpose"
        else:
            print("Unsuitable for drinking")
            WQI = "Unsuitable for drinking  "+"\n Proper Treatment is Required for any Kind of Usage"

        frame2 = Frame(self.frame_wqp, background='white')
        frame2.place(x=50, y=100, width=500, height=480)

        lst = [("Parameter","Index"),
            ('PH',float("{0:.2f}".format(df_WQI['wph'].mean()))),
            ('Hardness',float("{0:.2f}".format(df_WQI['whardness'].mean()))),
            ('Solids',float("{0:.2f}".format(df_WQI['wsolids'].mean()))),
            ('Conductivity',float("{0:.2f}".format(df_WQI['wconductivity'].mean()))),
            ('Sulphate',float("{0:.2f}".format(df_WQI['wsulphate'].mean()))),
            ('Turbidity',float("{0:.2f}".format(df_WQI['wturbidity'].mean())))]
  
        # find total number of rows and
        # columns in list
        total_rows = len(lst)
        total_columns = len(lst[0])
        
        total_rows = len(lst)
        total_columns = len(lst[0])
        for i in range(total_rows):
            for j in range(total_columns):
                 
                self.e = Entry(frame2, width=23, fg='black',bg="light blue",
                               font=('times new roman',16))
                 
                self.e.grid(row=i, column=j)
                self.e.insert(END, lst[i][j])
        Label(frame2,text = WQI,font=('times new roman' , 16 ) ,fg = "black",bg = "white").place(x=60, y=210)
        self.btn2 = Button(frame2,text="Return",command=self.logout_fn, bg='green', fg='white',font=('times new roman', 12), bd=0).place(x=150, y=280,width = 200)

    def Usage_of_water(self):
        plt.close()
        frame1 = Frame(self.frame, bg='white')
        frame1.place(x=100, y=0, width=500, height=600)
        Dummy_frame1 = Frame(frame1,bg='white')
        Dummy_frame1.place(x=50, y= 0, width=400 , height=200)
        Dummy_frame2 = Frame(frame1,bg='white')
        Dummy_frame2.place(x=50, y= 200, width=400 , height=200)
        Dummy_frame3 = Frame(frame1,bg='white')
        Dummy_frame3.place(x=50, y= 400, width=400 , height=200)
        if self.agg>=0 and self.agg<=25:
            # print("Excellent water quality")
            
            self.bg1 = ImageTk.PhotoImage(file="images/drink-water.png")
            bg1 = Label(Dummy_frame1 , image=self.bg1,background="white").place(x=0, y= 0,  relwidth=1 , relheight=1)
            Label(Dummy_frame1 , text="Drinking Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)
            self.bg2 = ImageTk.PhotoImage(file="images/water-filter.png")
            bg2 = Label(Dummy_frame2 , image=self.bg2,background="white").place(x=0, y=0, relwidth=1 , relheight=1)
            Label(Dummy_frame2 , text="Industry Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)
            self.bg3 = ImageTk.PhotoImage(file="images/watering.png")
            bg3 = Label(Dummy_frame3, image=self.bg3,background="white").place(x=0, y= 0, relwidth=1 , relheight=1)
            Label(Dummy_frame3 , text="Irrigation Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)

        elif self.agg>=26 and self.agg<=50:
            self.bg1 = ImageTk.PhotoImage(file="images/drink-water.png")
            bg1 = Label(Dummy_frame1 , image=self.bg1,background="white").place(x=0, y= 0,  relwidth=1 , relheight=1)
            Label(Dummy_frame1 , text="Drinking Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)
            self.bg2 = ImageTk.PhotoImage(file="images/water-filter.png")
            bg2 = Label(Dummy_frame2 , image=self.bg2,background="white").place(x=0, y=0, relwidth=1 , relheight=1)
            Label(Dummy_frame2 , text="Industry Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)
            self.bg3 = ImageTk.PhotoImage(file="images/watering.png")
            bg3 = Label(Dummy_frame3, image=self.bg3,background="white").place(x=0, y= 0, relwidth=1 , relheight=1)
            Label(Dummy_frame3 , text="Irrigation Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)

        elif self.agg>=51 and self.agg<=75:
            self.bg2 = ImageTk.PhotoImage(file="images/water-filter.png")
            bg2 = Label(Dummy_frame1 , image=self.bg2,background="white").place(x=0, y=0, relwidth=1 , relheight=1)
            Label(Dummy_frame1 , text="Industry Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)
            self.bg3 = ImageTk.PhotoImage(file="images/watering.png")
            bg3 = Label(Dummy_frame2, image=self.bg3,background="white").place(x=0, y= 0, relwidth=1 , relheight=1)
            Label(Dummy_frame2 , text="Irrigation Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)

        elif self.agg>=76 and self.agg<100:
            self.bg3 = ImageTk.PhotoImage(file="images/watering.png")
            bg3 = Label(frame1, image=self.bg3,background="white").place(x=0, y= 0, relwidth=1 , relheight=1)
            Label(frame1 , text="Irrigation Purpose",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)

        else:
            self.bg4 = ImageTk.PhotoImage(file="images/treatment.png")
            bg = Label(frame1 , image=self.bg4,background="white").place(x=0, y=0 , relwidth=1 , relheight=1)
            Label(frame1 , text="Proper Treatment of Water is Required",fg="black",bg="white",font=('times new roman', 12),justify=CENTER, bd=0).place(x=0, y= 170)

    def WQT_index(self):
        split_factor = 0.75
        K = 3
        phi = 2.05
        vel_fact = 0.5
        conf_type = 'RB'
        IntVar = None
        normalize = False
        rad = 0.1
        mu_delta = 0.2
        s_par = [0.5, 0.2]
        c_par = [1.0, 3.0]
        A_par = [-10.0, 10.0]

        data_file = pd.read_csv('Sample_test.csv',skiprows = 1,header = None).astype(float)
        data = data_file.iloc[:,:-1].to_csv('ran.csv',header=None,index = False)
        # data_file = pd.read_csv('User_test.csv',skiprows = 1,header = None).astype(float)
        # data = data_file.iloc[:,:-1].to_csv('ran.csv',header=None,index = False)
        xy = 'ran.csv'
        datav = np.loadtxt(xy, dtype=str, delimiter=',')
        n_samples, n_cols = datav.shape
        problem = 'C'
        n_mf = [1, 1, 1, 1, 1, 1]
        nPop = 40
        epochs = 500
        if (problem == 'C'):
            n_inputs = n_cols - 1
            n_outputs, class_list = utl.get_classes(datav[:, -1])

        # Continuous problem (the label columns are always at the end)
        else:
            n_inputs = len(n_mf)
            n_outputs = n_cols - n_inputs
        n_pf, n_cf, n_var = utl.info_anfis(n_mf, n_outputs)
        rows_tr = int(split_factor * n_samples)
        rows_te = n_samples - rows_tr
        idx_tr = np.random.choice(np.arange(n_samples), size=rows_tr, replace=False)
        idx_te = np.delete(np.arange(n_samples), idx_tr)
        data_tr = datav[idx_tr, :].astype(float)
        data_te = datav[idx_te, :].astype(float)
        
        # Split the data
        X_tr = data_tr[:, 0:n_inputs]
        Y_tr = data_tr[:, n_inputs:]
        X_te = data_te[:, 0:n_inputs]
        Y_te = data_te[:, n_inputs:]

        # System info
        print("\nNumber of samples = ", n_samples)
        print("Number of inputs = ", n_inputs)
        print("Number of outputs = ", n_outputs)

        if (problem == 'C'):
            print("\nClasses: ", class_list)

        print("\nNumber of training samples = ", rows_tr)
        print("Number of test samples= ", rows_te)

        print("\nANFIS layout = ", n_mf)
        print("Number of premise functions = ", n_pf)
        print("Number of consequent functions = ", n_cf)
        print("Number of variables = ", n_var)

        def interface_PSO(theta, args):
            """
            Function to interface the PSO with the ANFIS. Each particle has its own
            ANFIS instance.

            theta           (nPop, n_var)
            learners        (nPop, )
            J               (nPop, )
            """
            args_PSO = (args[0], args[1])
            learners = args[2]
            nPop = theta.shape[0]

            J = np.zeros(nPop)
            for i in range(nPop):
                J[i] = learners[i].create_model(theta[i, :], args_PSO)

            return J

        # Init learners (one for each particle)
        learners = []
        for i in range(nPop):
            learners.append(anf.ANFIS(n_mf=n_mf, n_outputs=n_outputs, problem=problem))

        # Always normalize inputs
        Xn_tr, norm_param = utl.normalize_data(X_tr)
        Xn_te = utl.normalize_data(X_te, norm_param)

        # Build boundaries using heuristic rules
        LB, UB = utl.bounds_pso(Xn_tr, n_mf, n_outputs, mu_delta=mu_delta, s_par=s_par,
                                c_par=c_par, A_par=A_par)

        # Scale output(s) in continuous problems to reduce the range in <A_par>
        if (problem != 'C'):
            Y_tr, scal_param = utl.scale_data(Y_tr)
            Y_te = utl.scale_data(Y_te, scal_param)

        func = interface_PSO
        args = (Xn_tr, Y_tr,learners)
        theta, info = pso.PSO(func, LB, UB, nPop=nPop, epochs=epochs, K=K, phi=phi,
                            vel_fact=vel_fact, conf_type=conf_type, IntVar=IntVar,
                            normalize=normalize, rad=rad, args=args)

        # ======= Solution ======= #
        best_learner = learners[info[1]]
        mu, s, c, A = best_learner.param_anfis()

        print("\nSolution:")
        print("J minimum = ", info[0])
        print("Best learner = ", info[1])
        print("Close learners = ", info[2])

        print("\nCoefficients:")
        print("mu = ", mu)
        print("s  = ", s)
        print("c  = ", c)
        print("A  = ", A)

        Yp_tr = best_learner.eval_data(Xn_tr)
        Yp_te = best_learner.eval_data(Xn_te)
        Yp_te_res = 40
        # Results for classification problems (accuracy and correlation)
        WQI_Train = 0
        WQI_Test = 0
        if (problem == 'C'):
            
            print("\nAccuracy data = ", utl.calc_accu(Yp_tr, Y_tr))
            #print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
            WQI_Train = "Accuracy training data = " + str(float("{0:.2f}".format(utl.calc_accu(Yp_tr, Y_tr))))
            # print("\nAccuracy test data = ", utl.calc_accu(Yp_te, Y_te))
            WQI_Test = "Accuracy test data = " + str(float("{0:.2f}".format(utl.calc_accu(Yp_te, Y_te))))
            #print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))

        # Results for continuous problems (RMSE and correlation)
        else:
            print("\nRMSE training data = ", utl.calc_rmse(Yp_tr, Y_tr))
            print("Corr. training data = ", utl.calc_corr(Yp_tr, Y_tr))
            print("\nRMSE test data = ", utl.calc_rmse(Yp_te, Y_te))
            print("Corr. test data = ", utl.calc_corr(Yp_te, Y_te))
            # -------------------------------------------------------
        
    def plot(self):
        self.frame = Frame(self.root, bg='white')
        self.frame.place(x=650, y=50, width=650, height=600)
        self.Usage_of_water()

    def logout_fn(self) :
        self.frame_wqp.pack()
        self.frame.pack()
        os.remove('User_test.csv')
        self.txt_ph.delete(0,END)
        self.txt_hardness.delete(0,END)
        self.txt_solids.delete(0,END)
        self.txt_sulphate.delete(0,END)
        self.txt_conductivity.delete(0,END)
        self.txt_turbidity.delete(0,END)
        
    

root = Tk()
obj = home_page(root)
root.mainloop()