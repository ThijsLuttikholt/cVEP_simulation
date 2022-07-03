import os
import numpy as np
import random
import math

from scipy.special import gamma as funGamma
from sklearn.cross_decomposition import CCA
from tqdm import tqdm


from scipy.stats import (
    norm, gamma
    )

from .signal_simulator import signal_simulator

class signal_gen_infer():
    def __init__(self, path="data", version=0,allChan=0):
        self.path = "data"
        self.subjects = ["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06",
            "sub-07","sub-08","sub-09","sub-10","sub-11","sub-12",
            "sub-13","sub-14","sub-15","sub-16","sub-17","sub-18",
            "sub-19","sub-20","sub-21","sub-22","sub-23","sub-24",
            "sub-25","sub-26","sub-27","sub-28","sub-29","sub-30"]
        
        self.norm_indices_short = [1,2]
        self.norm_indices_long = [2,3,7]

        self.norm_indices_short2 = []
        self.norm_indices_long2 = []
        self.allChan = allChan
        
        self.shortVals = np.load('distShortParams.npy')
        self.longVals = np.load('distLongParams.npy')

        self.shortVals2 = np.load('distShortParams2.npy')
        self.longVals2 = np.load('distLongParams2.npy')
        
        self.Xs = np.array([])
        self.ys = np.array([])
        
        self.full_short_int,self.full_long_int = self.full_templates(self.path,self.subjects) 
        self.full_short,self.full_long = self.changeSign(self.full_short_int,self.full_long_int)
        if version == 1:
            self.full_short = np.load('simShorts.npy')
            self.full_long = np.load('simLongs.npy')

        self.sig_sim = signal_simulator('custMan_pars_short.npy','custMan_pars_long.npy')

    def changeSign(self,full_r1,full_r2):
        results1 = np.zeros((len(full_r1),len(full_r1[0])))
        results2 = np.zeros((len(full_r2),len(full_r2[0])))
        av_1 = np.average(full_r1,axis=0)
        av_2 = np.average(full_r2,axis=0)
        for i in range(len(full_r1)):
            mult1 = 1
            mult2 = 1
            corr1 = np.correlate(full_r1[i],av_1)
            corr2 = np.correlate(full_r2[i],av_2)
            if corr1 < 0:
                mult1 = -1
            if corr2 < 0:
                mult2 = -1
        
            results1[i] = mult1 * full_r1[i]
            results2[i] = mult2 * full_r2[i]
        return results1,results2

    def load_data(self,path,subject,ind):
        # Load data
        fn = os.path.join(path, "derivatives", "offline", subject, f"{subject}_gdf.npz")
        tmp = np.load(fn)
        X = tmp["X"]
        y = tmp["y"]
        V = tmp["V"]
        self.y = y
        self.V = V
        
        fs = tmp["fs"]
        self.fs = fs
        
        #!!!!!!!!!!!!!!LOOK OUT, THIS IS HARDCODED !!!!!!!!!!!!!!!!!!!
        #It only currently uses the Oz channel

        if self.allChan==0:
            X = X[4,:,:]
            X = np.reshape(X,(1,X.shape[0],X.shape[1]))
        #X = X[4,:,:]
        #X = np.reshape(X,(1,X.shape[0],X.shape[1]))
        
        if len(self.Xs) == 0:
            self.Xs = np.zeros((len(self.subjects),X.shape[0],X.shape[1],X.shape[2]))
            self.Xs = self.Xs.astype(X.dtype)
        if len(self.ys) == 0:
            self.ys = np.zeros((len(self.subjects),y.shape[0]))
            self.ys = self.ys.astype(y.dtype)

        self.Xs[ind] = X
        self.ys[ind] = y
        
        # Extract data dimensions
        n_channels, n_samples, n_trials = X.shape
        n_classes = V.shape[1]
        self.n_classes = n_classes

        # Read cap file
        fn = os.path.join(path, "resources", "nt_cap8.txt")
        fid = open(fn, "r")
        channels = []
        for line in fid.readlines():
            channels.append(line.split()[0])
        
        return X,y,V,fs,n_channels,n_samples,n_trials,n_classes,channels

    def getRiseFall(self,V):
        # Get rising and falling edges
        V = V.astype("bool_")
        Vr = np.roll(V, 1, axis=0)
        Vr[0, :] = False  #To ensure that an initial flash is always counted as a rising edge
        rise = np.logical_and(V, np.logical_not(Vr)).astype("uint8") 
        fall = np.logical_and(np.logical_not(V), Vr).astype("uint8")
        return rise,fall

    def createEvent(self,V,rise,fall,n_classes):
        E = np.zeros((V.shape[0], 2, V.shape[1]), dtype="uint8")
        for i_class in range(n_classes):
            up = np.where(rise[:, i_class])[0]  #Results in a list of up indices
            down = np.where(fall[:, i_class])[0] #Results in a list of down indices
            if up.size > down.size:
                down = np.append(down, V.shape[0]) #Adding a down edge at the final point (as it should have)
            durations = down - up #Gets all individual durations
            unique_durations = np.unique(durations)    
            E[up, 0, i_class] = durations == unique_durations[0] #All events of duration type 1
            E[up, 1, i_class] = durations == unique_durations[1] #All events of duration type 2
        return E

    def createStructure(self,fs,V,E):
        # Create structure matrix
        n_samples_transient = int(0.3 * fs) #Ne*Nr,  0.3 is the length (in seconds) of a transient response. => 0.3*120=36 samples
        M1 = np.zeros((V.shape[0], n_samples_transient, V.shape[1]), dtype="uint8")
        M2 = np.zeros((V.shape[0], n_samples_transient, V.shape[1]), dtype="uint8")
        M1[:, 0, :] = E[:, 0, :] #Set initial equal to event matrix
        M2[:, 0, :] = E[:, 1, :] #Set initial equal to event matrix
        for i_sample in range(1, n_samples_transient):
            M1[:, i_sample, :] = np.roll(M1[:, i_sample-1, :], 1, axis=0)
            M1[0, i_sample, :] = 0  #Always start at 0, is carried over to following parts too
            M2[:, i_sample, :] = np.roll(M2[:, i_sample-1, :], 1, axis=0)
            M2[0, i_sample, :] = 0
        M = np.concatenate((M1, M2), axis=1)
        self.M = M
        self.n_samples_transient = n_samples_transient
        return M,n_samples_transient

    def getTransients(self,X,n_channels,M,y,n_samples_transient):
        X_ = np.reshape(X, (n_channels, -1)).T

        # Repeat the structure matrices to full single-trial length
        M_ = np.tile(M, (15, 1, 1))

        # Set the structure matrices in the order of single-trials
        M_ = M_[:, :, y]  #This is full-trial structure matrix X 72 (36 samples * 2 events) X 100 trials (20 trials * 5 runs)

        # Reshape the structure matrix
        M_ = M_.transpose((1, 0, 2)) #Becomes 72 X full-trial structure matrix (3780) X 100
        M_ = np.reshape(M_, (2*n_samples_transient, -1)).T #Becomes 378000 (concatenated structures across trials) X 72 

        # Fit CCA
        cca = CCA(n_components=1)
        cca.fit(X_.astype("float32"), M_.astype("float32"))

        # Extract learned filters
        w = cca.x_weights_.flatten()
        r = cca.y_weights_.flatten() #Size 72, meaning a template for the short and long events separately. 
        return w,r
    
    def full_1(self,path,subject,ind):
        X,y,V,fs,n_channels,n_samples,n_trials,n_classes,channels = self.load_data(path,subject,ind)
        rise,fall = self.getRiseFall(V)
        E = self.createEvent(V,rise,fall,n_classes)
        M, n_samples_transient = self.createStructure(fs,V,E)
        w,r = self.getTransients(X,n_channels,M,y,n_samples_transient) #w is the spatial filter here
        return r[:n_samples_transient],r[n_samples_transient:]

    def full_templates(self,path,subjects):
        numSub = len(subjects)
        for ind in tqdm(range(numSub)):
            r1,r2 = self.full_1(path,subjects[ind],ind)
            if ind==0:
                full_r1 = np.zeros((numSub,len(r1)))
                full_r2 = np.zeros((numSub,len(r2)))
            full_r1[ind,:] = r1
            full_r2[ind,:] = r2
        return full_r1,full_r2

    def getN(self,n,subjects=[]):
        if len(subjects)==0:
            options = [i for i in range(len(self.subjects))]
        else:
            options = subjects
        indices = random.choices(options,k=n)
        values = np.array([[self.full_short[i],self.full_long[i]] for i in indices])
        return values
    
    def getCodes(self):
        return self.V
    
    def getDurations(self):
        return np.array([2,4])

    def getTransientsOut(self):
        return self.full_short,self.full_long
    
    def get_needed_values(self):
        return self.Xs,self.ys,self.V,self.M,self.fs,self.n_classes,self.n_samples_transient

    def drawN(self,n,length=-1):
        if length <= 0:
            length = self.full_short.shape[1]
        params_short = self.drawSigParams(n,self.shortVals,self.norm_indices_short)
        params_long = self.drawSigParams(n,self.longVals,self.norm_indices_long)
        
        values = np.array([[self.genCustS4Adapt(params_short[:,i],length=length),
                            self.genCustS4Adapt(params_long[:,i],length=length)] for i in range(n)])
        return values
    
    def drawSigParams(self,n,inValues,norm_indices):
        values = np.zeros((8,n))
        for i in range(8):
            newData = self.distDraw(i,n,inValues,norm_indices)
            
            if i>0 and i<4:
                test = newData<values[i-1,:]
            else:
                test = np.zeros(n)
                
            values[i,:] = newData 
            pos_indices = [sub_i for sub_i, x in enumerate(test) if x==1]
            for pos_i in pos_indices:
                values[i,pos_i] = self.keepDraw(i,inValues,norm_indices,values[i-1,pos_i])
        
        return values
    
    def distDraw(self,ind,n,inValues,norm_indices):
        if ind in norm_indices:
            newData = norm.rvs(size=n,loc=inValues[ind,0],scale=inValues[ind,1])
        else:
            newData = gamma.rvs(inValues[ind,0],loc=inValues[ind,1],scale=inValues[ind,2],size=n)
        return newData
    
    #Might need a failsafe where the previous one is redrawn? Or just continue with latest draw
    def keepDraw(self,ind,inValues,norm_indices,minimum):
        draw = 0
        numDraws = 0
        while draw<minimum and numDraws<100:
            draw = self.distDraw(ind,1,inValues,norm_indices)
            numDraws += 1
        return draw
    
    
    ######################################################
    
    ######################################################
    def sub_gamma(self,alpha):
        return funGamma(alpha)

    def help_part(self,a,b,t):
        return (t**a) * (b**a) * (math.e**(-b*t))

    def genCustS4Adapt(self,values,length=36):

        num_extra = int(len(values)/2)
        alphas = values[:num_extra]
        #betas = values[num_extra:2*num_extra]
        betas = np.ones(num_extra)
        cs = values[num_extra:2*num_extra]
        #mult = values[-1]
    
        result=np.zeros(length)
        for i in range(1,length+1):
            partResult = cs[0]*self.help_part(alphas[0],betas[0],i)/self.sub_gamma(alphas[0])
            for i2 in range(1,num_extra):
                extraPart = self.help_part(alphas[i2],betas[i2],i)/self.sub_gamma(alphas[i2])
                partResult -= cs[i2]*extraPart
            result[i-1] = partResult
        return result

    #########################################
    #New drawing
    #########################################



    #This is the adapted version, using only 3 C values and a full multiplier. 
    def genCustS4Adapt2(self,values,amount=36):
        numAlpha = 4
        numC = 4
        alphas = values[:numAlpha]
        betas = np.ones(numAlpha)
        cs = values[numAlpha:numAlpha+numC]
        
        result=np.zeros(amount)
        for i in range(1,amount+1):
            partResult = cs[0]*self.help_part(alphas[0],betas[0],i)/self.sub_gamma(alphas[0]) 
            for i2 in range(1,numAlpha):
                #Can not directly use alphas[i2], need to calculate it
                newAlpha = alphas[0]+np.sum(alphas[1:i2+1])
                extraPart = self.help_part(newAlpha,betas[i2],i)/self.sub_gamma(newAlpha)
                partResult -= (cs[i2]*cs[i2-1])*extraPart
            result[i-1] = partResult
        return result

    def drawN2(self,n,length=-1):
        if length <= 0:
            length = self.full_short.shape[1]
        params_short = self.drawSigParams2(n,self.shortVals2,self.norm_indices_short2)
        params_long = self.drawSigParams2(n,self.longVals2,self.norm_indices_long2)
        
        values = np.array([[self.genCustS4Adapt2(params_short[:,i],amount=length),
                            self.genCustS4Adapt2(params_long[:,i],amount=length)] for i in range(n)])
        return values
    
    def drawSigParams2(self,n,inValues,norm_indices):
        values = np.zeros((8,n))
        for i in range(8):
            newData = self.distDraw2(i,n,inValues,norm_indices)
            
            if i<4:
                test = newData<values[i,:]
            else:
                test = np.zeros(n)
                
            values[i,:] = newData 
            pos_indices = [sub_i for sub_i, x in enumerate(test) if x==1]
            for pos_i in pos_indices:
                values[i,pos_i] = self.keepDraw2(i,inValues,norm_indices,0)
        
        return values
    
    def distDraw2(self,ind,n,inValues,norm_indices):
        if ind in norm_indices:
            newData = norm.rvs(size=n,loc=inValues[ind,0],scale=inValues[ind,1])
        else:
            newData = gamma.rvs(inValues[ind,0],loc=inValues[ind,1],scale=inValues[ind,2],size=n)
        return newData
    
    #Might need a failsafe where the previous one is redrawn? Or just continue with latest draw
    def keepDraw2(self,ind,inValues,norm_indices,minimum):
        draw = 0
        numDraws = 0
        while draw<minimum and numDraws<100:
            draw = self.distDraw2(ind,1,inValues,norm_indices)
            numDraws += 1
        return draw

    ############################

    ############################

    #The new drawN version
    def drawN3(self,n):
        draws = self.sig_sim.drawN(n)
        return draws