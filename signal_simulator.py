import numpy as np
from scipy.stats import truncnorm


class signal_simulator():
    def __init__(self,shortParsPath,longParsPath,rel_long_parsPath):
        self.short_dists,self.long_dists = self.make_distributions(shortParsPath,longParsPath)

        self.rel_long_params = np.load(rel_long_parsPath)
        self.rel_long_dists = self.help_dist(self.rel_long_params)

    def make_s(self,x,L,k,x0): #Acts weirdly when x values are low
        out = np.array(L/(1+np.exp(-k*(x-x0))))
        return out

    def find_closest_after(self,stepsize,peak):
        mySum=0
        i=0
        while mySum<peak:
            mySum+= stepsize
            i+=1
        return i

    def find_closest_prior(self,stepsize,peak):
        mySum=0
        i=0
        while mySum<peak:
            mySum+= stepsize
            i+=1
        return i-1

    def fit_s2(self,peak_points,amplitudes,fs,length,k,x0):
        stepSize = (length/fs)/length #Is 36 correct or should it be 35?
        begin = self.find_closest_after(stepSize,peak_points[0]) #Seems to be fine
        end = self.find_closest_prior(stepSize,peak_points[1])#Seems to be fine
        alt_x = np.arange(begin,end+1,1)
        L = amplitudes[1]-amplitudes[0]
        out = self.make_s(alt_x,L,k,x0/stepSize)
        return out

    def func_4parts2(self,peak_points,amplitudes,ks,x0s,fs,length):
        maxPoint = length-1
        stepSizer = (maxPoint/120)/maxPoint
        overlap_points = [x*stepSizer for x in range(length)]
        out = np.zeros(length*6)
        startInd=0
        curSum=0
        minuser = 0
        for i in range(1,5):
            newPeaks = [curSum,curSum+peak_points[i]]
            newX0 = newPeaks[0]+((newPeaks[1]-newPeaks[0])*x0s[i-1])
            part = self.fit_s2(newPeaks,amplitudes[i-1:i+1],fs,maxPoint,ks[i-1],newX0)
            partLen = len(part)
            part_shifted = part+amplitudes[i-1]
            if newPeaks[0] in overlap_points and i>1:
                minuser = 1
            #Need some kind of overlap handling
            #out[startInd-minuser:startInd-minuser+partLen] = part_shifted #Old version with means
            if minuser>0:
                out[startInd:startInd-minuser+partLen] = part_shifted[minuser:]#out[startInd-minuser]/2
            else:
                out[startInd:startInd+partLen] = part_shifted
            startInd -= minuser
            startInd += partLen
            curSum = newPeaks[1]
            minuser=0
        return out[:length]

    #Requires a list of 5 peak points, 5 amplitudes, 4 ks, and 4 x0s
    #First peak point should be 0, next 4 should all be compared to previous
    #x0s are relative, the percentage of the difference. 
    def intermediate(self,values,fs=120,length=36):
        return self.func_4parts2(values[:5],values[5:10],values[10:14],values[14:18],fs,length)

    ######################################################################################

    ######################################################################################

    def draw_wide_gaussian(self,inp):
        #Use scipy.stats.truncnorm
        #Calculate sigma based on distance to nearest bound somehow?
        mu = inp[0]
        bounds = np.array(inp[1:])
        bounds = np.array(bounds)
        sigma = np.min(np.abs(bounds-mu))/2
        lower = bounds[0]
        upper = bounds[1]
        distr = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        return distr

    def make_distributions(self,shortParsPath,longParsPath):
        shortParams = np.load(shortParsPath)
        longParams = np.load(longParsPath)
        short_dists = self.help_dist(shortParams)
        long_dists = self.help_dist(longParams)
        return short_dists,long_dists

    def help_dist(self,params):
        out = [self.draw_wide_gaussian(params[i,:]) for i in range(params.shape[0])]
        return out


    def drawN(self,n):
        shortPars,longPars = self.drawPars(n)
        xs = np.array([[self.intermediate(shortPars[i,:]),self.intermediate(longPars[i,:])] for i in range(n)])
        return xs

    def drawPars(self,n):
        short_pars = self.drawHalfPars(n,self.short_dists)
        long_pars = self.drawHalfPars(n,self.long_dists)
        return short_pars,long_pars

    def drawHalfPars(self,n,dists):
        all_params = np.zeros((n,18))
        all_params[:,:5] = self.drawTimes(n,dists[0:3])
        all_params[:,5:10] = self.drawAmps(n,dists[3:6])
        all_params[:,10:14] = np.ones((n,4))
        all_params[:,14:18] = self.drawX0s(n,dists[6:10])
        return all_params

    def drawTimes(self,n,dists):
        times = np.zeros((n,5))
        for i in range(3):
            times[:,i+1] = dists[i].rvs(n)
            times = self.fixTimes(n,times,dists)
        for i in range(n):
            times[i,4] = 36/120-0.00001 - np.sum(times[i,:4])
        return times
        
    def fixTimes(self,n,times,dists):
        for i in range(n):
            if np.sum(times[i,:]) >= 0.25 or np.sum(times[i,:])<0.15:
                times[i,1:4] = self.draw1Times(dists)
        return times
                
    def draw1Times(self,dists):
        notReady = True
        newTimes = np.zeros(3)
        while notReady:
            for i in range(3):
                newTimes[i] = dists[i].rvs(1)
            if np.sum(newTimes) < 0.25 and np.sum(newTimes)>0.15:
                notReady=False
        return newTimes

    def drawAmps(self,n,dists):
        amps = np.zeros((n,5))
        for i in range(3):
            amps[:,i+1] = dists[i].rvs(n)
        return amps

    def drawX0s(self,n,dists):
        x0s = np.zeros((n,4))
        for i in range(4):
            x0s[:,i] = dists[i].rvs(n)
        return x0s


    ##############################
    #Relative drawing of the long params:
    ##############################

    def drawN_rel(self,n):
        shortPars,longPars = self.drawPars_rel(n)
        #rel_longPars = self.drawPars_rel(n)
        xs = np.array([[self.intermediate(shortPars[i,:]),self.intermediate_rel(longPars[i,:])] for i in range(n)])
        return xs

    def drawPars_rel(self,n):
        short_pars = self.drawHalfPars(n,self.short_dists)
        long_pars = self.drawLong_relPars(n,short_pars, self.long_dists, self.rel_long_dists)
        return short_pars,long_pars

    def drawLong_relPars(self,n,short_pars,dists,rel_dists):
        all_params = np.zeros((n,18))
        all_params[:,:5] = self.drawRelTimes(n,short_pars,rel_dists[0:3])
        all_params[:,5:10] = self.drawRelAmps(n,short_pars,rel_dists[3:6])
        all_params[:,10:14] = np.ones((n,4))
        all_params[:,14:18] = self.drawX0s(n,dists[6:10])
        return all_params

    def drawRelTimes(self,n,short_pars,rel_dists):
        times = np.zeros((n,5))
        for i in range(3):
            times[:,i+1] = np.sum(short_pars[:,:i+1+1],axis=1)+rel_dists[i].rvs(n)
            times = self.fixRelTimes(n,times,short_pars,rel_dists)
        for i in range(n):
            times[i,4] = 36/120-0.00001 - times[i,3]
        return times
        
    def fixRelTimes(self,n,times,short_pars,rel_dists):
        for i in range(n):
            if times[i,3] >= 0.26 or times[i,3]<0.15:
                times[i,1:4] = self.draw1TimesRel(i,short_pars,rel_dists)
        return times
                
    def draw1TimesRel(self,index,short_pars,rel_dists):
        notReady = True
        newTimes = np.zeros(3)
        while notReady:
            for i in range(3):
                newTimes[i] = np.sum(short_pars[index,:i+1+1])+rel_dists[i].rvs(1)
            if newTimes[2] < 0.26 and newTimes[2]>0.15:
                notReady=False
        return newTimes

    def drawRelAmps(self,n,short_pars,rel_dists):
        amps = np.zeros((n,5))
        for i in range(3):
            amps[:,i+1] = short_pars[:,5+i+1]*rel_dists[i].rvs(n)
        return amps

    ######################
    #Lastly the simulation functions for the relative simulation
    ######################


    def func_4parts2_rel(self,peak_points,amplitudes,ks,x0s,fs,length):
        maxPoint = length-1
        stepSizer = (maxPoint/120)/maxPoint
        overlap_points = [x*stepSizer for x in range(length)]
        out = np.zeros(length*6)
        startInd=0
        curSum=0
        minuser = 0
        for i in range(1,5):
            newPeaks = [curSum,peak_points[i]]
            newX0 = newPeaks[0]+((newPeaks[1]-newPeaks[0])*x0s[i-1])
            part = self.fit_s2(newPeaks,amplitudes[i-1:i+1],fs,maxPoint,ks[i-1],newX0)
            partLen = len(part)
            part_shifted = part+amplitudes[i-1]
            if newPeaks[0] in overlap_points and i>1:
                minuser = 1
            #Need some kind of overlap handling
            #out[startInd-minuser:startInd-minuser+partLen] = part_shifted #Old version with means
            if minuser>0:
                out[startInd:startInd-minuser+partLen] = part_shifted[minuser:]#out[startInd-minuser]/2
            else:
                out[startInd:startInd+partLen] = part_shifted
            startInd -= minuser
            startInd += partLen
            curSum = newPeaks[1]
            minuser=0
        return out[:length]

    #Requires a list of 5 peak points, 5 amplitudes, 4 ks, and 4 x0s
    #First peak point should be 0, next 4 should all be compared to previous
    #x0s are relative, the percentage of the difference. 
    def intermediate_rel(self,values,fs=120,length=36):
        return self.func_4parts2_rel(values[:5],values[5:10],values[10:14],values[14:18],fs,length)
    