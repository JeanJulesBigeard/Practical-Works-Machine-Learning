import math
from math import sin,cos,atan2
from math import pi,ceil
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand,randn
from numpy import diag
import random
from numpy.linalg import norm,inv,eig

#%-------- Drawing Covariance -----%
def PlotEllipse(x,P,nSigma):
    eH = []
    P = P[0:2,0:2] #% only plot x-y part
    x = x[0:2]
    plt.plot(XStore[0, :], XStore[1, :], ".k")
    if (not np.all(diag(P))):
        D,V = eig(P)
#        y = nSigma*[cos(0:0.1:2*pi);sin(0:0.1:2*pi)];
#        el = V*sqrtm(D)*y;
#        el = [el el(:,1)]+repmat(x,1,size(el,2)+1);
#        eH = line(el(1,:),el(2,:));
    return eH
    
def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[math.cos(angle), math.sin(angle)],
                    [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0]).flatten()
    py = np.array(fx[1, :] + xEst[1]).flatten()
    plt.plot(px, py, "--r")

def DoVehicleGraphics(x,P,nSigma,Forwards):
    #plt.cla()
    ShiftTheta = atan2(Forwards[1],Forwards[0])
    # h = PlotEllipse(x,P,nSigma)
    plot_covariance_ellipse(x,P)
    #    set(h,'color','r');
    DrawRobot(x,'b',ShiftTheta);
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)


#%-------- Drawing Vehicle -----%
def DrawRobot(Xr,col,ShiftTheta):
    p=0.02 # % percentage of axes size
#    a=axis;
#    l1=(a(2)-a(1))*p;
#    l2=(a(4)-a(3))*p;
#    P=[-1 1 0 -1; -1 -1 3 -1];%basic triangle
#    theta = Xr(3)-pi/2+ShiftTheta;%rotate to point along x axis (theta = 0)
#    c=cos(theta);
#    s=sin(theta);
#    P=[c -s; s c]*P; %rotate by theta
#    P(1,:)=P(1,:)*l1+Xr(1); %scale and shift to x
#    P(2,:)=P(2,:)*l2+Xr(2);
#    H = plot(P(1,:),P(2,:),col,'LineWidth',0.1);% draw
#    plot(Xr(1),Xr(2),sprintf('%s+',col));


#
def DoGraphs(InnovStore,PStore,SStore,XStore,XErrStore):
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
    ax1.plot(InnovStore[0,:])
    #ax1.plot(SStore[0,:],'r')
    #ax1.plot(-SStore[0,:],'r')
    ax2.plot(InnovStore[1,:]*180/pi)
    fig1.show()
    input("Press Enter to continue...")

#
#    figure(1) print -depsc 'EKFLocation.eps'
#
#    figure(2)
#    subplot(2,1,1)plot(InnovStore(1,:))hold onplot(SStore(1,:),'r')plot(-SStore(1,:),'r')
#    title('Innovation')ylabel('range')
#    subplot(2,1,2)plot(InnovStore(2,:)*180/pi)hold onplot(SStore(2,:)*180/pi,'r')plot(-SStore(2,:)*180/pi,'r')
#    ylabel('Bearing (deg)')xlabel('time')
#    print -depsc 'EKFLocationInnov.eps'
#
#    figure(2)
#    subplot(3,1,1)plot(XErrStore(1,:))hold onplot(3*PStore(1,:),'r')plot(-3*PStore(1,:),'r')
#    title('Covariance and Error')ylabel('x')
#    subplot(3,1,2)plot(XErrStore(2,:))hold onplot(3*PStore(2,:),'r')plot(-3*PStore(2,:),'r')
#    ylabel('y')
#    subplot(3,1,3)plot(XErrStore(3,:)*180/pi)hold onplot(3*PStore(3,:)*180/pi,'r')plot(-3*PStore(3,:)*180/pi,'r')
#    ylabel('\theta')xlabel('time')
#    print -depsc 'EKFLocationErr.eps'

#
def GetObservation(k):
    global Mapglobal, xTrueglobal, PYTrueglobal, nSteps
    if (k>600 and k<900):
        z = None
        iFeature=-1
    else:
        iFeature = random.randint(0,Map.shape[1]-1)
        z = DoObservationModel(xTrue, iFeature,Map)+np.sqrt(PYTrue)@randn(2)
        z[1] = AngleWrap(z[1])
    return [z,iFeature]

def DoObservationModel(xVeh, iFeature,Map):
    Delta = Map[0:2,iFeature-1]-xVeh[0:2]
    z = np.array([norm(Delta),
        atan2(Delta[1],Delta[0])-xVeh[2]])
    z[1] = AngleWrap(z[1])
    return z

def AngleWrap(a):
    if (a>np.pi):
        a=a-2*pi
    elif (a<-np.pi):
        a = a+2*pi;
    return a

#
def SimulateWorld(k):
    global xTrue
    u = GetRobotControl(k)
    xTrue = tcomp(xTrue,u)
    xTrue[2] = AngleWrap(xTrue[2])

#
def GetOdometry(k):
    global LastOdom #internal to robot low-level controller
    global QTrue
    global xTrue
    if(LastOdom is None):
        LastOdom = xTrue
    u = GetRobotControl(k)
    xnow = tcomp(LastOdom,u)
    uNoise = np.sqrt(QTrue) @ randn(3)
    xnow = tcomp(xnow,uNoise)
    LastOdom = xnow
    return xnow

# construct a series of odometry measurements
def GetRobotControl(k):
    global nSteps
    u = np.array([0, 0.025,  0.1*np.pi/180*math.sin(3*np.pi*k/nSteps)])
    #u = [0 0.15  0.3*pi/180]
    assert u.ndim == 1
    return u

# Functions to be completed

# h(x) Jacobian
def GetObsJac(xPred, iFeature,Map):
    jH = np.zeros((2,3))
    Delta = (Map[0:2,iFeature]-xPred[0:2])
    r = norm(Delta)
    jH[0,0] = -Delta[0] / r
    jH[0,1] = -Delta[1] / r
    jH[1,0] = Delta[1] / (r**2)
    jH[1,1] = -Delta[0] / (r**2)
    jH[1,2] = -1
    return jH

# f(x,u) Jacobian # x
def A(x,u):
    s1 = math.sin(x[2])
    c1 = math.cos(x[2])

    Jac  = np.array([[1, 0, -u[0]*s1-u[1]*c1],
            [0, 1, u[0]*c1-u[1]*s1],
            [0, 0, 1]])
            
    return Jac
    
# f(x,u) Jacobian # u
def B(x,u):
    s1 = sin(x[2])
    c1 = cos(x[2])

    Jac  = np.array([[c1, -s1, 0],
            [s1, c1, 0],
            [0, 0, 1]])

    return Jac

def tinv(tab):
    assert tab.ndim == 1
    tba = 0.0*tab;
    for t in range(0,tab.shape[0],3):
       tba[t:t+3] = tinv1(tab[t:t+3])
    assert tba.ndim == 1
    return tba

def tinv1(tab):
    assert tab.ndim == 1
    # calculates the inverse of one transformations
    s = math.sin(tab[2])
    c = math.cos(tab[2])
    tba = np.array([-tab[0]*c - tab[1]*s,
            tab[0]*s - tab[1]*c,
           -tab[2]])
    assert tba.ndim == 1
    return tba
    
def tcomp(tab,tbc):
# composes two transformations
    assert tab.ndim == 1
    assert tbc.ndim == 1
    result = tab[2]+tbc[2]

    #result = AngleWrap(result)
 
    s = sin(tab[2])
    c = cos(tab[2])
    # print(np.array([[c, -s],[s, c]]) , tbc, tbc[0:2])
    tac = tab[0:2]+ np.array([[c, -s],[s, c]]) @ tbc[0:2]
    tac = np.append(tac,result)
    #print("tcomp:",tab,tbc,tac)
    return tac

LastOdom = None

# change this to see how sensitive we are to the number of particle
# (hypotheses run) especially in relation to initial distribution!
nParticles = 400;

nSteps = 1000
# Location of beacons
Map = 140*rand(2,30)-70
#Map = 140*rand(2,10)-70
#Map = 140*rand(2,1)-70

# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.01,0.01,1*math.pi/180]) ** 2
PYTrue = diag([2.0,3*math.pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst = np.eye(3,3) @ QTrue
PYEst = 1e0 * np.eye(2,2) @ PYTrue

xTrue = np.array([1,-40,-math.pi/2])
xOdomLast = GetOdometry(0)

##initial conditions:
#xEst = xTrue
#xEst = np.array([0,0,0])
PEst = 10*np.diag([1,1,(1*math.pi/180)**2])

# initial conditions: - a point cloud around truth
xPred = xTrue[:,np.newaxis]+diag([8,8,0.4]) @ randn(3,nParticles);

# uniform random initial point
#xPred = 140*rand(3,nParticles)-70
#xPred[2] = 2.0*pi*rand(nParticles)-pi

#  storage  #
InnovStore = np.nan*np.zeros((2,nSteps))
SStore = np.NaN*np.zeros((2,nSteps))
PStore = np.NaN*np.zeros((3,nSteps))
XStore = np.NaN*np.zeros((3,nSteps))
XErrStore = np.NaN*np.zeros((3,nSteps))

#initial graphics
plt.cla()
plt.plot(Map[0,:],Map[1,:],'.g')
#hObsLine = line([0,0],[0,0])
#set(hObsLine,'linestyle',':')

for k in range(1,nSteps):
    
    #do world iteration
    SimulateWorld(k)
    
    #all particles are equally important
    L = np.ones((nParticles))/nParticles
    
    #figure out control by subtracting current and previous odom values
    xOdomNow = GetOdometry(k)
    #print("Odom: ",xOdomNow,xOdomLast)
    u = tcomp(tinv(xOdomLast),xOdomNow)
    xOdomLast = xOdomNow
    
    # do prediction
    # for each particle we add in control vector AND noise
    # the control noise adds diversity within the generation
    for p in range(nParticles):
        xPred[:,p] = tcomp(xPred[:,p].squeeze(),u+np.squeeze(np.sqrt(QEst) @ randn(3,1)))
        
    #xPred[2] = AngleWrap(xPred[2])
        
    #observe a randomn feature
    [z,iFeature] = GetObservation(k)
        
    if z is not None:
        #predict observation
        for p in range(nParticles):
            zPred = DoObservationModel(xPred[:,p],iFeature,Map)
        
            #how different
            Innov = z-zPred
            #get likelihood (new importance). Assume gaussian here but any pdf works!
            #if predicted obs is very different from actual obs this score will be low
            #->this particle is not very good at representing state. A lower score means
            #it is less likely to be selected for the next generation...
            L[p] = np.exp(-0.5*Innov[np.newaxis,:] @ np.linalg.inv(PYEst) @ Innov[:,np.newaxis]) + 0.001
        #print("Weights: ",L)

    # Compute position as weighted mean of particles
    xEst = np.average(xPred,axis=1,weights=L)
    # squaredError = (xP-xEst).*(xP-xEst);
    # xVariance= [xVariance sqrt(mean(squaredError(1,:)+squaredError(2,:)))];

    
    # reselect based on weights:
    # particles with big weights will occupy a greater percentage of the
    # y axis in a cummulative plot
    CDF = np.cumsum(L)/np.sum(L)
    # so randomly (uniform) choosing y values is more likely to correspond to
    # more likely (better) particles...
    iSelect  = rand(nParticles)
    # find the particle that corresponds to each y value (just a look up)
    iNextGeneration = np.interp(iSelect,CDF,range(nParticles),left=0).astype(int).ravel()
    # print(iNextGeneration,iSelect,CDF,range(nParticles))
    # copy selected particles for next generation...
    xPred = xPred[:,iNextGeneration]
    L = L[iNextGeneration]
            
    # plot every 200 updates
    if (k)%(10)==0:
        print("max weight: ",max(L))
        DoVehicleGraphics(xEst,PEst[0:2,0:2],8,[0,1])
        plt.plot(xPred[0],xPred[1],'.b')
        if z is not None:
            plt.plot(Map[0,iFeature],Map[1,iFeature],'.g')

#            set(hObsLine,'XData',[xEst[0],Map[0,iFeature]])
#            set(hObsLine,'YData',[xEst[1],Map[1,iFeature]])
            pass
#        drawnow
    
    #store results:
    InnovStore[:,k] = Innov
    #PStore[:,k] = np.sqrt(diag(PEst))
    #SStore[:,k] = np.sqrt(diag(S))
    XStore[:,k] = xEst
    XErrStore[:,k] = xTrue-xEst

DoGraphs(InnovStore,PStore,SStore,XStore,XErrStore)


