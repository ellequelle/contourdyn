"""script that calculate contour dynamics
Requires numpy and matplotlib.
For interactive plots macos install Python.app.

Based on contour dynamics toy model from Darryn Waugh.

2022 LE Hanson
"""

import numpy as np
import matplotlib.pyplot as plt


###########################
## Run options

# itest defines the situation simulated
# 1 one elliptical eddy of axes ration rat = b/a
# 2 three cocentric elliptical eddies  of axes ratio rat = b/a
# 3 thin layer
# 4 two circular eddies of radius 0.4 at distance dist
# 5 three eddies not overlapping
itest = 5

# plot options
do_interactive = True # show plots as script runs
verbose = True # print out extra information
label_patches = True # label vortex patch number
label_0_point = True # show a stationary point on the contour
show_points = True # show all of the points in the contour
vector_field = True # plot the velocity vector field
boundary_quivers = False # label flow directions along the contour
print_circulation = True # show total circulation of each patch on plot
print_circ_change = True # show net change in circulation of each patch on plot
print_time = True # show time on plot
time_as_steps = False # show time as timesteps not MM:SS
save_to_png = True # save plots as png files

# define some other things
# colors of patches
mycol = ['r', 'b', 'g', 'gray', 'pink']

# time variable
t = 0

###########################
## Define scenarios

if itest == 1:
    # one elliptical eddy of axes ratio rat = b/a
    # number of points on each contour
    ip = 300
    # time step
    dt = 0.025
    # how many timesteps between plots
    nfreq = 5
    tmax = 300
    # number of time steps
    # number of patches
    KM = 1
    # eddy shape
    rat = 7
    # mode four perturbation
    nlobes = 3
    pmag = 0.01

    x = np.zeros((ip+1,KM), dtype='float64')
    y = np.zeros((ip+1,KM), dtype='float64')

    omofp = [None]*KM

    itt = np.arange(0, ip+1)/ip*np.pi

    # define KM patches
    for k in range(0,KM):
        # define the points of the contour
        # coordinates of the contour points
        th = itt*2
        r = 1 + pmag*np.cos(th*nlobes)
        x[:,k] = 0.00 + r*np.cos(itt*2)
        y[:,k] = 0.00 + r/rat*np.sin(itt*2)

        # value of omega over 4 pi for the patch
        omofp[k] = 1/(rat*np.pi)#-1/(4*np.pi)
        # colour for the plotting for the patch
    omega = (omofp[0]*rat/(1+rat)**2)
    f = np.pi*2*omega
    T = np.pi*2/omega
    
elif itest == 2:
    # three concentric elliptical eddies  of axes ratio rat = b/a
    # For parameters, see explanation of itest = 1
    
    # how many timesteps between plots
    nfreq = 1
    ip = 200
    dt = 0.07
    tmax = 10
    KM = 3
    rat = 3.5
    rat = 2.5
    #rat = 4
    nlobes = 1
    pmag = 0.0

    x = np.zeros((ip+1,KM), dtype='float64')
    y = np.zeros((ip+1,KM), dtype='float64')
    omofp = [None]*KM
    #mycol = [None]*KM
    itt = np.arange(0, ip+1)/ip*np.pi
    for k in range(0,KM):
        th = itt*2
        r = 1 + pmag*np.cos(th*nlobes) - k/KM
        x[:,k] = r*np.cos(th)
        y[:,k] = r/rat*np.sin(th)
        #for i in range(0,ip+1):
        #    th = i/ip*2*np.pi
        #    r = 1+0.00*np.cos(th*nlobes)-(k-1)/KM
        #    x[i,k] = 1*r*np.cos(i/ip*2*np.pi)
        #    y[i,k] = 1/rat*r*np.sin(i/ip*2*np.pi)

        omofp[k] = 1/(4*np.pi)
        #mycol[k] = 'b'
    #mycol[1] = 'g'
    #mycol[2] = 'y'
    
elif itest == 3:
    # thin layer
    
    # how many timesteps between plots
    nfreq = 5
    ip = 200
    dt = 0.05
    tmax = 33
    KM = 3
    rat = 1
    nlobes = 10
    pmag = 0.0

    x = np.zeros((ip+1,KM), dtype='float64')
    y = np.zeros((ip+1,KM), dtype='float64')
    omofp = [None]*KM
    itt = np.arange(0, ip+1)/ip*np.pi
    for k in range(0,KM):
        th = itt*2
        r = 1 + pmag*np.cos(th*nlobes + (k-1)*np.pi/2) - 0.05*(k-1)
        x[:,k] = r*np.cos(th)
        y[:,k] = r/rat*np.sin(th)
        #for i in range(0,ip+1):
        #    th = i/ip*2*np.pi
        #    r = 1+0.003*np.cos(th*nlobes+(k-1)*np.pi/2)-0.05*(k-1)
        #    x[i,k] = 1*r*np.cos(i/ip*2*np.pi)
        #    y[i,k] = 1/rat*r*np.sin(i/ip*2*np.pi)
        omofp[k] = -1/(4*np.pi)
    #mycol[1] = 'b'
    #mycol[2] = 'w'
    omofp[2] = -omofp[0]
    
elif itest == 4:
    # two circular eddies of radius 0.4 at distance dist

    r = 0.4
    #dist = 1.4#r*2.1
    dist = r*2
    # how many timesteps between plots
    nfreq = 4
    ip = 300
    dt = 0.05
    tmax = 20
    KM = 2
    rat = 1

    x = np.zeros((ip+1,KM), dtype='float64')
    y = np.zeros((ip+1,KM), dtype='float64')
    omofp = [None]*KM
    #mycol = [None]*KM
    itt = np.arange(0, ip+1)/ip*np.pi
    for k in range(0,KM):
        th = itt*2
        #r = 0.4
        x[:,k] = r*np.cos(th) + dist*(k-0.5)
        y[:,k] = r/rat*np.sin(th)
        #for i in range(0,ip+1):
        #    th = i/ip*2*np.pi
        #    r = 0.4
        #    x[i,k] = 1*r*np.cos(i/ip*2*np.pi)+dist*(k-1.5)
        #    y[i,k] = 1/rat*r*np.sin(i/ip*2*np.pi)
        omofp[k] = 1/(4*np.pi)#*(k-0.35)
    
elif itest == 5:
    
    dist=.8
    # how many timesteps between plots
    nfreq = 1
    ip = 75
    dt = 0.15
    ip = 300
    tmax = 90
    dt = 0.05
    KM = 3
    rat = 1

    x = np.zeros((ip+1,KM), dtype='float64')
    y = np.zeros((ip+1,KM), dtype='float64')
    omofp = [None]*KM
    itt = np.arange(0, ip+1)/ip*np.pi
    for k in range(0,KM):
        th = itt*2
        r = 0.3
        if k == 2:
            r = 0.3*np.sqrt(2)
        x[:,k] = r*np.cos(th) + dist*(k-2)
        y[:,k] = r/rat*np.sin(th)
        #for i in range(0,ip+1):
        #    th = i/ip*2*np.pi
        #    r = 0.3
        #    if k == 2:
        #        r = 0.3*np.sqrt(2)
        #    x[i,k] = 1*r*np.cos(i/ip*2*np.pi)+dist*(k-2)
        #    y[i,k] = 1/rat*r*np.sin(i/ip*2*np.pi)

        omofp[k] = 1/(4*np.pi)

    omofp[2] = -omofp[2]

##############################################
## Define some functions

def contourintdef(ip,x,y):
    """Defines central point of a segment and corresponding segment dx and dy
    assumes that on entry
      x(ip+1)=x(1)
      y(ip+1)=y(1)"""
    
    # Allocate
    if True:
        xx = np.zeros((4,ip), dtype='float64')
        xx[0,:] = np.convolve(x, [1, 1])[1:-1]/2
        xx[1,:] = np.convolve(y, [1, 1])[1:-1]/2
        xx[2,:] = np.diff(x)
        xx[3,:] = np.diff(y)
        return xx

    else:
        xm = np.zeros((ip,1))
        ym = xm.copy()
        dx = xm.copy()
        dy = xm.copy()


        # Now fill in
        for i in range(0,ip):
            xm[i] = (x[i]+x[i+1])*0.5
            ym[i] = (y[i]+y[i+1])*0.5
            dx[i] = x[i+1]-x[i]
            dy[i] = y[i+1]-y[i]
        return xm,ym,dx,dy

conv_med = np.array([1,1])*0.5
def xmdx(xx,yy):
    xm = np.convolve(xx, conv_med)[1:-1]
    ym = np.convolve(yy, conv_med)[1:-1]
    dx = np.diff(xx)
    dy = np.diff(yy)
    return (xm,ym,dx,dy)

def xmdx_ext(xx,yy):
    sh = (xx.shape[0]-1, xx.shape[1])
    xm = np.zeros(sh, dtype='float64')
    ym = np.zeros(sh, dtype='float64')
    dx = np.zeros(sh, dtype='float64')
    dy = np.zeros(sh, dtype='float64')
    for k in range(xx.shape[1]):
        xm[:,k] = np.convolve(xx[:,k], conv_med)[1:-1]
        ym[:,k] = np.convolve(yy[:,k], conv_med)[1:-1]
        dx[:,k] = np.diff(xx[:,k])
        dy[:,k] = np.diff(yy[:,k])
    return (xm,ym,dx,dy)

def xy0int(xm, ym, dx, dy, xi, yi):
    """for a single k"""
    dist = (xi-xm)**2 + (yi-ym)**2
    cof = np.log(dist)
    return np.sum(dx*cof), np.sum(dy*cof)

def contourint(ip,xm,ym,dx,dy,xi,yi):
    """Calculate the contribution of contour defined by xm, ym with segments dx dy
    to the velocity diagnose in point xi,yi.

    Assumes that on entry
     - x(ip+1)=x(1)
     - y(ip+1)=y(1)"""

    # Integrate over all points of contour xm,ym 
    dist = (xi-xm)**2 + (yi-ym)**2
    cof = np.log(dist)

    return np.sum(cof*dx), np.sum(cof*dy)

def vorttovel(ip,KM,x,y,omofp):
    """From vorticity patches to velocities
    There are KM patches, each of which defined by ip points
    """
    
    # Declare arrays with zeros
    u = np.zeros((ip,KM), dtype='float64')
    v = u.copy()
    myi = u.copy()
    myj = u.copy()
    
    old = False
    if old:
        xm = u.copy()
        ym = u.copy()
        dx = u.copy()
        dy = u.copy()

    # initialize
    xx = np.zeros((ip,1), dtype='float64')
    yy = xx.copy()
    zz = xx.copy()
    ww = xx.copy()

    if not old:
        xm, ym, dx, dy = xmdx_ext(x,y)
    else:
        for k in range(0,KM):
            # For each patch, calculate the mid point of segments and dx dy values
            xx, yy, zz, ww = contourintdef(ip,x[:,k],y[:,k])
            xm[:,k] = xx.flatten()
            ym[:,k] = yy.flatten()
            dx[:,k] = zz.flatten()
            dy[:,k] = ww.flatten()

    for k in range(0,KM):
        # For each contour make the integral with all other contours
        for m in range(0,KM):
            # For eack point on the contour k, get the contribution from all other contours (m)
            for i in range(0,ip):
                myi[i,m], myj[i,m] = xy0int(xm[:,m],ym[:,m],dx[:,m],dy[:,m],x[i,k],y[i,k])

        for m in range(0,KM):
        # add the contribution of each of the contours to the velocity field of each point in contour k
            u[:,k] = u[:,k]-omofp[m]*myi[:,m]
            v[:,k] = v[:,k]-omofp[m]*myj[:,m]
    return u,v

def updatex(ip,KM,x,y,u,v,dt):
    """Simple Euler displacement for all points using the velocities u and v
    """
    # for k=1:KM
    #     for i=1:ip
    #         xn(i,k)=x(i,k)+u(i,k)*dt
    #         yn(i,k)=y(i,k)+v(i,k)*dt

    # The following should be faster than a double loop
    xn[0:ip,:] = x[0:ip,:]+u*dt
    yn[0:ip,:] = y[0:ip,:]+v*dt
    xn[ip,:] = xn[0,:]
    yn[ip,:] = yn[0,:]
    return xn, yn


def redis_points(x, y):
    """redistribute points along boundary"""
    ll = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
    xnew = x.copy()
    ynew = y.copy()
    #print('pathlength: {:0.3f}, {:0.3f}'.format(*np.sum(ll, axis=0)))
    lx = np.cumsum(ll, axis=0) - ll[0,:]
    from scipy import interpolate
    for k in range(ll.shape[1]):
        fx = interpolate.interp1d(lx[:,k], x[:-1,k], kind='linear', axis=0)
        fy = interpolate.interp1d(lx[:,k], y[:-1,k], kind='linear', axis=0)
        lxnew = np.linspace(0, lx[-1,k], len(ll))
        xnew[:-1,k] = fx(lxnew)
        ynew[:-1,k] = fy(lxnew)
        #xnew[:-1,k] = np.interp(lxnew, lx[:,k], x[:-1,k])
        #ynew[:-1,k] = np.interp(lxnew, lx[:,k], y[:-1,k])
    xnew[-1,:] = xnew[0,:]
    ynew[-1,:] = ynew[0,:]
    return xnew, ynew
    


#################################################
## From here generic code

#Close contour with reduntant point, also useful for integration
#for k in range(0,KM):
#    x[ip+1,k] = x[0,k]
#    y[ip+1,k] = y[0,k]
x[ip,:] = x[0,:]
y[ip,:] = y[0,:]

#allocate tables
u = np.zeros((ip,KM), dtype='float64')
v = np.zeros((ip,KM), dtype='float64')

# Initialize old values
xo = x
yo = y
xn = x
yn = y

# number of time steps
ntot = int(tmax/dt)
# start at -dt time so it plots after first step
t = -dt

xall = np.zeros((ip+1,KM,ntot//nfreq+1), dtype='float64')
yall = np.zeros((ip+1,KM,ntot//nfreq+1), dtype='float64')

# use this as an event listener singleton
class IsClosed():
    plot_closed = False
    @classmethod
    def set_closed(cls, *args):
        cls.plot_closed = True
isClosed = IsClosed()

# close any existing figures
plt.close('all')
if do_interactive:
    plt.ion()
else:
    plt.ioff()

# make figure
fig = plt.figure()
ax = plt.gca()

# add event listener to detect if when figure is closed
fig.canvas.mpl_connect("close_event", isClosed.set_closed)

print('calculating...')

# get minimum and maximum, with limits a and b
minmax = lambda x,a,b: [min(np.min(x),a), max(np.max(x),b)]

# set initial axes limits
axlim0 = minmax(x,-1.5,1.5) + minmax(y,-1.5,1.5)

# ntot time step
for n in range(0,ntot+1):
            
    x0 = x.copy()
    y0 = y.copy()

    #Predictor 
    # Diagnose velocity from vorticity
    u, v = vorttovel(ip,KM,x,y,omofp)
    # Move contour points with this velocity
    xn, yn = updatex(ip,KM,x,y,u,v,dt)
    # Store first guess of velocity
    un = u.copy()
    vn = v.copy()

    #Corrector
    # Diagnose velocity at the expected location
    u, v = vorttovel(ip,KM,xn,yn,omofp)
    # Average velcities from both estimates
    u = (u+un)/2
    v = (v+vn)/2
    # Make the real move from the old position
    x, y = updatex(ip,KM,xo,yo,u,v,dt)

    if False:
        # attempt to redistribute points around boundary
        # IT DOESN'T WORK
        #x, y = redis_points(x, y)

        ll = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
        xnew = x.copy()
        ynew = y.copy()
        #print('pathlength: {:0.3f}, {:0.3f}'.format(*np.sum(ll, axis=0)))
        lx = np.cumsum(ll, axis=0)

        nx = len(x[:,0])
        xy0 = np.stack((x,y), axis=0)
        xy0 = np.concatenate((xy0[:,-10:-1,:],xy0,xy0[:,1:10,:]), axis=1)
        dxy = xy0.copy()
        dxy[:,:-1,:] = np.diff(dxy, axis=1)
        dxy[:,-1,:] = dxy[:,18,:]
        ll = np.hypot(*dxy)
        lx = np.cumsum(ll, axis=0) - ll[0]
        from scipy import interpolate
        for k in range(ll.shape[1]):
            fx = interpolate.interp1d(lx[:,k], xy0[0,:,k], kind='cubic', axis=0)
            fy = interpolate.interp1d(lx[:,k], xy0[1,:,k], kind='cubic', axis=0)
            lxnew = np.linspace(0, lx[-1,k], len(ll))
            xnew[:,k] = fx(lxnew)[9:nx+9]
            ynew[:,k] = fy(lxnew)[9:nx+9]
            #xnew[:-1,k] = np.interp(lxnew, lx[:,k], x[:-1,k])
            #ynew[:-1,k] = np.interp(lxnew, lx[:,k], y[:-1,k])
            #xnew[:,k] = interpolate.pchip_interpolate(lx[:,k], x[:,k], lxnew)
            #ynew[:,k] = interpolate.pchip_interpolate(lx[:,k], y[:,k], lxnew)
        xnew[-1,:] = xnew[0,:]
        ynew[-1,:] = ynew[0,:]

    # Save the position as the old position for the next step
    xo = x.copy()
    yo = y.copy()

    # update time
    t = t+dt

    # catch computational error and exit
    if np.any(~np.isfinite(x+y)):
        print('NON-FINITE VALUE')
        break

    if isClosed.plot_closed:
        print("Plot closed.")
        break

    # Once in a while make a nice plot
    if n % nfreq == 0:
        if verbose:
            print(f"t = {t}")

        xall[:,:,n//nfreq] = x
        yall[:,:,n//nfreq] = y
        i = n//nfreq
        ax.cla()
        ax.axis(axlim0)

        #xn, yn = redis_points(x,y)
        for k in range(0,KM):

            ax.fill(xall[:,k,i], yall[:,k,i], color=mycol[k], alpha=0.5)#, edgecolor='k', linewidth=1, linestyle='--')
            if label_0_point:
                ax.plot([xall[0,k,i]], [yall[0,k,i]], 'sk', markersize=6)
            #ax.plot(xn[:,k], yn[:,k], 'xm', markersize=2)
            if show_points:
                ax.plot(xall[:,k,i], yall[:,k,i], '.k', markersize=1)
            if boundary_quivers:
                ax.quiver(x[::5,k], y[::5,k], (x-x0)[::5,k], (y-y0)[::5,k], pivot='mid', color='g', scale=0.5)

        ax.axis('scaled')

        ax.axis(minmax(x*1.05,*axlim0[:2]) + minmax(y*1.05,*axlim0[2:]))
        #ax.axis([-1.5, 1.5, -1.5, 1.5])
        #ax.axis([-3, 3, -3, 3])

        if vector_field:
            #xgrid = np.arange(-1.4, 1.5, 0.2)
            ygrid = np.arange(-1.3, 1.4, 0.2)
            xgrid = np.linspace(*ax.get_xlim(), 16)
            ygrid = np.linspace(*ax.get_ylim(), 16)
            grid = np.round(np.meshgrid(xgrid,ygrid), 1)
            xg = np.tile(grid[...,None].copy(), (1,KM))
            xm, ym, dx, dy = xmdx_ext(x,y)

        if vector_field:
            for k in range(0,KM):
                for ig in range(grid.shape[1]):
                    for jg in range(grid.shape[2]):
                        xg[:,ig,jg,k] = xy0int(xm[:,k],ym[:,k],dx[:,k],dy[:,k],
                                                 grid[0,ig,jg], grid[1,ig,jg])
                xg[...,k] = xg[...,k]*omofp[k]
            nxg = xg/np.sum(np.hypot(*xg), axis=-1)[None,...,None]
            xg = np.log(np.abs(xg))*nxg
            ax.quiver(*grid, *np.sum(xg, axis=-1), color='k', pivot='mid', scale=None, alpha=0.5)#, scale_units='xy')

        # CALCULATE CIRCULATION
        circ = np.sum(np.diff(x,axis=0)*u+np.diff(y,axis=0)*v, 0)
        if i == 0:
            circ0 = circ.copy()

        dd = np.abs(np.sqrt((x-x0)**2+(y-y0)**2))[:-1,:]
        cn = dd - (np.sqrt(u**2+v**2)*dt)
        if np.max(cn) > 0.05:
            print(t, np.max(cn), np.min(cn))
        fdcirc = (circ-circ0)/circ
        if np.max(np.abs(fdcirc)) > 0.005:
            ixm = np.argmax(np.abs(fdcirc))
            print('Change in circulation (t={:0.4f}): {:0.5g} to {:0.5g} ({:0.2f}%)'.\
                      format(t,circ0[ixm],circ[ixm],np.max(fdcirc)*1e2))


        #axis equal
        #axis off
        #ax.set_title('time {:0.3g}'.format(t))
        htt = int(t//3600)
        for k in range(KM):
            if print_circulation:
                ctxt = ax.text(0.02, 0.95-0.05*k,
                                '$\\Gamma_{}$ = {:0.5f}'.format(k,circ[k]),
                                transform=ax.transAxes, color=mycol[k],
                                horizontalalignment='left')
            if print_circ_change:
                ctxt = ax.text(0.98, 0.95-0.05*k,
                                '$\\Delta\\Gamma_{}$ = {:0.4g}'.format(k,circ[k]-circ0[k]),
                                transform=ax.transAxes, color=mycol[k],
                                horizontalalignment='right')
            if label_patches:
                ltxt = ax.text(np.mean(x[:,k]), np.mean(y[:,k]), '{}'.format(k))

        mtt = int((t % 3600)//60)
        stt = (t % 3600 % 60)
        #txt = ax.text(0.98, 0.03, 'time: {:02d}:{:02d}:{:07.4f}'.format(htt,mtt,stt), transform=ax.transAxes, horizontalalignment='right')
        if print_time:
            if time_as_steps:
                ttxt = 'time: {}'.format(n)
            else:
                ttxt = 'time: {:02d}:{:06.3f}'.format(mtt,stt)
            txt = ax.text(0.98, 0.03, ttxt, transform=ax.transAxes, horizontalalignment='right')
        if save_to_png:
            ax.figure.savefig(f"i{itest}-step{n:06d}.png", dpi=150)
        if do_interactive:
            plt.pause(0.01)
        #input(t)
        
print('done')


