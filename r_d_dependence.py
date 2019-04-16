import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cmath
from matplotlib import cm, colors
import scipy.integrate as cp_integrate
from sympy import *
from sympy import integrate as sp_integrate
from sympy.plotting import plot_implicit
import mpmath
from mpmath import quad as sp_quad

m1 = 5.280
mlep = 0.105658
mtau = 1.776 

m2 = 1.870
q2_min = 3.67
q2_max = (m1-m2)**2

q2 = symbols('q2', real=True, positive=True)

def lam(q2):
    return m1**4 + m2**4 + q2**2 - 2*(((m1**2)*(m2**2))+((m2**2)*q2)+((m1**2)*q2))
def p2(q2):
    return sp.sqrt(lam(q2))/(2*m1)
def s(q2):
    return q2/m1**2
def fplus(q2):
    return 0.79/(1-0.75*s(q2) + 0.039*(s(q2)**2))
def fminus(q2):
    return -0.36/(1-0.77*s(q2)+0.046*(s(q2)**2))
def fs(q2):
    return 0.80/(1-0.22*s(q2) - 0.098*(s(q2)**2))
def ft(q2):
    return 0.77/(1-0.76*s(q2) + 0.043*(s(q2)**2))
#def fplus_w(q2):
#    return (1/(2*sp.sqrt(m1*m2)))*((m1+m2)*hplus(q2)-(m1-m2)*hminus(q2))
#def f0_w(q2):
#    return ((((m1+m2)**2-q2)*(hplus(q2)))/(m1+m2)-(((m1-m2)**2-q2)*(hminus(q2)))/(m1-m2))/(2*sp.sqrt(m1*m2))
#def w(q2):
#    return (m1**2 + m2**2 - q2)/(2*m1*m2)
#r = m2/m1
#rho2 = 1.186
#def z(q2):
#    return (sp.sqrt(w(q2)+1)-np.sqrt(2))/(sp.sqrt(w(q2)+1)+np.sqrt(2))
#v_1 = 1.074
#def v1(q2):
#    return v_1*(1-8*rho2*z(q2)+(51*rho2-10)*(z(q2)**2)-(252*rho2-84)*(z(q2)**3))
#def s1(q2):
#    return v1(q2)*(1-0.019+0.041*(w(q2)-1)-0.015*(w(q2)-1)**2)
#def hplus(q2):
#    return (1/(2*(1+r**2 - 2*r*w(q2))))*(((1-r)**2)*(w(q2)+1)*s1(q2)-((1+r)**2)*(w(q2)-1)*v1(q2))
#def hminus(q2):
#    return (((1-r**2)*(w(q2)+1))/(2*(1+r**2 - 2*r*w(q2))))*(s1(q2)-v1(q2))
#def ht(q2):
#    return ((m1**2 - m2**2)*(f0_w(q2)))/sp.sqrt(q2)
#def h0(q2):
#    return (2*m1*p2(q2)*fplus_w(q2))/(sp.sqrt(q2))
def delta_tau(q2):
    return mtau**2/(2*q2)
def delta_lep(q2):
    return mlep**2/(2*q2)
def ht(q2):
    return (1.0/sp.sqrt(q2))*((m1**2 - m2**2)*fplus(q2)+q2*fminus(q2))
def h0(q2):
    return (2*m1*p2(q2)*fplus(q2))/(sp.sqrt(q2))
def lep(q2):
    return p2(q2)*q2*((1.0-((mlep**2)/q2))**2)*(((1.0+delta_lep(q2))*(Abs(h0(q2))**2))+(3.0*delta_lep(q2)*(Abs(ht(q2))**2))) 
lep_int = cp_integrate.quad(lep, mlep**2, q2_max)[0]

#def tau_sm(q2):
#    return p2(q2)*q2*((1.0-((mtau**2)/q2))**2)*(((1.0+delta_tau(q2))*(Abs(h0(q2))**2))+(3.0*delta_tau(q2)*(Abs(ht(q2))**2)))
#tau_int_sm = cp_integrate.quad(tau_sm, mtau**2, q2_max)[0]
#r_d_sm = tau_int_sm/lep_int
#print r_d_sm

def hsp(q2):
    return (m1+m2)*fs(q2)
def h_t(q2):
    return (2*m1*p2(q2)*ft(q2))/(m1+m2)

#v_l,v_r,s_l,s_r,t_l = symbols('v_l v_r s_l s_r t_l')

def tau_np(q2,v_l,v_r,s_l,s_r,t_l):
    return Abs(p2(q2))*q2*((1-((mtau**2)/q2))**2)*((Abs(1+v_l+v_r)**2)*(((1+delta_tau(q2))*(Abs(h0(q2))**2))+(3*delta_tau(q2)*(Abs(ht(q2))**2)))+((3/2)*(Abs(s_l+s_r)**2)*(Abs(hsp(q2))**2))+(8*Abs(t_l)**2*(1+4*delta_tau(q2)*Abs(h_t(q2))**2)+(3*sp.sqrt(2*delta_tau(q2))*re(s_l+s_r)*hsp(q2)*ht(q2))+(12*sp.sqrt(2*delta_tau(q2))*re(t_l)*h0(q2)*h_t(q2))))
def tau_int_np(v_l,v_r,s_l,s_r,t_l):
    return sp_quad(lambda q2: tau_np(q2,v_l,v_r,s_l,s_r,t_l), [mtau**2, q2_max])
def r_d_np(v_l,v_r,s_l,s_r,t_l):
    return tau_int_np(v_l,v_r,s_l,s_r,t_l)/lep_int
print r_d_np(0.0,0.0,0.0,0.0,0.0)

m2_star = 2.01
q2_max_star = (m1 - m2_star)**2

def lamstar(q2):
    return m1**4 + m2_star**4 + q2**2 - 2*(((m1**2)*(m2_star**2))+((m2_star**2)*q2)+((m1**2)*q2))
def p2star(q2):
    return sp.sqrt(lamstar(q2))/(2*m1)
def A0(q2):
    return 1.62/(1-0.34*s(q2) -0.16*(s(q2)**2))
def Aplus(q2):
    return 0.67/(1-0.87*s(q2) + 0.057*(s(q2)**2))
def Aminus(q2):
    return -0.77/(1-0.89*s(q2) + 0.070*(s(q2)**2))
def V(q2):
    return 0.77/(1-0.90*s(q2) + 0.075*(s(q2)**2))
def gs(q2):
    return -0.5/(1-0.87*s(q2) + 0.060*(s(q2)**2))
def gt1(q2):
    return -0.073/(1-1.23*s(q2) + 0.33*(s(q2)**2))
def gt2(q2):
    return 0.73/(1-0.90*s(q2) + 0.074*(s(q2)**2))
def gt0(q2):
    return -0.37/(1-0.88*s(q2) + 0.064*(s(q2)**2))
def h00(q2):
    return (-(m1**2 - m2_star**2)*(m1**2-m2_star**2-q2)*A0(q2)+4*(m1**2)*(p2(q2)**2)*Aplus(q2))/(2*m2_star*sp.sqrt(q2)*(m1+m2_star))
def hplusplus(q2):
    return (-(m1**2-m2_star**2)*A0(q2)+2*m1*p2(q2)*V(q2))/(m1+m2_star)
def hminmin(q2):
    return (-(m1**2-m2_star**2)*A0(q2)-2*m1*p2(q2)*V(q2))/(m1+m2_star)
def ht0(q2):
    return (m1*p2star(q2)*((m1**2 - m2_star**2)*(-A0(q2)+Aplus(q2))+q2*Aminus(q2)))/(m2_star*sp.sqrt(q2)*(m1+m2_star))
def lepstar(q2):
    return p2star(q2)*q2*((1-((mlep**2)/q2))**2)*((1+(mlep**2)/(2*q2))*(Abs(h00(q2))**2+Abs(hplusplus(q2))**2+Abs(hminmin(q2))**2)+3*(mlep**2/(2*q2))*Abs(ht0(q2))**2)
lepstar_int = cp_integrate.quad(lepstar, mlep**2, q2_max_star)[0]
#def taustar_sm(q2):
#    return p2star(q2)*q2*((1-((mtau**2)/q2))**2)*((1+delta_tau(q2))*(Abs(h00(q2))**2+Abs(hplusplus(q2))**2+Abs(hminmin(q2))**2)+3*(delta_tau(q2))*Abs(ht0(q2))**2)
#taustar_int_sm = cp_integrate.quad(taustar_sm, mtau**2, q2_max_star)[0]
#r_d_star_sm = taustar_int_sm/lepstar_int
#print r_d_star_sm
def hsv(q2):
    return m1*p2star(q2)*gs(q2)/m2_star 
def h0_t(q2):
    return -((m1**2 + 3*m2_star**2 - q2)*gt1(q2)+(m1**2 - m2_star**2 -q2)*gt2(q2)-(lamstar(q2)/((m1+m2_star)**2))*gt0(q2))/(2*m2_star)
def hplus_t(q2):
    return -((m1**2-m2_star**2+sp.sqrt(lamstar(q2)))*gt1(q2)+q2*gt2(q2))/sp.sqrt(q2)
def hminus_t(q2):
    return -((m1**2-m2_star**2-sp.sqrt(lamstar(q2)))*gt1(q2)+q2*gt2(q2))/sp.sqrt(q2)
def taustar_np(q2,v_l,v_r,s_l,s_r,t_l):
    return p2star(q2)*q2*((1-((mtau**2)/q2))**2)*((Abs(1+v_l)**2+Abs(v_r)**2)*((1+delta_tau(q2))*(Abs(h00(q2))**2+Abs(hplusplus(q2))**2+Abs(hminmin(q2))**2)+3*(delta_tau(q2))*Abs(ht0(q2))**2)-2*re(v_r)*((1+delta_tau(q2))*(Abs(h00(q2))**2+2*hplusplus(q2)*hminmin(q2))+3*delta_tau(q2)*Abs(ht0(q2))**2)+(3/2)*(Abs(s_r-s_l)**2)*(Abs(hsv(q2))**2)+3*sp.sqrt(2*delta_tau(q2))*re(s_r-s_l)*hsv(q2)*ht0(q2)+8*(Abs(t_l)**2)*(1+4*delta_tau(q2))*((Abs(h0_t(q2))**2)+(Abs(hplus_t(q2))**2)+(Abs(hminus_t(q2))**2))-12*sp.sqrt(2*delta_tau(q2))*re(t_l)*(hplusplus(q2)*hplus_t(q2)+hminmin(q2)*hminus_t(q2)+h00(q2)*h0_t(q2)))
def taustar_int_np(v_l,v_r,s_l,s_r,t_l):
    return cp_integrate.quad(taustar_np, mtau**2, q2_max_star, args=(v_l,v_r,s_l,s_r,t_l))[0]
def r_d_star_np(v_l,v_r,s_l,s_r,t_l):
    return taustar_int_np(v_l,v_r,s_l,s_r,t_l)/lepstar_int
print r_d_star_np(0.0,0.0,0.0,0.0,0.0)

#constraints from bc lifetime

lifetime_bc = (0.507e-12)/(6.582e-25)
m_bc = 6.275
fbc = 0.434
g_f = 1.1663787e-5
V_cb = 0.0422
mb = 4.18
mc = 1.28

def bc_decay(v_l, s_r):
    return (lifetime_bc*m_bc*(mtau**2)*(fbc**2)*(g_f**2)*(V_cb**2)*((1-(mtau**2/m_bc**2))**2))*(Abs(1+v_l+(m_bc**2/(mtau*(mb+mc)))*s_r)**2)/(8*np.pi)

print bc_decay(0.0,0.63)

#jpsi anomaly

m1j = 6.2751
m2j = 3.0969
q2_jpsi_max = (m1j-m2j)**2
def lamj(q2):
    return m1j**4 + m2j**4 + q2**2 - 2*(((m1j**2)*(m2j**2))+((m2j**2)*q2)+((m1j**2)*q2))
def p2j(q2):
    return sp.sqrt(lamj(q2))/(2*m1j)
def sj(q2):
    return q2/m1j**2
def A0j(q2):
    return 1.65/(1-1.19*sj(q2) +0.17*(sj(q2)**2))
def Aplusj(q2):
    return 0.55/(1-1.68*sj(q2) + 0.70*(sj(q2)**2))
def Aminusj(q2):
    return -0.87/(1-1.85*sj(q2) + 0.91*(sj(q2)**2))
def Vj(q2):
    return 0.78/(1-1.82*sj(q2) + 0.86*(sj(q2)**2))
def gsj(q2):
    return -0.61/(1-1.84*s(q2) + 0.91*(s(q2)**2))
def gt1j(q2):
    return 0.56/(1-1.86*s(q2) + 0.93*(s(q2)**2))
def gt2j(q2):
    return -0.27/(1-1.91*s(q2) + 1.00*(s(q2)**2))
def gt0j(q2):
    return -0.21/(1-2.16*s(q2) + 1.33*(s(q2)**2))
def h0j(q2):
    return (-(m1j**2 - m2j**2)*(m1j**2-m2j**2-q2)*A0j(q2)+4*(m1j**2)*(p2j(q2)**2)*Aplusj(q2))/(2*m2j*sp.sqrt(q2)*(m1j+m2j))
def hplusplusj(q2):
    return (-(m1j**2-m2j**2)*A0j(q2)+2*m1j*p2j(q2)*Vj(q2))/(m1j+m2j)
def hminminj(q2):
    return (-(m1j**2-m2j**2)*A0j(q2)-2*m1j*p2j(q2)*Vj(q2))/(m1j+m2j)
def htj(q2):
    return (m1j*p2j(q2)*((m1j**2 - m2j**2)*(-A0j(q2)+Aplusj(q2))+q2*Aminusj(q2)))/(m2j*sp.sqrt(q2)*(m1j+m2j))
def hsvj(q2):
    return m1j*p2j(q2)*gsj(q2)/m2j 
def h0_tj(q2):
    return -((m1j**2 + 3*m2j**2 - q2)*gt1j(q2)+(m1j**2 - m2j**2 -q2)*gt2j(q2)-(lamj(q2)/((m1j+m2j)**2))*gt0j(q2))/(2*m2j)
def hplus_tj(q2):
    return -((m1j**2-m2j**2+sp.sqrt(lamj(q2)))*gt1j(q2)+q2*gt2j(q2))/sp.sqrt(q2)
def hminus_tj(q2):
    return -((m1j**2-m2j**2-sp.sqrt(lamj(q2)))*gt1j(q2)+q2*gt2j(q2))/sp.sqrt(q2)
def jpsi_lep(q2):
    return p2j(q2)*q2*((1-((mlep**2)/(q2)))**2)*(Abs(h0j(q2))**2 + Abs(hplusplusj(q2))**2 + Abs(hminminj(q2))**2 + delta_lep(q2)*(Abs(h0j(q2))**2 + Abs(hminminj(q2))**2 + Abs(hplusplusj(q2))**2 + 3*(Abs(htj(q2))**2)))
jpsi_lep_int = cp_integrate.quad(jpsi_lep, mlep**2, q2_jpsi_max)[0]
def jpsi_tau(q2, v_l, v_r, s_l, s_r, t_l):
    return p2j(q2)*q2*((1-((mtau**2)/(q2)))**2)*((Abs(1 + v_l)**2 + Abs(v_r)**2)*(Abs(h0j(q2))**2 + Abs(hplusplusj(q2))**2 + Abs(hminminj(q2))**2 + delta_tau(q2)*((Abs(h0j(q2))**2 + Abs(hplusplusj(q2))**2 + Abs(hminminj(q2))**2) + 3*(Abs(htj(q2))**2)))+(3/2)*(Abs(s_l - s_r)**2)*(Abs(hsvj(q2))**2)-2*re(v_r)*((1.0 + delta_tau(q2))*(Abs(h0j(q2))**2 + 2*hplus_tj(q2)*hminus_tj(q2))+3*delta_tau(q2)*Abs(htj(q2))**2) - 3.0*sp.sqrt(2*delta_tau(q2))*re(s_l - s_r)*hsvj(q2)*htj(q2) + 8*(Abs(t_l)**2)*(1+4*delta_tau(q2))*(Abs(h0_tj(q2))**2+Abs(hplus_tj(q2))**2+Abs(hminus_tj(q2))**2) - 12.0*sp.sqrt(2*delta_tau(q2))*re(t_l)*(h0j(q2)*h0_tj(q2) + hplusplusj(q2)*hplus_tj(q2) + hminminj(q2)*hminus_tj(q2)))
def jpsi_tau_int(v_l,v_r,s_l,s_r,t_l):
    return cp_integrate.quad(jpsi_tau, mtau**2, q2_jpsi_max, args=(v_l,v_r,s_l,s_r,t_l))[0]
def rjpsi(v_l, v_r, s_l, s_r, t_l):
    return jpsi_tau_int(v_l,v_r,s_l,s_r,t_l)/jpsi_lep_int
print rjpsi(0.0,0.0,0.0,0.0,0.0)

sampleno=150

mv_l=sp.zeros(sampleno, dtype=complex)

x=np.linspace(-2.0,2.0,sampleno, endpoint=True)
y=np.linspace(-2.0,2.0,sampleno, endpoint=True)
for i in range(sampleno):
    for j in range(sampleno):
        mv_l[i,j] = x[i]+I*y[j]

soln1=[]
for i in range(sampleno):
    for j in range(sampleno):
        nv_l = mv_l[i,j]
        if 0.46<=rjpsi(nv_l,0.0,0.0,0.0,0.0)<=0.96:
            soln1.append(nv_l)

#soln2=[]
#for i in range(sampleno):
#    for j in range(sampleno):
#        nv_l = mv_l[i,j]
#        if 0.254<=r_d_np(0.0,0.0,0.0,0.0,nv_l)<=0.36:
#            soln2.append(nv_l)
#npsoln1=[]
#for a in range(len(soln1)):
#    if 0.286<=r_d_star_np(0.0,0.0,0.0,0.0,soln1[a])<=0.326:
#        npsoln1.append(soln1[a])
#
#npsoln2=[]
#for d in range(len(soln2)):
#    if 0.251<=r_d_star_np(0.0,0.0,0.0,0.0,soln2[d])<=0.315:
#        npsoln2.append(soln2[d])
#
#x1=[]
#y1=[]
#for b in range(len(npsoln1)):
#    x1.append(re(npsoln1[b]))
#    y1.append(im(npsoln1[b]))
#
#x2=[]
#y2=[]
#for c in range(len(npsoln2)):
#    x2.append(re(npsoln2[c]))
#    y2.append(im(npsoln2[c]))

x=[]
y=[]
for c in range(len(soln1)):
    x.append(re(soln1[c]))
    y.append(im(soln1[c]))
fig, ax = plt.subplots()
plt.scatter(x, y, edgecolors='none')
#plt.scatter(x2, y2, color='r', edgecolors='none')
plt.xlabel('$Re(V_L)$')
plt.ylabel('$Im(V_L)$')
plt.axhline(color='k', linewidth=0.5)
plt.axvline(color='k', linewidth=0.5)
plt.tick_params(direction='in')
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

plt.show()

#sampleno=150
#
#mv_l=sp.zeros(sampleno, dtype=complex)
#
#x=np.linspace(0.0,1.0,sampleno, endpoint=True)
#y=np.linspace(-1.0,1.0,sampleno, endpoint=True)
#for i in range(sampleno):
#    for j in range(sampleno):
#        mv_l[i,j] = x[i]+I*y[j]
#
#soln=[]
#for i in range(sampleno):
#    for j in range(sampleno):
#        nv_l = mv_l[i,j]
#        if bc_decay(0.0,nv_l)<=0.3:
#            soln.append(nv_l)
#
#x=[]
#y=[]
#for b in range(len(soln)):
#    x.append(re(soln[b]))
#    y.append(im(soln[b]))
#
#plt.scatter(x, y)
#plt.xlabel('$Re(S_R)$')
#plt.ylabel('$Im(S_R)$')
#plt.show()

#sampleno=150
#
#mv_l=sp.zeros(sampleno, dtype=complex)
#
#x=np.linspace(-3.0,1.0,sampleno, endpoint=True)
#y=np.linspace(-1.5,1.5,sampleno, endpoint=True)
#for i in range(sampleno):
#    for j in range(sampleno):
#        mv_l[i,j] = x[i]+I*y[j]
#
#soln=[]
#for i in range(sampleno):
#    for j in range(sampleno):
#        nv_l = mv_l[i,j]
#        print i, j
#        if 0.326<=r_d_np(nv_l,0.0,0.0,0.0,0.0)<=0.426:
#            soln.append(nv_l)
#npsoln=[]
#for a in range(len(soln)):
#    if 0.29<=r_d_star_np(soln[a],0.0,0.0,0.0,0.0)<=0.33:
#        npsoln.append(soln[a])
#
#x=[]
#y=[]
#for b in range(len(npsoln)):
#    x.append(re(npsoln[b]))
#    y.append(im(npsoln[b]))
#
#plt.scatter(x, y)
#plt.xlabel('$Re(V_L)$')
#plt.ylabel('$Im(V_L)$')
#plt.show()

