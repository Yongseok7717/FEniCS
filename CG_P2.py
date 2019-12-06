#!/usr/bin/python
from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys

# SS added
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
parameters["ghost_mode"] = "shared_facet"
set_log_active(False)


def usage():
  print("-h   or --help")  
  print("-k   ")
  print("-i i or --iMin  i       to specify iMin")
  print("-j j or --jMin  j       to specify jMin")
  print("-I i or --iMax  i       to specify iMax")
  print("-J j or --jMax  j       to specify jMax")
  print(" ")
  os.system('date +%Y_%m_%d_%H-%M-%S')
  print (time.strftime("%d/%m/%Y at %H:%M:%S"))

# parse the command line
try:
  opts, args = getopt.getopt(sys.argv[1:], "hk:i:I:j:J:",
                   [
                    "help",           # obvious
                    "degree of polynomial="
                    "iMin=",          # iMin
                    "iMax=",          # iMax
                    "jMin=",          # jMin
                    "jMax=",          # jMax
                    ])

except getopt.GetoptError as err:
  # print help information and exit:
  print(err) # will print something like "option -a not recognized"
  usage()
  sys.exit(2)

for o, a in opts:
  if o in ("-h", "--help"):
    usage()
    sys.exit()
  elif o in ("-k"):
    k = int(a)
    print('setting:  k = %d;' % k),
  elif o in ("-i", "--iMin"):
    iMin = int(a)
    print('setting:  iMin = %f;' % iMin),
  elif o in ("-I", "--iMax"):
    iMax = int(a)
    print('setting:  iMax = %f;' % iMax),
  elif o in ("-j", "--jMin"):
    jMin = int(a)
    print('setting:  jMin = %f;' % jMin),
  elif o in ("-J", "--jMax"):
    jMax = int(a)
    print('setting:  jMax = %f;' % jMax),
  else:
    assert False, "unhandled option"


#Save data for error
V_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
L2_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)
Lu_error=np.zeros((iMax-iMin+1,jMax-jMin+1), dtype=np.float64)

# problem data
T = 1.0     # total simulation time
varphi0=0.5
varphi1=0.1
varphi2=0.4

tau1=0.5
tau2=1.5

# Define solution for error and BC's
ux = Expression('exp(-tn)*sin(x[0]*x[1])', tn=0, degree=5)
wx = Expression('-exp(-tn)*sin(x[0]*x[1])', tn=0, degree=5)

zetax1 = Expression('-(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)*tau1*sin(x[0]*x[1])',varphi1=0.1,tau1=0.5,  tn=0, degree=5)
zetax2 = Expression('-(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2)*tau2*sin(x[0]*x[1])',varphi2=0.4,tau2=1.5,  tn=0, degree=5)



tol=1E-14


for i in range(iMin,iMax):
    for j in range(jMin,jMax):
        Nxy=pow(2,i)
        mesh = UnitSquareMesh(Nxy, Nxy)
        V = FunctionSpace(mesh, 'Lagrange', k)
    # Define boundary conditions

    #bottom edge
        def bottom(x, on_boundary):
            return near(x[1],0.0,tol) and on_boundary
    # left edge
        def left(x, on_boundary):
            return near(x[0],0.0,tol) and on_boundary

        bc_B = DirichletBC(V, ux, bottom)
        bc_L = DirichletBC(V,ux,left)
        bcs = [bc_B, bc_L]

    # define the boundary partition
        boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

    # Mark subdomain 0 for \Gamma_0 etc
    #Gamma0 has Neumann BC(top)
        class Gamma0(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1],1.0,tol)
        Gamma_0 = Gamma0()
        Gamma_0.mark(boundary_parts, 0)

    #Gamma1 has Neumann BC(right)
        class Gamma1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0],1.0,tol)
        Gamma_1 = Gamma1()
        Gamma_1.mark(boundary_parts, 1)


        ds=ds(subdomain_data = boundary_parts)
        dx=Measure('dx')
    
    # Initial condition
        ux.tn=0.0; wx.tn=0.0;zetax1.tn=0.0; zetax2.tn=0.0;

        U = FunctionSpace(mesh, 'Lagrange', 5)
        u=TrialFunction(V)
        v=TestFunction(V)
        Ux=interpolate(ux,U)
        A0=inner(grad(u),grad(v))*dx
        L0=inner(grad(Ux),grad(v))*dx
        U0=Function(V);u0=Function(V)
        solve(A0==L0,U0,bcs)
        u0.assign(U0)

        w0 = project(wx, V)

    # since our initial internal variables are zero
        zeta01 = project(zetax1, V)
        zeta02 = project(zetax2, V)

        Nt=2**j   # mesh density and number of time steps
        dt = T/Nt      # time step

        # Define average value functions for f and Neumann boundary conditions
        fav = Expression('0.5*(exp(-tn)*sin(x[0]*x[1])+(x[0]*x[0]+x[1]*x[1])*sin(x[0]*x[1])\
            *(exp(-tn)-(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)-(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2)))\
            +0.5*(exp(-tn+dt)*sin(x[0]*x[1])+(x[0]*x[0]+x[1]*x[1])*sin(x[0]*x[1])\
            *(exp(-tn+dt)-(exp(-tn+dt)-exp(-(tn-dt)/tau1))*varphi1/(1.0-tau1)-(exp(-tn+dt)-exp(-(tn-dt)/tau2))*varphi2/(1.0-tau2)))',
                   tau1=0.5,tau2= 1.5, varphi1=0.1,varphi2=0.4,dt=dt, tn=0, degree=5)
        g0av =Expression('0.5*(exp(-tn)-(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)-(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2))\
            *x[0]*cos(x[0]*x[1])\
            +0.5*(exp(-tn+dt)-(exp(-tn+dt)-exp(-(tn-dt)/tau1))*varphi1/(1.0-tau1)-(exp(-tn+dt)-exp(-(tn-dt)/tau2))*varphi2/(1.0-tau2))\
            *x[0]*cos(x[0]*x[1])', tau1=0.5,tau2= 1.5, varphi1=0.1,varphi2=0.4,dt=dt, tn=0, degree=5)
        g1av =Expression('0.5*(exp(-tn)-(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)-(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2))\
            *x[1]*cos(x[0]*x[1])\
            +0.5*(exp(-tn+dt)-(exp(-tn+dt)-exp(-(tn-dt)/tau1))*varphi1/(1.0-tau1)-(exp(-tn+dt)-exp(-(tn-dt)/tau2))*varphi2/(1.0-tau2))\
            *x[1]*cos(x[0]*x[1])', tau1=0.5,tau2= 1.5, varphi1=0.1,varphi2=0.4,dt=dt, tn=0, degree=5)
        
        Rav = Expression('(varphi1*(exp(-tn/tau1)+exp(-(tn-dt)/tau1))+varphi2*(exp(-tn/tau2)+exp(-(tn-dt)/tau2)))*0.5',tau1=0.5,tau2= 1.5, varphi1=0.1,varphi2=0.4,dt=dt, tn=0, degree=5)
        
        # Define variational problem
        u, v = TrialFunction(V), TestFunction(V)

        # Compute solution
        uh = Function(V)   # the unknown at a new time level
        wh = Function(V)
        zetah1 = Function(V)
        zetah2 = Function(V)

        # bilinear form for the solver
        mass = u*v*dx
        stiffness = inner(nabla_grad(u), nabla_grad(v))*dx

        # linear form for the right hand side
        L=2.0/dt*w0*v*dx+2.0/dt/dt*u0*v*dx\
            +(tau1*varphi1/(2*tau1+dt)+tau2*varphi2/(2*tau2+dt)-0.5*varphi0)*inner(nabla_grad(u0), nabla_grad(v))*dx\
            -((2*tau1)/(2*tau1+dt))*inner(nabla_grad(zeta01), nabla_grad(v))*dx\
            -((2*tau2)/(2*tau2+dt))*inner(nabla_grad(zeta02), nabla_grad(v))*dx\
                +fav*v*dx+g0av*v*ds(0)+g1av*v*ds(1)-Rav*inner(nabla_grad(U0), nabla_grad(v))*dx  
   
        # assemble the system matrix once and for all
        M = assemble(mass)
        A = assemble(stiffness)
        CurlB=(tau1*varphi1/(2*tau1+dt)+tau2*varphi2/(2*tau2+dt))*A
        B=(2/dt/dt*M)+CurlB+(varphi0/2*A)

   
        for n in range(1,Nt+1):
         #   progbar += 1;
            b = None
            # update data and solve for tn+k
            tn = n*dt; ux.tn = tn; wx.tn = tn; fav.tn = tn; g0av.tn = tn; g1av.tn = tn;Rav.tn=tn;
            b = assemble(L, tensor=b)
            for bc in bcs:
                bc.apply(B,b)      
            #print(n)   
            solver = KrylovSolver('cg','sor')
            prm = solver.parameters
            prm.absolute_tolerance = 1E-16  # from 10
            prm.relative_tolerance = 1E-16
            prm.maximum_iterations = 10000000000 
            solve(B, uh.vector(), b)   
            wh.assign(2/dt*(uh-u0)-w0)
            zetah1.assign(2*tau1*varphi1/(2*tau1+dt)*(uh-u0)+(2*tau1-dt)/(2*tau1+dt)*zeta01)
            zetah2.assign(2*tau2*varphi2/(2*tau2+dt)*(uh-u0)+(2*tau2-dt)/(2*tau2+dt)*zeta02)
    
            # update old terms
            w0.assign(wh);u0.assign(uh);zeta01.assign(zetah1);zeta02.assign(zetah2)
            
        # compute error at last time step
        ux.tn = T; wx.tn = T; 
        err0 = errornorm(ux,u0,'H10')
        err1 = errornorm(wx,w0,'L2')
        err2 = errornorm(ux,u0,'L2')
        
        V_error[0,j-jMin+1]=Nt; V_error[i-iMin+1,0]=Nxy; V_error[i-iMin+1,j-jMin+1]=err0;
        L2_error[0,j-jMin+1]=Nt; L2_error[i-iMin+1,0]=Nxy; L2_error[i-iMin+1,j-jMin+1]=err1;
        Lu_error[0,j-jMin+1]=Nt; Lu_error[i-iMin+1,0]=Nxy; Lu_error[i-iMin+1,j-jMin+1]=err2;
       
# Print big tables
print('Tables of error for P2 where k=%d' %k)
print ('V_error for u')
print('\\begin{tabular}{|l|',end="")
for j in range(jMin,jMax): print('l',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % V_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % V_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % V_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('L2_error for w')
print('\\begin{tabular}{|l|',end="")
for j in range(jMin,jMax): print('l',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % L2_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % L2_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % L2_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

print ('L2_error for u')
print('\\begin{tabular}{|l|',end="")
for j in range(jMin,jMax): print('l',end="")
print('|}\\hline')
for j in range(jMin,jMax):
  if j==jMin: print('    ', end="")
  print(' & %10d ' % Lu_error[0,j-jMin+1], end="")
print('  \\\\\\hline')
for i in range(iMin,iMax):
  print('%4d ' % Lu_error[i-iMin+1,0], end="")
  for j in range(jMin,jMax):
      print('& %11.4le ' % Lu_error[i-iMin+1,j-jMin+1], end="")
  print(' \\\\')
print('\\hline\\end{tabular}')

# Save results of errors in text files
if k==1:
    np.savetxt("energy_linear_p2.txt",V_error,fmt="%2.3e")
    np.savetxt("L2w_linear_p2.txt",L2_error,fmt="%2.3e")
    np.savetxt("L2u_linear_p2.txt",Lu_error,fmt="%2.3e")
elif k==2:    
    np.savetxt("energy_quad_p2.txt",V_error,fmt="%2.3e")
    np.savetxt("L2w_quad_p2.txt",L2_error,fmt="%2.3e")
    np.savetxt("L2u_quad_p2.txt",Lu_error,fmt="%2.3e")

'''
# Compute error orders
l2Diag=[]
h1Diag=[]
l2wDiag=[]
m= min(iMax-iMin,jMax-jMin)+1
for i0 in range(1,m):
    l2Diag.append(Lu_error[i0,i0])
    h1Diag.append(V_error[i0,i0])
    l2wDiag.append(Lu_error[i0,i0])
        
v1=np.array(l2Diag)
t1=np.log(v1[0:m-2]/v1[1:m-1])
d1=np.mean(t1/np.log(2))

v2=np.array(h1Diag)
t2=np.log(v2[0:m-2]/v2[1:m-1])
d2=np.mean(t2/np.log(2))

v3=np.array(l2wDiag)
t3=np.log(v3[0:m-2]/v3[1:m-1])
d3=np.mean(t3/np.log(2))
print('Numeical convergent order of P2 when h=dt for k= %d ' %k)
print('L2 error of u = %5.4f,  H1 error of u= %5.4f,  L2 error of w= %5.4f' %(d1,d2,d3))
'''