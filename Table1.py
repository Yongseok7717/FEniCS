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
psix1 = Expression('(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)*sin(x[0]*x[1])',tau1=0.5, varphi1=0.1,  tn=0, degree=5)
psix2 = Expression('(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2)*sin(x[0]*x[1])',tau2=1.5, varphi2=0.4,  tn=0, degree=5)
zetax1 = Expression('-(exp(-tn)-exp(-tn/tau1))*varphi1/(1.0-tau1)*tau1*sin(x[0]*x[1])',varphi1=0.1,tau1=0.5,  tn=0, degree=5)
zetax2 = Expression('-(exp(-tn)-exp(-tn/tau2))*varphi2/(1.0-tau2)*tau2*sin(x[0]*x[1])',varphi2=0.4,tau2=1.5,  tn=0, degree=5)
# Fixed timestep size
iMin=2
iMax=6
jMin=0
jMax=1

error_DisplacementForm=np.zeros((4,4), dtype=np.float64)
error_VelocityForm=np.zeros((4,4), dtype=np.float64)
tol=1E-14
#---------------------------------------------------------------------------------------------------------#
# Solve P1
for i in range(iMin,iMax):
    for j in range(jMin,jMax):
        Nxy=pow(2,i)
        mesh = UnitSquareMesh(Nxy, Nxy)
        V = FunctionSpace(mesh, 'Lagrange', 2)
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
        ux.tn=0.0; wx.tn=0.0;psix1.tn=0.0; psix2.tn=0.0;

        U = FunctionSpace(mesh, 'Lagrange', 5)
        u=TrialFunction(V)
        v=TestFunction(V)
        Ux=interpolate(ux,U)
        A0=inner(grad(u),grad(v))*dx
        L0=inner(grad(Ux),grad(v))*dx

        u0=Function(V)
        solve(A0==L0,u0,bcs)

        w0 = project(wx, V)

    # since our initial internal variables are zero
        psi01 = interpolate(psix1, V)
        psi02 = interpolate(psix2, V)

        Nt=1200   # mesh density and number of time steps
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

        # Define variational problem
        u, v = TrialFunction(V), TestFunction(V)

        # Compute solution
        uh = Function(V)   # the unknown at a new time level
        wh = Function(V)
        psih1 = Function(V)
        psih2 = Function(V)

        # bilinear form for the solver
        mass = u*v*dx
        stiffness = inner(nabla_grad(u), nabla_grad(v))*dx

        # linear form for the right hand side and internal variables
        L=2.0/dt*w0*v*dx+2/dt/dt*u0*v*dx-0.5*(1-varphi1/(tau1/dt+0.5)/2-varphi2/(tau2/dt+0.5)/2)*inner(nabla_grad(u0), nabla_grad(v))*dx\
            +2*tau1/(2*tau1+dt)*inner(nabla_grad(psi01), nabla_grad(v))*dx\
            +2*tau2/(2*tau2+dt)*inner(nabla_grad(psi02), nabla_grad(v))*dx\
            +fav*v*dx+g0av*v*ds(0)+g1av*v*ds(1)
        
        # assemble the system matrix once and for all
        M = assemble(mass)
        A = assemble(stiffness)
        CurlA=0.5*(1-varphi1/(tau1/dt+0.5)/2-varphi2/(tau2/dt+0.5)/2)*A
        B=2/dt/dt*M+CurlA


        for n in range(1,Nt+1):
         #   progbar += 1;
            b = None
            # update data and solve for tn+k
            tn = n*dt; ux.tn = tn; wx.tn = tn; fav.tn = tn; g0av.tn = tn; g1av.tn = tn;
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
            psih1.assign(2*dt/(2*tau1+dt)*((2*tau1-dt)/2/dt*psi01+varphi1/2.0*(uh+u0)))
            psih2.assign(2*dt/(2*tau2+dt)*((2*tau2-dt)/2/dt*psi02+varphi2/2.0*(uh+u0)))
            #update old terms
            w0.assign(wh);u0.assign(uh);psi01.assign(psih1);psi02.assign(psih2)
        
        # compute error at last time step
        ux.tn = T; wx.tn = T; 
        err0 = errornorm(ux,u0,'H10')
        err1 = errornorm(wx,w0,'L2')        
        err2 = errornorm(ux,u0,'L2')
        error_DisplacementForm[i-iMin,0]=Nxy; error_DisplacementForm[i-iMin,1]=err0; error_DisplacementForm[i-iMin,2]=err1; error_DisplacementForm[i-iMin,3]=err2;
        

# Compute error orders
        
t1=np.log(error_DisplacementForm[0:-2,1]/error_DisplacementForm[1:-1,1])
d1=np.mean(t1/np.log(2))

t2=np.log(error_DisplacementForm[0:-2,2]/error_DisplacementForm[1:-1,2])
d2=np.mean(t2/np.log(2))

t3=np.log(error_DisplacementForm[0:-2,3]/error_DisplacementForm[1:-1,3])
d3=np.mean(t3/np.log(2))


# Display table of Displacement form
print('')
print('Tables for P1 with fixed timestep size ')

'''
Let us define Vu = energy norm error of u, L2w = L2norm error of w, L2u = L2 norm error of u,
where w is a time derivative of u
'''
print(' %4s  %10s  %10s  %10s ' % ('h','Vu','L2w','L2u'))
for i in range(0,4):
    print(' 1/%2d ' % error_DisplacementForm[i,0], end="")
    for j in range (1,4):
        print(' %10.4e ' % error_DisplacementForm[i,j], end="")
    print('')
print(' %4s  %10.2f  %10.2f  %10.2f ' % ('rate',d1,d2,d3))
print('')

#---------------------------------------------------------------------------------------------------------#
# Solve P2
for i in range(iMin,iMax):
    for j in range(jMin,jMax):
        Nxy=pow(2,i)
        mesh = UnitSquareMesh(Nxy, Nxy)
        V = FunctionSpace(mesh, 'Lagrange', 2)
    # Define boundary conditions

    #bottom edge
        def bottom(x, on_boundary):
            return near(x[1],0.0,tol) and on_boundary
    # left edge
        def left(x, on_boundary):
            return near(x[0],0.0,tol) and on_boundary
        ux.tn=0.0
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

        Nt=1200 # the number of timesteps
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
        
        error_VelocityForm[i-iMin,0]=Nxy; error_VelocityForm[i-iMin,1]=err0; error_VelocityForm[i-iMin,2]=err1; error_VelocityForm[i-iMin,3]=err2;
        

# Compute error orders
        
t1=np.log(error_VelocityForm[0:-2,1]/error_VelocityForm[1:-1,1])
d1=np.mean(t1/np.log(2))

t2=np.log(error_VelocityForm[0:-2,2]/error_VelocityForm[1:-1,2])
d2=np.mean(t2/np.log(2))

t3=np.log(error_VelocityForm[0:-2,3]/error_VelocityForm[1:-1,3])
d3=np.mean(t3/np.log(2))


# Display table of Velocity form
print('')
print('Tables for P2 with fixed timestep size ')

'''
Let us define Vu = energy norm error of u, L2w = L2norm error of w, L2u = L2 norm error of u,
where w is a time derivative of u
'''
print(' %4s  %10s  %10s  %10s ' % ('h','Vu','L2w','L2u'))
for i in range(0,4):
    print(' 1/%2d ' % error_VelocityForm[i,0], end="")
    for j in range (1,4):
        print(' %10.4e ' % error_VelocityForm[i,j], end="")
    print('')
print(' %4s  %10.2f  %10.2f  %10.2f ' % ('rate',d1,d2,d3))
print('')