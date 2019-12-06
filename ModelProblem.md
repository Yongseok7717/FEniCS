# Mathematical model of viscoelasticity


Dynamic vicoelastic problem of generalised Maxwell solid is represented as a second kind of Volterra integral equation with expontially decaying kernel. Our aim is to solve the hyperbolic PDE with memory terms by spatially continuous Galerkin finite element method (CGFEM) and Crank-Nicolso finite difference scheme for time discretisation.


**We breifly introduce the model problem and give a numerical scheme.**
*Stability and error analysis as well as more details are seen in my research parper and PhD thesis (the link will appear here soon).*


Let $\Omega\in\mathbb{R}^d$ be our open bounded for $d=2,3$ and $[0,T]$ be a time domain for $T>0$. The model problem is given by
$$ \rho\ddot{u}(t)-\nabla\cdot D\nabla \left(u(t)-\sum_{{q}=1}^{N_\varphi}\psi_q(t)\right)={f}(t) $$ 
where $D>0$, $\rho$ is a density, $u$ is a displacement, $f$ is an external force and $\\{\psi_q \\} _ {q=1}^{N_\varphi}$ is a set of internal variables of displacement form defined by
$$\psi_{q}(t):=\frac{\varphi_{q}}{\tau_{q}}\int^t_0e^{-(t-s)/\tau_{q}}u(s)\ ds,\qquad\text{for } q=1,\ldots,N_\varphi,$$
for sets of positive constants $\\{\varphi_q\\}_ {q=0}^{N_\varphi}$ and $\\{\tau_q\\}_ {q=1}^{N_\varphi}$ with $\sum_{q=0}^{N_\varphi}{\varphi_q\}=1$.


For the stability, we impose pure Dirichlet boundary or mixed boundary with a positive measure of Dirichlet boundary $\Gamma_D$.
Let us assume a homogeneous Dirichlet boundary condition. Then we have the following boundary condition such that
\\begin{align}
u(t,x)&=0,&x\in \Gamma_D,\ \forall t\text{ (Dirichlet boundary)},\\\\
D\nabla \left(u(t)-\sum_{{q}=1}^{N_\varphi}\psi_q(t)\right)\cdot \boldsymbol{n}&=g_N,&x\in \Gamma_N,\ \forall t\text{ (Neumann boundary)},
\\end{align}where $\boldsymbol{n}$ is an outward normal vector.

In addition, use of integration by parts leads us to obtain internal variables of velocity form as following. 
$\psi_{q}(t)=\varphi_{q} u(t)-\varphi_{q}e^{-t/\tau_{q}}u_0-\zeta_{q}(t)$ where
$$ \zeta_{q}(t)=\int^t_0\varphi_{q}e^{-(t-s)/\tau_{q}} \dot u(s)\ ds.$$

Define a finite element space $V^h\subset V= \\{ H^1(\Omega)|v=0\  \text{for }x\in\Gamma_D \\} $ of Lagrange finite element. Then we can derive variational problem with respect to internal variables.
### Displacement form
\\begin{alignat}{2}
({\rho\ddot u(t)},{v})_ {L_2(\Omega)}+a(u(t),v)-\sum_{q=1}^{N_\varphi}a(\psi_{q}(t),v)
&=F_d(t;v)\qquad
&&\forall v\in V,\label{eq:disp}
\\\\
\tau_{q}a(\dot\psi_{q}(t),v)+a(\psi_{q}(t),v)
&=\varphi_{q}a(u(t),v) \qquad
&&\forall v\in V,\ q=1,\ldots,N_\varphi\label{eq:disp:ode} 
\\end{alignat}where $
a(w,v)=({D\nabla w}, {\nabla v})_ {L_2(\Omega)}$ and $F_d(t;v)=({f(t)},{v})_ {L_2(\Omega)}+(g_N(t),v)_ {L_2(\Gamma_N)}$
with $u(0)=u_0,$ $\dot u(0)=w_0$ and $\psi_{q}(0)=0, \ \forall {q}\in\\{1,\ldots,N_\varphi\\}$.

In a similar way, we can also obtain a weak formulation of velocity form.
### Velocity form
\\begin{alignat}{2}
({\rho\ddot u(t)},{v})_ {L_2(\Omega)}+\varphi_0a(u(t),v)+\sum_{q=1}^{N_\varphi}a(\zeta_{q}(t),v)
=F_v(t;v)\qquad
&&\forall v\in V,\label{eq:velo}
\\\\
\tau_{q}a(\dot\zeta_{q}(t),v)+a(\zeta_{q}(t),v)
=\tau_q\varphi_{q}a(\dot u(t),v) \qquad
&&\forall v\in V,\ q=1,\ldots,N_\varphi\label{eq:velo:ode} 
\\end{alignat}where 
$F_v(t;v)= F_d(t;v)
-\sum_{{q}=1}^{N_\varphi}\varphi_qe^{-t/\tau_{q}}a(u_0,v)$
with $u(0)=u_0,$ $\dot u(0)=w_0$ and $\zeta_{q}(0)=0, \ \forall {q}\in\\{1,\ldots,N_\varphi\\}$.


**\eqref{eq:disp:ode} and \eqref{eq:velo:ode} are governed by ODEs from differentiating internal variables.**


## Fully discrete formulation

Let us discretise time space. Define $\Delta t=T/N$ for $N\in\mathbb{N}$ and $t_n=n\Delta t$ for $n=0,\ldots,N$. We approximate the solution by
$$u(t_n,x)\approx Z_h^n(x)\in V^h,\qquad \dot u(t_n,x)\approx W_h^n(x)\in V^h\text{ for $n=0,\ldots,N$}$$based on Crank-Nicolson method with
$$\frac{W_h^{n+1}(x)+W_h^n(x)}{2}=\frac{Z_h^{n+1}(x)-Z_h^n(x)}{\Delta t}\text{ for $n=0,\ldots,N-1$}.$$
