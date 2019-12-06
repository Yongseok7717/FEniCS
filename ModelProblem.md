# Mathematical model of viscoelasticity


Dynamic vicoelastic problem of generalised Maxwell solid is represented as a second kind of Volterra integral equation with expontially decaying kernel. Our aim is to solve the hyperbolic PDE with memory terms by spatially continuous Galerkin finite element method (CGFEM) and Crank-Nicolso finite difference scheme for time discretisation.


**We breifly introduce the model problem and give a numerical scheme**
*Stability and error analysis as well as more details are seen in my research parper and PhD thesis (the link will be appear here soon).*


Let $\Omega\in\mathbb{R}^d$ be our open bounded for $d=2,3$. The model problem is given by
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

Define a finite element space $V^h\subset V= \\{ H^1(\Omega)|v=0\  \text{for }x\in\Gamma_D \\} $ of Lagrange finite element.
