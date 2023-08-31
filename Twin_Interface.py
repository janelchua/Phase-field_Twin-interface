# Edited code from NeedleTwins2_Testing14_2_withVisStress.py
# Janel Chua
#############################################################################
# Preliminaries and mesh
from dolfin import *
import numpy as np
import math
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
mesh = Mesh('Trial16_0_008.xml')

# Introduce manually the material parameters
Gc         = 32 # N/mm
l          = 0.1 # mm this goes into the H_l function and doesn't have dimensions of length
alpha      = 0.12 # This is the epsilon in front of grad phi in Vaibhav's paper, has dimensions of length
E_modulus  = 206*1000 # MPa = MKg.m^-1.s^-2 we are using N/mm^2, 1MPa = 1N/mm^2
nu         = 0.3 # Poisson's ratio
mu         = Constant(E_modulus / (2.0*(1.0 + nu))) # N/mm^2
lmbda      = Constant(E_modulus*nu / ((1.0 + nu)*(1.0 - 2.0*nu))) # N/mm^2
kappa      = 1
zeta       = 10
eps        = 1 # Value that goes in front of the term in W0 that makes the system want to go to either phi=1 or phi=0

# Mass density
rho = Constant(7.8*10**(-9)) # (N.s^2)/mm^4

# Rayleigh damping coefficients
eta_m = Constant(0)
eta_k = Constant(1e-9) # viscous stress parameter 

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5 + alpha_f - alpha_m)
beta    = Constant((gamma + 0.5)**2/4.)

# Time-stepping parameters
T       = 2 
Nsteps  = 10**6 # deltaT = 2e-6s
dt = Constant(T/Nsteps)

# Define traction
tract = Constant((100000, 0)) # MPa it takes ~275MPa to rip steel apart

#############################################################################
# Define Space for pristine Run
V = FunctionSpace(mesh, 'CG', 1)
p, q = TrialFunction(V), TestFunction(V)
pnew = Function(V)
pnewTemp = Function(V)
pnew2 = Function(V)
pold = Function(V)
# Make an interface in phi
def HL_initial(x1, x2):
    return 0.5*( 1 + ((exp(2*((x2 - x1 + 0.5)/l)) - 1)/(exp(2*((x2 - x1 + 0.5)/l))+1)) )
class InitialCondition(UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        value[0] = HL_initial(x[0], x[1])
pold.interpolate(InitialCondition())
storeHl = Function(V)
f, g = TrialFunction(V), TestFunction(V)
fnew = Function(V)
fnewTemp = Function(V)
fold = Function(V)

W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
u, v_ = TrialFunction(W), TestFunction(W)
# Current (unknown) displacement
unew = Function(W)
unewTemp = Function(W)
vnew = Function(W)
# Fields from previous time step (displacement, velocity, acceleration)
uold = Function(W)
vold = Function(W)
aold = Function(W)

#################################
# Added in for Needle Twins
# Interpolating
TS = TensorFunctionSpace(mesh, "CG", 1)
U1 = Function(TS)
U2 = Function(TS)
U1 = interpolate(Expression((("0.8958", "0"), 
                            ("0", "1.09659")),degree=2), TS)
U2 = interpolate(Expression((("1.09659", "0"), 
                            ("0", "0.8958")),degree=2), TS)
E1 = 0.5*(U1.T*U1 - Identity(2)) 
E2 = 0.5*(U2.T*U2 - Identity(2)) 

stress = Function(TS) # partial_W0/partial_F: Cauchy Stress

# Check Difference between interpolate and project
# :https://fenicsproject.org/qa/6832/what-is-the-difference-between-interpolate-and-project/

#############################################################################
# Boundary conditions
def top(x, on_boundary):
    return near(x[1], 1) and on_boundary #(x[0], 1)
def bot(x, on_boundary):
    return near(x[1], 0) and on_boundary #(x[0], 0)

def left(x, on_boundary):
    tol = 1E-15
    return abs(x[0] - 0.0) <= DOLFIN_EPS and on_boundary #(0, x[1])
def right(x, on_boundary):
    tol = 1E-15
    return abs(x[0] - 2) <= DOLFIN_EPS and on_boundary #(2, x[1])

def leftcorner(x, on_boundary):
    tol = 1E-15
    return (abs(x[0] + 0) < tol) and (abs(x[1] + 0) < tol) #(0,0)
def rightcorner(x, on_boundary):
    tol = 1E-15
    return (abs(x[0] - 2) < tol) and (abs(x[1] + 0) < tol) #(2,0)

def Crack(x):
    return abs(x[1]) < 5e-03 and x[0] <= -0.25 

loadtop = Expression("t", t = 0.0, degree = 1)
loadbot = Expression("t", t = 0.0, degree = 1)
loadright = Expression("t", t = 0.0, degree = 1)
bcbot= DirichletBC(W.sub(1), loadbot, bot) # Bottom displacement loaded
bctop = DirichletBC(W.sub(1), loadtop, top) # Top displacement loaded
bcleft = DirichletBC(W.sub(0), Constant(0), left) # u1=0, Left boundary not allowed to move horizontally
bcright = DirichletBC(W.sub(0), loadright, right) # Right boundary with loading rate

bcs = [bcleft]
bc_u = [bcleft, bcright]
bc_phi = []
bc_f = []
n = FacetNormal(mesh)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)

AutoSubDomain(right).mark(boundary_subdomains, 1)
# Define measure for boundary condition integral
dss = Measure("ds")(subdomain_data = boundary_subdomains)

#############################################################################
# Constituive functions
def HL(p):
    return 0.5*( 1 + ((exp(2*((p - 0.5)/l)) - 1)/(exp(2*((p - 0.5)/l))+1)) )
def W0(p, u, v):
    d = u.geometric_dimension()
    F = variable(grad(u)+Identity(d))
    C = F.T*F
    EE = 0.5*(C - Identity(d))
    A = EE - E1
    B = EE - E2
    # energy = (1 - HL(p))*0.5*(lmbda*(tr(A))**2 + 2*mu*inner(A, A)) + (HL(p))*0.5*(lmbda*(tr(B))**2 + 2*mu*inner(B, B))
    energy = (1 - HL(p))*0.5*(lmbda*(tr(A))**2 + 2*mu*inner(A, A)) + (HL(p))*0.5*(lmbda*(tr(B))**2 + 2*mu*inner(B, B)) + eps*(p**2)*(1-p)**2
    stress_elas = diff(energy, F)

    Fdot = grad(v)
    Tvis = eta_k*0.5*(Fdot*inv(F) + inv(F).T*Fdot.T)
    stress_vis  = det(F)*Tvis*inv(F).T
    stress = stress_elas + stress_vis
    W0_diffPhi = diff(energy, p)
    return [energy, stress, W0_diffPhi]

def E(u): # 0.5*(C - I)
    d = u.geometric_dimension()
    F = grad(u)+Identity(d)
    C = F.T*F
    return 0.5*(C - Identity(d))
def A(u):
   return E(u) - E1
def B(u):
   return E(u) - E2
def W0_diffPhi(p, u):
    HLdiffPhi = Expression('pow(1/(cosh(p - 0.5/l)),2)/(2*l)', l = l, p = p, degree =2)
    return (1 - HLdiffPhi)*0.5*(lmbda*(tr(A(u)))**2 + 2*mu*inner(A(u), A(u))) + (HLdiffPhi)*0.5*(lmbda*(tr(B(u)))**2 + 2*mu*inner(B(u), B(u)))

def v_hat(f):
    sign = Expression('f> 0 ? 1: f< 0 ? -1: 0', f = f, degree = 2) 
    sign_pjt = project(sign, V)
    sign_nodal_values = sign_pjt.vector()
    sign_array = sign_nodal_values.get_local()
    print(sign_array, 'sign array')
    absf_pjt = project(abs(f), V)
    absf_nodal_values = absf_pjt.vector()
    absf_array = absf_nodal_values.get_local()
    print(absf_array, 'absf array')
    return sign*kappa*abs(f)# 

def H(e):
    return 0.5*( 1 + ((exp(2*((e)/l)) - 1)/(exp(2*((e)/l))+1)) )

def G_nucleation(p, u):
    G0 = 1e4 # Modify this to increase the strength of the nucleation driving force term G
    # max Eigenvalue for E1
    E100, E101, E110, E111 = E1[0,0], E1[0,1], E1[1,0], E1[1,1]
    print(E1[0,0],'E1[0,0]')
    E1a = 1
    E1b = (-E100-E111)
    E1c = (-E101**2 + E100*E111)
    E1_max = (-E1b + sqrt(E1b**2 - 4*E1a*E1c))/(2*E1a)
    # max eigenvalue for E2
    E200, E201, E210, E211 = E2[0,0], E2[0,1], E2[1,0], E2[1,1]
    print(E2[0,0],'E2[0,0]')
    E2a = 1
    E2b = (-E200-E211)
    E2c = (-E201**2 + E200*E211)
    E2_max = (-E2b + sqrt(E2b**2 - 4*E2a*E2c))/(2*E2a)

    e_m = 0.1 # this is the strain at which we want a phase change in the case of Equation 6.2
    print(e_m,'e_m')
    Strain = E(u)
    e00, e01, e10, e11 = Strain[0,0], Strain[0,1], Strain[1,0], Strain[1,1]
    print(Strain[0,0],'Strain[0,0]')
    a = 1
    b = (-e00-e11)
    c = (-e01**2 + e00*e11)
    e_max = (-b + sqrt(b**2 - 4*a*c))/(2*a)
    # G_nuc = G0*( (1 - HL(p))*H(e00 - e_m)  ) # G used in Equation 6.2
    G_nuc = G0*( (1 - HL(p))*(1 - H(e11 - e00)) + HL(p)*(-H(e11 - e00)) ) # G in Equation 6.5

    return G_nuc

# Mass form
def m(a, v_):
    return rho*inner(a, v_)*dx
# Elastic stiffness form 
def k(p, u, v, v_):
    return inner(W0(p, u, v)[1], sym(grad(v_)))*dx 
# Work of external forces
def Wext(p, v_):
    return dot(v_, tract)*dss(1)
def preventHeal(pold, pnew):
    coor = mesh.coordinates()
    p_cr = 0.995
    pnew_nodal_values = pnew.vector()
    pnew_array = pnew_nodal_values.get_local()
    pold_nodal_values = pold.vector()
    pold_array = pold_nodal_values.get_local()
    for i in range(len(pold_array)):
        if pnew_array[i] < pold_array[i]:
            pnew_array[i] = pold_array[i]
        elif pold_array[i] > p_cr and pnew_array[i] >= pold_array[i]:
            pnew_array[i] = pnew_array[i]
        elif pold_array[i] <= p_cr:
            pnew_array[i] = pnew_array[i]
    #Reverse the projection
    pnew2.vector()[:] = pnew_array[:]   
    return pnew2

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(unew, uold, vold, aold, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (unew-uold-dt_*vold)/beta_/dt_**2 - (1-2*beta_)/2/beta_*aold

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, uold, vold, aold, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return vold + dt_*((1-gamma_)*aold + gamma_*a)

def update_fields(unew, uold, vold, aold):
    """Update fields at the end of each time step."""
    # Get vectors (references)
    u_vec, u0_vec  = unew.vector(), uold.vector()
    v0_vec, a0_vec = vold.vector(), aold.vector()
    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    # Update (y_old <- y)
    vold.vector()[:], aold.vector()[:] = v_vec, a_vec
    uold.vector()[:] = unew.vector()		

def avg(xold, xnew, alpha):
    return alpha*xold + (1-alpha)*xnew
    
#############################################################################
# # Minimization to set initial strains
# Total potential energy: minimized with respect to displacement while keeping phi fixed
Pi = W0(pold, uold, vold)[0]*dx + 0.5*alpha*dot(grad(pold),grad(pold))*dx #Check this.... 
FF = derivative(Pi, uold, v_) 
J = derivative(FF, uold, u)
problem_minimization = NonlinearVariationalProblem(FF, uold, bcs, J)
solver_minimization = NonlinearVariationalSolver(problem_minimization)
prm_min = solver_minimization.parameters
info(prm_min, True)
prm_min["nonlinear_solver"] = "newton"
prm_min["newton_solver"]["absolute_tolerance"] = 1E-8
prm_min["newton_solver"]["relative_tolerance"] = 1E-8
solver_minimization.solve() 

#############################################################################
# Initialization of the output requests
store_phi = File ("mydata/phi.pvd")
store_Hl = File ("mydata/Hl.pvd")
store_u = File ("mydata/u.pvd")
store_vel = File ("mydata/vel.pvd")

sigma_fs = TensorFunctionSpace(mesh, "CG", 1)
stress_total = Function(sigma_fs, name='Stress')
store_stress_total = File ("mydata/stress_total.pvd")
stress_elas = Function(sigma_fs, name='Stress')
store_stress_elas = File ("mydata/stress_elas.pvd")
stress_vis = Function(sigma_fs, name='Stress')
store_stress_vis = File ("mydata/stress_vis.pvd")

strain = TensorFunctionSpace(mesh, "CG", 1)
strain_total = Function(strain, name='Strain')
store_strain_total = File ("mydata/strain_total.pvd")
store_strain_11 = File ("mydata/strain_11.pvd")
store_strain_11_minus_strain_22 = File ("mydata/strain_11minus22.pvd")

poynting = VectorFunctionSpace(mesh, "CG", 1)
poynting_temp = Function(poynting, name='poynting')
store_poynting = File ("mydata/poynting.pvd")

#############################################################################
# Storing things at t = 0 (already minimized):
store_phi << pold
storeHl.assign(project(HL(pold), V, solver_type="cg", preconditioner_type="amg"))
store_Hl << storeHl
store_u << uold # mm
store_vel << vold # mm/s
stress_total.assign(project(W0(pold, uold, vold)[1], sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
store_stress_total << stress_total
# epsilon_11 
# split :
# https://fenicsproject.org/qa/13017/ordering-of-u-split-for-tensorfunctionspace/
strain_total.assign(project(E(uold), strain, solver_type="cg", preconditioner_type="amg"))
store_strain_total << strain_total
[e_11, e_12, e_21, e_22] = strain_total.split(True) 
store_strain_11 << e_11 # e_11.vector() has len that is same as number of nodes 
print(e_11.vector()[:])
print ('Saving initial condition')

#############################################################################
# Quasistatics : minimized with respect to displacement and phi
unew.assign(uold)
vnew.assign(vold)
pnew.assign(pold)
# Weak form for momentum balance
E_u_q = inner(W0(pnew, unew, vnew)[1], sym(grad(v_)))*dx 
J_u_q = derivative(E_u_q, unew, u)
problem_disp_q = NonlinearVariationalProblem(E_u_q, unew, bcs, J_u_q)
solver_disp_q = NonlinearVariationalSolver(problem_disp_q)
prm_q = solver_disp_q.parameters
info(prm_q, True)
prm_q["nonlinear_solver"] = "newton"
prm_q["newton_solver"]["absolute_tolerance"] = 1E-8
prm_q["newton_solver"]["relative_tolerance"] = 1E-8
prm_q["newton_solver"]["maximum_iterations"] = 100

# Phi equation steepest descent
P_q = -((W0(pnew, unew, vnew)[2]*q + alpha*inner(grad(pnew),grad(q))) )*dx 
J_p_q = derivative(P_q, pnew, p)
problem_phi_q = NonlinearVariationalProblem(P_q, pnew, bc_phi, J_p_q)
solver_phi_q = NonlinearVariationalSolver(problem_phi_q)
prm4_q = solver_phi_q.parameters
info(prm4_q, True)
prm4_q["nonlinear_solver"] = "newton"
prm4_q["newton_solver"]["absolute_tolerance"] = 1E-8
prm4_q["newton_solver"]["relative_tolerance"] = 1E-8
prm4_q["newton_solver"]["maximum_iterations"] = 100
#prm4_q['newton_solver']['relaxation_parameter'] = 1.0

#############################################################################
# Dyamics
# Weak form for momentum balance
anew = update_a(unew, uold, vold, aold, ufl=True)
vnew = update_v(anew, uold, vold, aold, ufl=True)
res = m(avg(aold, anew, alpha_m), v_) \
      + k(pnewTemp, avg(uold, unew, alpha_f), avg(vold, vnew, alpha_f), v_) #- Wext(pnew2, v_)
J_u = derivative(res, unew, u)
problem_disp = NonlinearVariationalProblem(res, unew, bc_u, J_u)
solver_disp = NonlinearVariationalSolver(problem_disp)
prm = solver_disp.parameters
info(prm, True)
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1E-6
prm["newton_solver"]["relative_tolerance"] = 1E-6
prm["newton_solver"]["maximum_iterations"] = 100

# # Phi evolution equations:
# F equation
F = (-inner(fnew, g)  -  W0(pnewTemp, unew, vnew)[2]*g  -  alpha*inner(grad(pnewTemp), grad(g)))*dx
J_f = derivative(F, fnew, f)
problem_f = NonlinearVariationalProblem(F, fnew, bc_f, J_f)
solver_f = NonlinearVariationalSolver(problem_f)
prm2 = solver_f.parameters
info(prm2, True)
prm2["nonlinear_solver"] = "newton"
prm2["newton_solver"]["absolute_tolerance"] = 1E-8
prm2["newton_solver"]["relative_tolerance"] = 1E-8
prm2["newton_solver"]["maximum_iterations"] = 100

# Phi equation
grad_p = project(grad(pnew), W)
grad_p_inner  = inner(grad_p, grad_p)*dx 
grad_p_norm = sqrt(assemble(grad_p_inner)) 
P = (-inner(pnew, q) + inner(pold, q) + dt*(grad_p_norm*v_hat(fnew)*q + G_nucleation(pnew, unew)*q))*dx 
J_p = derivative(P, pnew, p)
problem_phi = NonlinearVariationalProblem(P, pnew, bc_phi, J_p)
solver_phi = NonlinearVariationalSolver(problem_phi)
prm3 = solver_phi.parameters
info(prm3, True)
prm3["nonlinear_solver"] = "newton"
prm3["newton_solver"]["absolute_tolerance"] = 1E-8
prm3["newton_solver"]["relative_tolerance"] = 1E-8
prm3["newton_solver"]["maximum_iterations"] = 100

#############################################################################
# Initialization of the iterative procedure
time = np.linspace(0, T, Nsteps+1)
tol = 1e-3

#############################################################################
# Looping through time here.
for (i, dt) in enumerate(np.diff(time)):

    if i>1:# This means Quasistatic (line 502 onwards) will run for i=0 and i=1 before dynamics starts
        print ('DYNAMIC NOW')
        if i>=0:
            u_r = 1e3
        else:
            u_r = 0

        t = time[i+1]
        print("Time: ", t)
        ## Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
        if t <= (4/4)*T:
            loadright.t = t*u_r
            print ('displacement', loadright.t)
        iter = 0
        err = 1

        while err > tol:
            iter += 1
            # Solve for new displacement, f and phi  
            solver_disp.solve()
            solver_f.solve()
            solver_phi.solve()
            # Prevent healing using Bourdin's way
            pnew2 = preventHeal(pnewTemp, pnew)

            # Calculate error
            err_u = errornorm(unew, unewTemp, norm_type = 'l2',mesh = None)
            err_f = errornorm(fnew, fnewTemp, norm_type = 'l2',mesh = None)
            err_phi = errornorm(pnew2, pnewTemp, norm_type = 'l2',mesh = None)
            err = max(err_u, err_f, err_phi)
        	
            # Update new fields in same timestep with new calculated quantities
            unewTemp.vector()[:] = unew.vector()
            fnewTemp.assign(fnew)
            pnewTemp.assign(pnew2)
            print ('Iterations:', iter, ', Total time', t, ', error', err)
    	   
            if err < tol:
                # Update old fields from previous timestep with new quantities
                update_fields(unew, uold, vold, aold)
                fold.assign(fnew)
                pold.assign(pnew2)
                print ('err < tol :D','Iterations:', iter, ', Total time', t, ', error', err)
                if round(t*1e6) % 2 == 0: # each saved data point is 2e-6s
                    store_phi << pold
                    storeHl.assign(project(HL(pold), V, solver_type="cg", preconditioner_type="amg"))
                    store_Hl << storeHl
                    store_u << uold # mm
                    store_vel << vold # mm/s
                                    
                    stress_total.assign(project(W0(pold, uold, vold)[1], sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                    store_stress_total << stress_total

                    # epsilon_11 
                    # split: https://fenicsproject.org/qa/13017/ordering-of-u-split-for-tensorfunctionspace/
                    strain_total.assign(project(E(uold), strain, solver_type="cg", preconditioner_type="amg"))
                    store_strain_total << strain_total
                    [e_11, e_12, e_21, e_22] = strain_total.split(True) 
                    store_strain_11 << e_11 # e_11.vector() has len that is same as number of nodes

                    File('mydata/saved_mesh.xml') << mesh
                    File('mydata/saved_phi.xml') << pold
                    File('mydata/saved_u.xml') << uold
                    File('mydata/saved_v.xml') << vold
                    File('mydata/saved_a.xml') << aold

                    print ('Iterations:', iter, ', Total time', t, 'Saving datapoint')

    else:
        print ('QUASISTATIC NOW')
        u_r = 0
        t = time[i+1]
        print("Time: ", t)

        ## Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
        if t <= (4/4)*T:
            loadright.t = t*u_r
            print ('displacement', loadright.t)
        iter = 0
        err = 1

        while err > tol:
            iter += 1
            # Solve for new displacement, f and phi  
            solver_disp_q.solve()
            solver_phi_q.solve()

            # Prevent healing using Bourdin's way
            pnew2 = preventHeal(pnewTemp, pnew)

            # Calculate error
            err_u = errornorm(unew, unewTemp, norm_type = 'l2',mesh = None)
            err_phi = errornorm(pnew2, pnewTemp, norm_type = 'l2',mesh = None)
            err = max(err_u, err_phi)
            
            # Update new fields in same timestep with new calculated quantities
            unewTemp.vector()[:] = unew.vector()
            pnewTemp.assign(pnew2)
            print ('Iterations:', iter, ', Total time', t, ', error', err)
           
            if err < tol:
                # Update old fields from previous timestep with new quantities
                update_fields(unew, uold, vold, aold)
                pold.assign(pnew2)
                print ('err < tol :D','Iterations:', iter, ', Total time', t, ', error', err)
                if round(t*1e6) % 2 == 0: # each saved data point is 2e-6s
                    store_phi << pold
                    storeHl.assign(project(HL(pold), V, solver_type="cg", preconditioner_type="amg"))
                    store_Hl << storeHl
                    store_u << uold # mm
                    store_vel << vold # mm/s
                                    
                    stress_total.assign(project(W0(pold, uold, vold)[1], sigma_fs, solver_type="cg", preconditioner_type="amg")) # 1MPa = 1N/mm^2
                    store_stress_total << stress_total

                    # epsilon_11 
                    # split: https://fenicsproject.org/qa/13017/ordering-of-u-split-for-tensorfunctionspace/
                    strain_total.assign(project(E(uold), strain, solver_type="cg", preconditioner_type="amg"))
                    store_strain_total << strain_total
                    [e_11, e_12, e_21, e_22] = strain_total.split(True) 
                    store_strain_11 << e_11 # e_11.vector() has len that is same as number of nodes

                    File('mydata/saved_mesh.xml') << mesh
                    File('mydata/saved_phi.xml') << pold
                    File('mydata/saved_u.xml') << uold
                    File('mydata/saved_v.xml') << vold
                    File('mydata/saved_a.xml') << aold

                    print ('Iterations:', iter, ', Total time', t, 'Saving datapoint')

print ('Simulation completed') 
#############################################################################
