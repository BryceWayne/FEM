import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint


def f(x, y):
	answer = -2*x*y*(2*exp(2*x)*(x**2+y**2)+2*exp(2*x)*x)
	answer += -(1+x**2*y)*(4*exp(2*x)*(x**2+y**2)+8*exp(2*x)*x+2*exp(2*x)) 
	answer += - 2*x**2*exp(2*x)*y-2*(1+x**2*y)*exp(2*x)
	return  answer
def kappa(x, y):
	return 1+x**2*y
def u(x,y):
	return np.exp(2*x)*(x**2 + y**2)
def ux(x,y):
	return 2*np.exp(2*x)*(x**2 + y**2)+2*np.exp(2*x)*x
def uy(x,y):
	return 2*np.exp(2*x)*y
def get_nodes(mesh, k):
	eptrs = mesh['Elements'][k-1,:]
	j = mesh['Edges']
	temp = abs(eptrs)
	j = j[temp-1,0:2]
	indices = np.zeros((3,), dtype=np.int)
	if eptrs[0] > 0:
		indices[0] = j[0,0]
		indices[1] = j[0,1]
	else:
		indices[0] = j[0,1]
		indices[1] = j[0,0]
	if eptrs[1] > 0:
		indices[2] = j[1,1]
	else:
		indices[2] = j[1,0]
	indices -= 1
	coords = mesh['Nodes']
	coords = coords[indices]
	ptrs   = mesh['NodePtrs']
	return coords, indices
def get_normal(mesh, i, j):
	e = mesh['Elements'][i-1,j]
	edges = mesh['Edges']
	c = mesh['Nodes'][edges[abs(e)-1,0:2]-1,:]
	n = np.array([[c[1,1]-c[0,1]],
			 [c[0,0]-c[1,0]]])
	if e < 0:
		n *= -1
	else:
		n *= 1
	n /= np.linalg.norm(n)
	return n
def get_tri_node_indices(mesh):
	Nt = len(mesh['Elements'])
	ElList = np.zeros((Nt, 3), dtype=int)
	for k in range(1, Nt+1):
		coords, indices = get_nodes(mesh, k)
		ElList[k-1] = indices.T
	return ElList
def plot_square_mesh(N, mesh):
	X, Y= mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	plt.triplot(trimesh, marker='o')
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.show()
def plot_solution(mesh, U, g):
	X, Y= mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	FNodePtrs = mesh['FNodePtrs'][:,0]
	Z[FNodePtrs - 1] = U
	CNodePtrs = mesh['CNodePtrs'][:,0]
	Z[CNodePtrs - 1] = g.reshape(len(g), 1)
	Z = Z.T[0]
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title("Clever Plot Title")
	sol = u(X,Y)
	ax.plot_trisurf(X, Y, Z, cmap=cm.Spectral_r, linewidth=0)
	# ax.plot_trisurf(X, Y, sol)
	# ax.plot_trisurf(trimesh, Z-sol, cmap=cm.jet, linewidth=5)
	plt.show()
# def rel_energy_norm():

def square_mesh_dirichlet(N, L):
	mesh = {}
	Nv = (N+1)**2
	Nodes = np.zeros((Nv,2))
	NodePtrs = np.zeros((Nv,1))
	Nf = (N-1)**2
	FNodePtrs = np.zeros((Nf, 1))
	Nc = Nv - Nf
	CNodePtrs = np.zeros((Nc, 1))
	Nt = 2*N**2
	Elements = np.zeros((Nt, 3))
	Ne = N + N*(3*N+1)
	Edges = np.zeros((Ne, 2))
	EdgeEls = np.zeros((Ne, 2))
	EdgeCFlags = np.zeros((Ne, 1))
	Nb = 0
	FBndyEdges = np.zeros((Nb, 1))
	k, kf, kc = 0, 0, 0
	dx = dy = L/N
	for j in range(N+1):
		y = j*dy
		for i in range(N+1):
			x = i*dx
			k +=1
			Nodes[k-1,:] = [x, y]
			if i in (0, N) or j in (0, N):
				kc += 1
				NodePtrs[k-1] = -kc
				CNodePtrs[kc-1] = k
			else:
				kf += 1
				NodePtrs[k-1] = kf
				FNodePtrs[kf-1] = k


	# Dirichlet Boundary
	for i in range(1, N+1):
		Edges[i-1,:] = [i, i+1]
		EdgeEls[i-1,:] = [2*i, 0]

	k = -1
	l = N

	for j in range(1, N+1):
		l += 1
		Edges[l-1,:] = [(j-1)*(N+1)+1, j*(N+1)+1]
		EdgeEls[l-1,:] = [2*N*(j-1)+1, 0]
		for i in range(1, N+1):
			k += 2
			Elements[k-1,:] = [-l,l+1,-(l+2*(N+1)-i)]
			Elements[k, :] = [l-N-i+1,l+2,-(l+1)]

			Edges[l,:] = [(j-1)*(N+1)+i,j*(N+1)+i+1]
			EdgeEls[l,:] = [k, k+1]

			Edges[l+1, :] = [(j-1)*(N+1)+i+1,j*(N+1)+i+1]
			if i < N:
				EdgeEls[l+1,:] = [k+1, k+2]
			else:
				EdgeEls[l+1,:] = [k+1, 0]
			l += 2

		for i in range(1, N+1):
			l += 1
			Edges[l-1,:]=[j*(N+1)+i,j*(N+1)+i+1];
			if j < N:
				EdgeEls[l-1,:]=[(j-1)*2*N+2*i-1,j*2*N+2*i]
			else:
				EdgeEls[l-1,:] = [(j-1)*2*N+2*i-1,0]

	mesh['Nodes'] = Nodes
	mesh['NodePtrs'] = NodePtrs.astype(int)
	mesh['FNodePtrs'] = FNodePtrs.astype(int)
	mesh['CNodePtrs'] = CNodePtrs.astype(int)
	mesh['Elements'] = Elements.astype(int)
	mesh['Edges'] = Edges.astype(int)
	mesh['EdgeEls'] = EdgeEls.astype(int)
	mesh['EdgeCFlags'] = EdgeCFlags.astype(int)
	mesh['FBndyEdges'] = FBndyEdges.astype(int)
	return mesh
def square_mesh_top_left(N, L):
	mesh = {"Degree": 1}
	Nv = (N+1)**2
	Nodes = np.zeros((Nv,2))
	NodePtrs = np.zeros((Nv,1), dtype=int)
	Nf = Nv - 2*N - 1
	FNodePtrs = np.zeros((Nf, 1), dtype=int)
	for j in range(1, N+1):
		index1, index2 = (j-1)*N, j*N
		array = [*range((j-1)*(N+1)+2, j*(N+1)+1)]
		FNodePtrs[index1:index2,0] = np.array(array, dtype=int)
	Nc = 2*N + 1
	CNodePtrsx = [*range(1, N**2+1, N+1)]
	CNodePtrsy = [*range(N**2+N+1, Nv+1)]
	CNodePtrs = np.array([CNodePtrsx + CNodePtrsy]).T
	temp = len(FNodePtrs)
	NodePtrs[FNodePtrs.astype(int).reshape(temp,)-1] = np.arange(1, Nf+1).reshape(temp,1)
	temp = len(CNodePtrs)
	NodePtrs[CNodePtrs.astype(int).reshape(temp,)-1] = -np.arange(1, Nc+1).reshape(temp,1)
	Nt = 2*N**2
	Elements = np.zeros((Nt, 3))
	Ne = N + N*(3*N+1)
	Edges = np.zeros((Ne, 2))
	EdgeEls = np.zeros((Ne, 2))
	EdgeCFlags = np.zeros((Ne, 1))
	Nb = 2*N
	FBndyEdges = np.zeros((Nb, 1))
	for i in range(1, N+1):
		FBndyEdges[i-1] = i
	for j in range(1, N+1):
		FBndyEdges[N+j-1] = N+(j-1)*(3*N+1)+2*N+1
	k = 0
	dx = dy = L/N
	for j in range(N+1):
		y = j*dy
		for i in range(N+1):
			x = i*dx
			Nodes[k,:] = [x, y]
			k +=1

	# Dirichlet Boundary
	for i in range(1, N+1):
		Edges[i-1,:] = [i, i+1]
		EdgeEls[i-1,:] = [2*i, -i]

	k = -1
	l = N

	for j in range(1, N+1):
		l += 1
		Edges[l-1,:]   = [(j-1)*(N+1)+1, j*(N+1)+1]
		EdgeEls[l-1,:] = [2*N*(j-1)+1, 0]
		for i in range(1, N+1):
			k += 2
			Elements[k-1,:] = [-l,l+1,-(l+2*(N+1)-i)]
			Elements[k, :] = [l-N-i+1,l+2,-(l+1)]
			Edges[l,:] = [(j-1)*(N+1)+i,j*(N+1)+i+1]
			EdgeEls[l,:] = [k, k+1]
			Edges[l+1, :] = [(j-1)*(N+1)+i+1,j*(N+1)+i+1]
			if i < N:
				EdgeEls[l+1,:] = [k+1, k+2]
			else:
				EdgeEls[l+1,:] = [k+1, -(N+j)]
			l += 2

		for i in range(1, N+1):
			l += 1
			Edges[l-1,:]=[j*(N+1)+i,j*(N+1)+i+1];
			if j < N:
				EdgeEls[l-1,:]=[(j-1)*2*N+2*i-1,j*2*N+2*i]
			else:
				EdgeEls[l-1,:] = [(j-1)*2*N+2*i-1,0]

	mesh['Nodes'] = Nodes
	mesh['NodePtrs'] = NodePtrs.astype(int)
	mesh['FNodePtrs'] = FNodePtrs.astype(int)
	mesh['CNodePtrs'] = CNodePtrs.astype(int)
	mesh['Elements'] = Elements.astype(int)
	mesh['Edges'] = Edges.astype(int)
	mesh['EdgeEls'] = EdgeEls.astype(int)
	mesh['EdgeCFlags'] = EdgeCFlags.astype(int)
	mesh['FBndyEdges'] = FBndyEdges.astype(int)
	return mesh

def stiffness(mesh):
	Nf = len(mesh['FNodePtrs'])
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	for k in range(1, len(mesh['Elements'])+1):
		c, ll = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][ll].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.lstsq(M, np.eye(3), rcond=None)[0], 14)
		G = C[1:3,:].T@C[1:3,:]
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					 [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		qpt = sum(c)/3
		try:
			qpt = sum(c)/3
			I = A*kappa(qpt[0], qpt[1])
		except:
			I = A
		for s in range(1,4):
			phi_i = sum([C[0, s-1]] + [C[i+1,s-1]*qpt[i] for i in range(2)])
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,s+1):
					phi_j = sum([C[0, r-1]] + [C[i+1,r-1]*qpt[i] for i in range(2)])
					llr = ll[r-1]
					if llr > 0:
						if llr <= lls:
							K[llr-1,lls-1] = K[llr-1, lls-1] + G[r-1,s-1]*I
						else:
							K[lls-1,llr-1] = K[lls-1, llr-1] + G[r-1,s-1]*I
	K = np.round(K + np.triu(K, 1).T, 14)
	# NONSYMMETRIC COMPONENT

	# plt.spy(K)
	# plt.show()
	return K
def get_dirichlet_data(mesh):
	try:
		CNodePtrs = (mesh['CNodePtrs']-1)[:,0]
		x = mesh['Nodes'][CNodePtrs,0]
		x.reshape(len(x),)
		y = mesh['Nodes'][CNodePtrs,1]
		y.reshape(len(y),)
		answer = u(x, y)
		return answer
	except:
		return np.zeros_like(mesh['CNodePtrs'])
def get_neumann_data(mesh):
	Nb = len(mesh['FBndyEdges'])
	h = np.zeros((Nb, 2))
	for j in range(1, Nb + 1):
		eptr = mesh['FBndyEdges'][j-1]
		e = mesh['Edges'][eptr-1][0]
		z1 = mesh['Nodes'][e[0]-1,:]
		z2 = mesh['Nodes'][e[1]-1,:]
		k = mesh['EdgeEls']
		k = k[eptr[0]-1, 0]
		elems = mesh['Elements']
		elems = elems[k-1,:]
		ll = np.where(elems == eptr)[0][0]
		nv = get_normal(mesh, k, ll)
		try:
			k1 = kappa(z1[0], z1[1])
			k2 = kappa(z2[0], z2[1])
		except:
			k1, k2 = 1, 1
		k1 = np.round(k1, 14)
		k2 = np.round(k2, 14)
		try:
			g1x, g1y = ux(z1[0], z1[1]), uy(z1[0], z1[1])
			g2x, g2y = ux(z2[0], z2[1]), uy(z2[0], z2[1])
		except:
			g1x, g1y = 0, 0
			g2x, g2y = 0, 0
		g1, g2 = [g1x, g1y], [g2x, g2y]
		g = np.array([g1, g2])
		hx = k1*np.dot(g1,nv)
		hy = k2*np.dot(g2,nv)
		h[j-1,:] = [hx, hy]
	return h
def load(mesh, f, k, g, h):
	Nf = len(mesh['FNodePtrs'])
	F  = np.zeros((Nf, 1))
	vo = np.ones((3,))
	Nt = len(mesh['Elements'])
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		qpt   = (1/3)*sum(c)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					 [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		fval = f(qpt[0], qpt[1])

		I = (A*fval)/3
		for r in range(1, 4):
			llr = ll[r-1]
			if llr > 0:
				F[llr-1] = F[llr-1] + I
		if not (g == np.zeros_like(g)).all():
			if ll[ll < 0].any():
				w = np.zeros((3,1))
				for j in range(3):
					if ll[j] < 0:
						w[j] = g[-ll[j]-1]
				M = np.zeros((3,3), dtype=float)
				M[:,0], M[:,1], M[:,2] = vo, c[:,0], c[:,1]
				C = np.round(np.linalg.inv(M), 14)
				h1 = C[1:3,:].T@(C[1:3,:]@w)
				try:
					qpt = sum(c)/3
					I = kappa(qpt[0], qpt[1])*A*h1
				except:
					I = A*h1
				I = I.T[0]
				for r in range(3):
					llr = ll[r]
					if llr > 0:
						F[llr -1] = F[llr-1] - I[r]
		if not (h == np.zeros_like(h)).all():
			for j in range(3):
				element = mesh['Elements'][k-1,:]
				edge = abs(mesh['Elements'][k-1,j])
				eptr = mesh['EdgeEls'][edge-1, 1]
				if eptr < 0:
					ii = mesh['Edges'][edge-1,0:3]
					c = mesh['Nodes'][ii-1,:]
					hval = 0.5*sum(h[abs(eptr)-1,:])
					norm = np.linalg.norm(c[0,:]-c[1,:])
					I = 0.5*norm*hval
					ll = mesh['NodePtrs'][ii-1].T[0]
					for r in range(1,3):
						llr = ll[r-1]
						if llr > 0:
							F[llr-1] = F[llr-1] + I
	return F


N = 25
# mesh = square_mesh_dirichlet(N, 1)
mesh = square_mesh_top_left(N, 1)
# plot_square_mesh(N, mesh)
K = stiffness(mesh)
g = get_dirichlet_data(mesh)
h = get_neumann_data(mesh)
F = load(mesh, f, kappa, g, h)
U = np.linalg.solve(K,F)
# pprint(U)
plot_solution(mesh, U, g)