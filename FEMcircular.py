import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from PIL import Image
import pandas as pd
import os, sys
import pickle
from tqdm import tqdm
from scipy.optimize import curve_fit
import scipy


def sech(x):
	return 1/np.cosh(x)
def f(x, y, t=0):
	r2 = x**2 + y**2
	"""
		e(-2*pi*r^2)*cos(2*pi*r^2)
	"""
	# answer = -8*np.pi*np.exp(-2*np.pi*r2)*((4*np.pi*r2 - 1)*np.sin(2*np.pi*r2) - np.cos(2*np.pi*r2))
	"""
		sech^2(r^2)
	"""
	# answer = -4*sech(r2)**4*np.sin(2*np.pi*t)*(-4*r2 + 2*r2*np.cosh(2*r2) - np.sinh(2*r2)) + u(x,y)
	"""
		# x^2-y^2/(r2 + 1)
	"""
	# answer = -(4*(3*x**2 + x**4 - y**2*(3 + y**2)))/(1 + r2)**3 - (x**2-y**2)/(1 + r2) + u(x,y)
	"""
		Modified e(-r^2)*cos(r^2)
	"""
	# answer = 4*np.pi*np.exp(-np.pi*r2)*((15*np.pi*r2 + 1)*np.cos(4*np.pi*r2) - 4*(2*np.pi*r2 - 1)*np.sin(4*np.pi*r2))
	"""
		HW3P1
	"""
	# answer = -2*x*y*(2*np.exp(2*x)*r2+2*np.exp(2*x)*x)-kappa(x,y)*(4*np.exp(2*x)*r2+8*exp(2*x)*x+2*np.exp(2*x))-2*x**2*np.exp(2*x)*y-2*kappa(x,y)*np.exp(2*x)+u(x,y)
	"""
		HW3P2
	"""
	# answer = -2*x*y*(2*np.exp(2*x)*(r2)+2*np.exp(2*x)*x)-(1+x**2*y)*(4*np.exp(2*x)*(r2)+8*np.exp(2*x)*x+2*np.exp(2*x))-2*x**2*np.exp(2*x)*y-2*(1+x**2*y)*np.exp(2*x)+ux(x,y)+uy(x,y)
	"""
		BRYCE 101
	"""
	# COMBINED WITH HW3P1
	# answer = -2*np.pi*np.sin(2*np.pi*t)*u(x,y,t) + np.cos(2*np.pi*t)*answer
	# COMBINED WITH HW3P2
	# answer = -2*np.pi*np.sin(2*np.pi*t)*u(x,y) + np.cos(2*np.pi*t)*answer
	# COMBINED WITH SECH^2 ## ALSO COMBINED WITH POTATO CHIP
	# answer *= np.cos(2*np.pi*x*t)
	# HONG EXAMPLE
	# answer = 4*(1 + 2*y + x*(2 + 3*y)) + u(x,y)
	# answer = answer*np.cos(2*np.pi*t) -2*np.pi*np.sin(2*np.pi*t)*u(x,y)
	# answer = -8*np.exp(-2*np.pi*r2)*np.pi*(-1 + 2*np.pi*r2) + u(x,y)
	answer = np.cos(2*np.pi*r2-2*np.pi*t)
	return  answer
def kappa(x, y):
	r2 = x**2 + y**2
	return 1
	# return r2
	# return 1+x**2*y
	# return np.sqrt(r2)
def u(x,y, t=0):
	r2 = x**2 + y**2
	# return np.exp(-2*np.pi*r2)*np.cos(2*np.pi*r2)
	# return sech(r2)**2
	# return (x**2-y**2)/(1 + r2)
	# return np.exp(-2*np.pi*r2)
	# return np.exp(2*x)*r2
	# return np.exp(2*x)*r2*np.cos(2*np.pi*t)
	# return 1-r2
	return np.zeros_like(x)
	# return (1+x)*(1+y)*(1-r2)*np.cos(2*np.pi*t)
def ux(x,y,t=0):
	r2 = x**2 + y**2
	# return -2*2*np.pi*x*np.cos(np.pi*r2)
	# return -4*sech(r2)**2*np.tanh(r2)*x
	# return (3*x**2 - y**2)/(1 + r2)
	# return 2*np.exp(2*x)*(r2)+2*np.exp(2*x)*x
	# return np.cos(2*np.pi*t)*(2*np.exp(2*x)*(r2)+2*np.exp(2*x)*x)
	# return -2*x
	# return -2*np.pi*2*x*u(x,y)
	return np.zeros_like(x)
	# return ((1+y)*(1-r2)-2*x*(1+x)*(1+y))*np.cos(2*np.pi*t)
def uy(x,y,t=0):
	r2 = x**2 + y**2
	# return -2*2*np.pi*y*np.cos(np.pi*r2)
	# return -4*sech(r2)**2*np.tanh(r2)*y
	# return -(2*x*y)/(1 + r2)
	# return 2*np.exp(2*x)*y
	# return np.cos(2*np.pi*t)*(2*np.exp(2*x)*y)
	# return -2*y
	# return -2*np.pi*2*y*u(x,y)
	return np.zeros_like(x)
	# return ((1+x)*(1-r2)+(1+x)*(1+y)*(-2*y))*np.cos(2*np.pi*t)
"""
MESH BOUNDARY FUNCTIONS
"""
def circlef(p1, p2, k=2):
	if len(p1) is not 2:
		print("First input must be a vector of length 2.")
	if len(p2) is not 2:
		print("Second input must be a vector of length 2.")
	n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
	th1, th2 = np.arctan2(p1[1], p1[0]), np.arctan2(p2[1], p2[0])
	if abs(th1-th2) > np.pi:
		if th1 < th2:
			th1 = th1 + 2*np.pi
		else:
			th2 = th2 + 2*np.pi
	if th1 < th2:
		dth = (th2 - th1)/k
		th = np.array([np.linspace(th1 + dth, th2 - dth, k-1, endpoint=True)]).T[0][0]
		pts = n1*np.array([np.cos(th), np.sin(th)])

	else:
		dth = (th1 - th2)/k
		th = np.array([np.linspace(th2 + dth, th1 - dth, k-1, endpoint=True)]).T[0][0]
		pts = np.flipud(n1*np.array([np.cos(th), np.sin(th)]))
	return pts.T
def ellipsef(p1, p2, a=1, b=1, k=2):
	if len(p1) is not 2:
		print("First input must be a vector of length 2.")
	if len(p2) is not 2:
		print("Second input must be a vector of length 2.")
	th1, th2 = np.arctan2(p1[1]/b, p1[0]/a), np.arctan2(p2[1]/b, p2[0]/a)
	if abs(th1-th2) > np.pi:
		if th1 < th2:
			th1 = th1 + 2*np.pi
		else:
			th2 = th2 + 2*np.pi
	if th1 < th2:
		dth = (th2 - th1)/k
		th = np.array([np.linspace(th1 + dth, th2 - dth, k-1, endpoint=True)]).T[0][0]
		pts = np.array([a*np.cos(th), b*np.sin(th)])

	else:
		dth = (th1 - th2)/k
		th = np.array([np.linspace(th2 + dth, th1 - dth, k-1, endpoint=True)]).T[0][0]
		pts = np.flipud(np.array([a*np.cos(th), b*np.sin(th)]))
	return pts.T
"""
PLOTTING FUNCTIONS
"""
def plot_matrix_density(mesh, N, K):
	plt.close()
	plt.figure(figsize=(10,6))
	plt.spy(K)
	plt.title(f'Stiffness Matrix: N={N}')
	plt.savefig(f'matrix_density{N}.png')
	plt.close()
def plot_circle_mesh(mesh, N):
	plt.close()
	X, Y= mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	plt.figure(figsize=(6,6))
	theta = np.linspace(0, 2*np.pi, 1000, endpoint=True)
	x, y = np.cos(theta), np.sin(theta)
	plt.triplot(trimesh, marker='o', markersize=2, c='b', linewidth=1) #
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.plot(x,y,'r-')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.tight_layout()
	plt.axis('equal')
	plt.grid(alpha=0.618)
	plt.savefig(f'meshes/circular_mesh{N}.png')
	plt.close()
def plot_elliptic_mesh(mesh, N):
	a, b = mesh['a'], mesh['b']
	plt.close()
	X, Y= mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	plt.figure(figsize=(6,6))
	theta = np.linspace(0, 2*np.pi, 1000, endpoint=True)
	x, y = a*np.cos(theta), b*np.sin(theta)
	plt.triplot(trimesh, marker='o', markersize=2, c='b', linewidth=1) #
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.plot(x,y,'r-')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.tight_layout()
	plt.axis('equal')
	plt.grid(alpha=0.618)
	plt.savefig(f'meshes/elliptic_mesh{N}.png')
	# plt.show()
	plt.close()
def plot_init_solution(mesh, U, g, N, t=0):
	triangles = get_tri_node_indices(mesh)
	X, Y = mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	Z = np.zeros((len(X), 1))
	FNodePtrs = mesh['FNodePtrs'][:,0]
	Z[FNodePtrs - 1] = U
	CNodePtrs = mesh['CNodePtrs'][:,0]
	Z[CNodePtrs - 1] = g.reshape(len(g), 1)
	Z = np.round(Z.T[0], 14)
	sol = u(X,Y,t)
	# ZMIN, ZMAX = min([sol.min(), Z.min()]), max([sol.max(), Z.max()])
	# DIFFMIN, DIFFMAX = min([(sol-Z).min(), (Z-sol).min()]), max([(sol-Z).max(), (Z-sol).max()])
	"""
	APPROX SOL PIC
	"""
	plt.close()
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	ax.set_zlim(-.05, .05)
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title(f"Approximate: N={N}, t={t}")
	surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
	fig.colorbar(surf, shrink=0.62, aspect=5)
	plt.savefig(f'approx{N}{str(int(100*t)).zfill(3)}.png')
	plt.close()
	"""
	EXACT SOLUTION PIC
	"""
	# plt.close()
	# fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	# plt.xlim(X.min(), X.max())
	# plt.ylim(Y.min(), Y.max())
	# ax.set_zlim(-1, 1)
	# plt.xlabel("$x$")
	# plt.ylabel("$y$")
	# plt.title(f"Exact: N={N}, t={t}")
	# surf = ax.plot_trisurf(X, Y, sol, cmap=cm.jet, linewidth=0)
	# fig.colorbar(surf, shrink=0.62, aspect=5)
	# plt.savefig(f'sol{N}{str(int(100*t)).zfill(3)}.png')
	# plt.close()
	"""
	DIFFERENCE PIC
	"""
	# plt.close()
	# fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	# # plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	# plt.xlim(X.min(), X.max())
	# plt.ylim(Y.min(), Y.max())
	# ax.set_zlim(DIFFMIN,DIFFMAX)
	# plt.xlabel("$x$")
	# plt.ylabel("$y$")
	# plt.title(f"Difference: N={N}, t={t}")
	# surf = ax.plot_trisurf(trimesh, Z-sol, cmap=cm.jet, linewidth=0)
	# fig.colorbar(surf, shrink=0.62, aspect=5)
	# plt.savefig(f'diff{N}{str(int(100*t)).zfill(3)}.png')
	# plt.close()
	"""
	MERGE APPROX AND EXACT
	"""
	# plist = [f'approx{N}{str(int(100*t)).zfill(3)}.png', f'sol{N}{str(int(100*t)).zfill(3)}.png']
	# concatenate_pics(plist, N, t)
def gif_creator(mesh, U, g):
	plt.close()
	X, Y = mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	FNodePtrs = mesh['FNodePtrs'][:,0]
	Z[FNodePtrs - 1] = U
	CNodePtrs = mesh['CNodePtrs'][:,0]
	Z[CNodePtrs - 1] = g.reshape(len(g), 1)
	Z = Z.T[0]
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	# plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.xticks([])
	plt.yticks([])
	plt.ylabel("$y$")
	plt.title(f"Approximate Solution")
	ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
	plist = []
	count = 0
	for ii in range(46):
		ax.view_init(elev=ii, azim=0)
		plt.savefig(f'gifs/approx{str(count).zfill(3)}.png')
		image = Image.open(f'gifs/approx{str(count).zfill(3)}.png')
		plist.append(image)
		count += 1
	for ii in range(361):
		ax.view_init(elev=45, azim=ii)
		plt.savefig(f'gifs/approx{str(count).zfill(3)}.png')
		image = Image.open(f'gifs/approx{str(count).zfill(3)}.png')
		plist.append(image)
		count += 1
	degs = list(range(46))
	degs.reverse()
	for ii in degs:
		ax.view_init(elev=ii, azim=0)
		plt.savefig(f'gifs/approx{str(count).zfill(3)}.png')
		image = Image.open(f'gifs/approx{str(count).zfill(3)}.png')
		plist.append(image)
		count += 1
	plt.close()
def concatenate_pics(plist, N, t=0):
	images = [Image.open(i) for i in plist]
	widths, heights = zip(*(i.size for i in images))
	total_width = sum(widths)
	max_height = max(heights)
	new_im = Image.new('RGB', (total_width, max_height))
	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset,0))
		x_offset += im.size[0]
	new_im.save(f'merged{N}{str(int(100*t)).zfill(3)}.png')
	for pic in plist:
		os.remove(pic)
def save_obj(obj, name):
    with open('meshes/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open('meshes/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
"""
ENERGY NORM FUNCTIONS
"""
def energy_norm_error(mesh, kappa, ux, uy, U, g):
	m = 0
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		u = np.zeros((3,1))
		for j in range(3):
			if ll[j] > 0:
				u[j] = U[ll[j]-1]
			else:
				u[j] = g[-ll[j]-1]
		M = np.ones((3,3))
		M[:,1:] = c
		gu = np.linalg.solve(M, u)
		m += quad_energy_error(c, kappa, ux, uy, gu)
	m = np.round(np.sqrt(m), 6)
	return m
def quad_energy_error(c, kappa, ux, uy, av):
	J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
				  [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
	A = abs(np.linalg.det(J))
	c1, c2 = 1/6, 2/3
	qpts = np.array([[c1, c1, c2],
					 [c1, c2, c1],
					 [c2, c1, c1]])
	qpts = qpts@c
	kvals = kappa(qpts[:, 0], qpts[:, 1])
	uxvals = ux(qpts[:, 0], qpts[:, 1])
	uyvals = uy(qpts[:, 0], qpts[:, 1])
	dx = uxvals-av[1]
	dy = uyvals-av[2]
	if type(kvals) == int:
		kvals = kvals*np.ones_like(uxvals).T
		I = (A/6)*kvals@(dx**2 + dy**2)
	else:
		I = (A/6)*kvals.T@(dx**2 + dy**2)
	return I
def energy_norm(mesh, kappa, ux, uy):
	m = 0
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		m += quad_energy(c, kappa, ux, uy)
	m = np.sqrt(m)
	return m
def quad_energy(c, kappa, ux, uy):
	x13 = c[0,0] - c[2,0]
	x23 = c[1,0] - c[2,0]
	y13 = c[0,1] - c[2,1]
	y23 = c[1,1] - c[2,1]
	J = abs(x13*y23 - y13*x23)
	c1, c2 = 2/3, 1/6
	T = np.array([[c1, c2, c2],
		          [c2, c1, c2],
		          [c2, c2, c1]])
	coords = T@c
	kvals = kappa(coords[:,0], coords[:,1])
	uxvals = ux(coords[:,0], coords[:,1])
	uyvals = uy(coords[:,0], coords[:,1])
	if type(kvals) == int:
		kvals = kvals*np.ones_like(uxvals).T
		I = (J/6)*(kvals@(uxvals**2 + uyvals**2))
	else:
		I = (J/6)*(kvals@(uxvals**2 + uyvals**2))
	return I
"""
MESH INITIALIZATION
"""
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
def course_circle_mesh_dirichlet():
	mesh = {}
	mesh['Degree'] = 1
	mesh['BndyFcn'] = circlef
	elements = [[1, 2, 3],
				[-3, 4, 5],
				[-5, 6, 7],
				[-7, 8, -1]]
	elements = np.array(elements)
	edges = [[1, 2],
			[2, 3],
			[3, 1],
			[3, 4],
			[4, 1],
			[4, 5],
			[5, 1],
			[5, 2]]
	edges = np.array(edges)
	edge_els = [[1, 4],
				[1, 0],
				[1, 2],
				[2, 0],
				[2, 3],
				[3, 0],
				[3, 4],
				[4, 0]]
	edge_els = np.array(edge_els)
	edge_cflags = [[0, 1, 0, 1, 0, 1, 0, 1]]
	edge_cflags = np.array(edge_cflags).T
	nodes = [[0,0],
			[1,0],
			[0,1],
			[-1,0],
			[0,-1]]
	nodes = np.array(nodes)
	node_ptrs = [[1, -1, -2, -3, -4]]
	node_ptrs =	np.array(node_ptrs).T
	mesh['FNodePtrs'] = np.array([[1]])
	c_node_ptrs = np.array([np.linspace(2,5,4, endpoint=True)]).T
	mesh['CNodePtrs'] = c_node_ptrs.astype(int)
	mesh['FBndyEdges'] = np.array(np.zeros((0, 1))).astype(int)
	mesh['Elements'] = elements.astype(int)
	mesh['Edges'] = edges.astype(int)
	mesh['EdgeEls'] = edge_els.astype(int)
	mesh['EdgeCFlags'] = edge_cflags.astype(int)
	mesh['NodePtrs'] = node_ptrs.astype(int)
	mesh['Nodes'] = nodes
	return mesh
def course_elliptic_mesh_dirichlet(a, b):
	mesh = {}
	mesh['Degree'] = 1
	mesh['BndyFcn'] = ellipsef
	elements = [[1, 2, 3],
				[-3, 4, 5],
				[-5, 6, 7],
				[-7, 8, -1]]
	elements = np.array(elements)
	edges = [[1, 2],
			[2, 3],
			[3, 1],
			[3, 4],
			[4, 1],
			[4, 5],
			[5, 1],
			[5, 2]]
	edges = np.array(edges)
	edge_els = [[1, 4],
				[1, 0],
				[1, 2],
				[2, 0],
				[2, 3],
				[3, 0],
				[3, 4],
				[4, 0]]
	edge_els = np.array(edge_els)
	edge_cflags = [[0, 1, 0, 1, 0, 1, 0, 1]]
	edge_cflags = np.array(edge_cflags).T
	nodes = [[0,0],
			[a,0],
			[0,b],
			[-a,0],
			[0,-b]]
	nodes = np.array(nodes)
	node_ptrs = [[1, -1, -2, -3, -4]]
	node_ptrs =	np.array(node_ptrs).T
	mesh['FNodePtrs'] = np.array([[1]])
	c_node_ptrs = np.array([np.linspace(2,5,4, endpoint=True)]).T
	mesh['CNodePtrs'] = c_node_ptrs.astype(int)
	mesh['FBndyEdges'] = np.array(np.zeros((0, 1))).astype(int)
	mesh['Elements'] = elements.astype(int)
	mesh['Edges'] = edges.astype(int)
	mesh['EdgeEls'] = edge_els.astype(int)
	mesh['EdgeCFlags'] = edge_cflags.astype(int)
	mesh['NodePtrs'] = node_ptrs.astype(int)
	mesh['Nodes'] = nodes
	mesh['a'] = a
	mesh['b'] = b
	return mesh
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
	if e > 0:
		n = np.array([[c[1,1]-c[0,1]],
			      [c[0,0]-c[1,0]]])
	else:
		n = -np.array([[c[1,1]-c[0,1]],
			      [c[0,0]-c[1,0]]])
	n /= np.linalg.norm(n)
	return n
def get_tri_node_indices(mesh):
	Nt = len(mesh['Elements'])
	ElList = np.zeros((Nt, 3), dtype=int)
	for k in range(1, Nt+1):
		coords, indices = get_nodes(mesh, k)
		ElList[k-1] = indices.T
	return ElList
"""
MESH REFINEMENT
"""
def refine(mesh_):
	if mesh_['Degree'] is not 1:
		print('This method only refines meshes for linear Lagrange triangles.')
		return mesh_
	Nt_ = mesh_['Elements'][:,0].shape[0]
	Ne_ = mesh_['Edges'][:,0].shape[0]
	Nv_ = mesh_['Nodes'][:,0].shape[0]
	Nf_ = mesh_['FNodePtrs'].shape[0]
	Nc_ = mesh_['CNodePtrs'].shape[0]
	Nb_ = mesh_['FBndyEdges'].shape[0]
	mesh = {}
	mesh['Degree'] = 1
	bndy_fcn = mesh_['BndyFcn']
	Nt = 4*Nt_
	elements = np.zeros((Nt, 3))
	Ne = 2*Ne_ + 3*Nt_
	edges = np.zeros((Ne, 2))
	edge_els = np.zeros((Ne, 2))
	edge_cflags = np.zeros((Ne, 1))
	edge_cflags[0:2*Ne_:2] = mesh_['EdgeCFlags']
	edge_cflags[1:2*Ne_+1:2] = mesh_['EdgeCFlags']
	nodes = [mesh_['Nodes'], np.zeros((len(mesh_['Nodes']) + Ne_, 2))]
	nodes[1][:len(nodes[0][:,0]),:] = nodes[0]
	length = mesh_['NodePtrs'].shape[0]
	node_ptrs = [mesh_['NodePtrs'], np.zeros((length + Ne_, 1))]
	node_ptrs[1][:length,:] = node_ptrs[0]
	Nv = Nv_
	Nc1 = sum(mesh_['EdgeEls'][:,1] == 0)
	length = mesh_['CNodePtrs'].shape[0]
	c_node_ptrs = [mesh_['CNodePtrs'],
				   np.zeros((length + Nc1, 1))]
	c_node_ptrs[1][0:length,:] = c_node_ptrs[0]
	Nc = Nc_
	length = mesh_['FNodePtrs'].shape[0]
	f_node_ptrs = [mesh_['FNodePtrs'], np.zeros((length + Nv_ + Ne_ - Nc_ - Nc1 - Nf_, 1))]
	f_node_ptrs[1][0:length,0] = f_node_ptrs[0].reshape(length,)
	Nf = Nf_
	Nb = 2*Nb_
	f_bndy_edges = np.zeros((Nb, 1))
	f_bndy_edges[0:Nb+1:2] = np.vstack([mesh_['FBndyEdges'], mesh_['FBndyEdges']]) - 1
	f_bndy_edges[1:Nb+2:2] = np.vstack([mesh_['FBndyEdges'], mesh_['FBndyEdges']])
	try:
		level_nodes = np.vstack([mesh_['LevelNodes'], Nv_ + Ne_])
		node_parents = [mesh_['NodeParents'], np.zeros((len(mesh_['NodeParents']) + Ne_, 2))]
		mesh['LevelNodes'] = level_nodes
		mesh['NodeParents'] = node_parents
	except:
		level_nodes = np.array([Nv_, Nv_ + Ne_])
		node_parents = [np.linspace(1,Nv_,Nv_, endpoint=True), np.zeros((len(np.linspace(1,Nv_,Nv_,endpoint=True)) + Ne_, 2))]
		mesh['LevelNodes'] = level_nodes
		mesh['NodeParents'] = node_parents
	for i in range(1, Ne_+1):
		v1, v2 = mesh_['Edges'][i-1,0], mesh_['Edges'][i-1,1]
		if mesh_['EdgeCFlags'][i-1,0] == 0:
			nodes[1][Nv,:] = 0.5*(nodes[0][v1-1,:] + nodes[0][v2-1,:])
		else:
			if bndy_fcn == ellipsef:
				pts = bndy_fcn(nodes[0][v1-1,:], nodes[0][v2-1,:], a=mesh_['a'], b=mesh_['b'])
			else:
				pts = bndy_fcn(nodes[0][v1-1,:], nodes[0][v2-1,:])
			nodes[1][Nv,:] = pts
		node_parents[1][Nv,:] = np.array([v1, v2])
		if mesh_['EdgeEls'][i-1, 1] == 0:
			c_node_ptrs[1][Nc,0] = Nv+1
			node_ptrs[1][Nv,0] = -(Nc+1)
			Nc += 1
		else:
			f_node_ptrs[1][Nf,0] = Nv+1
			node_ptrs[1][Nv,0] = Nf+1
			Nf += 1
		edges[2*i-2,:] = np.array([v1, Nv+1])
		edges[2*i-1,:]   = np.array([Nv+1, v2])
		Nv += 1
	for i in range(1, Nt_+1):
		eptr = mesh_['Elements'][i-1,:]
		e = abs(eptr)
		s = np.sign(eptr)
		newe = np.zeros((3,2))
		for j in range(3):
			if s[j] > 0:
				newe[j,:] = np.array([2*e[j]-1, 2*e[j]])
			else:
				newe[j,:] = -np.array([2*e[j], 2*e[j]-1])
		elements[4*i-4,:] = np.array([newe[0,0], 2*Ne_+3*i-2, newe[2,1]])
		elements[4*i-3,:] = np.array([newe[0,1], newe[1,0], 2*Ne_+3*i-1])
		elements[4*i-2,:] = np.array([2*Ne_+3*i, newe[1,1], newe[2,0]])
		elements[4*i-1,:]   = -np.array([2*Ne_+3*i-2, 2*Ne_+3*i-1, 2*Ne_+3*i])
		newe = abs(newe).astype(int)
		if edge_els[newe[0,0]-1,0] == 0:
			edge_els[newe[0,0]-1,0] = 4*i-3
			edge_els[newe[0,1]-1,0] = 4*i-2
		else:
			edge_els[newe[0,0]-1,1] = 4*i-3
			edge_els[newe[0,1]-1,1] = 4*i-2
		if edge_els[newe[1,0]-1,0] == 0:
			edge_els[newe[1,0]-1,0] = 4*i-2
			edge_els[newe[1,1]-1,0] = 4*i-1
		else:
			edge_els[newe[1,0]-1,1] = 4*i-2
			edge_els[newe[1,1]-1,1] = 4*i-1
		if edge_els[newe[2,0]-1,0] == 0:
			edge_els[newe[2,0]-1,0] = 4*i-1
			edge_els[newe[2,1]-1,0] = 4*i-3
		else:
			edge_els[newe[2,0]-1,1] = 4*i-1
			edge_els[newe[2,1]-1,1] = 4*i-3
		edges[2*Ne_+3*i-3,:]     = np.array([edges[2*(e[0])-2, 1], edges[2*(e[2])-2,1]])
		edges[2*Ne_+3*i-2,:]     = np.array([edges[2*(e[1])-2, 1], edges[2*(e[0])-2,1]])
		edges[2*Ne_+3*i-1,:]     = np.array([edges[2*(e[2])-2, 1], edges[2*(e[1])-2,1]])
		edge_els[2*Ne_+3*i-3,:]  = np.array([4*i-3,4*i])
		edge_els[2*Ne_+3*i-2,:]  = np.array([4*i-2,4*i])
		edge_els[2*Ne_+3*i-1,:]  = np.array([4*i-1,4*i])
	edge_els[(f_bndy_edges).astype(int)-1,1] = -np.array([np.linspace(1, Nb, Nb, endpoint=True)]).T
	mesh['BndyFcn'] = mesh_['BndyFcn']
	mesh['CNodePtrs'] = c_node_ptrs[1].astype(int)
	mesh['FNodePtrs'] = f_node_ptrs[1].astype(int)
	mesh['FBndyEdges'] = f_bndy_edges.astype(int)
	mesh['Elements'] = elements.astype(int)
	mesh['Edges'] = edges.astype(int)
	mesh['EdgeEls'] = edge_els.astype(int)
	mesh['EdgeCFlags'] = edge_cflags.astype(int)
	mesh['NodePtrs'] = node_ptrs[1].astype(int)
	mesh['Nodes'] = nodes[1]
	if mesh['BndyFcn'] == ellipsef:
		mesh['a'] = mesh_['a']
		mesh['b'] = mesh_['b']
	return mesh
"""
CREATE PDE
"""
def laplacian(mesh):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	# SYMMETRIC LAPLACIAN
	for k in range(1, len(mesh['Elements'])+1):
		coords, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, coords
		C = np.round(np.linalg.inv(M), 14)
		G = C[1:3,:].T@C[1:3,:]
		J = np.array([[coords[1,0]-coords[0,0], coords[2,0]-coords[0,0]],
					  [coords[1,1]-coords[0,1], coords[2,1]-coords[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		qpt = sum(coords)/3
		try:
			I = A*kappa(qpt[0], qpt[1])
		except:
			I = A
		for s in range(1,4):
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,s+1):
					llr = ll[r-1]
					if llr > 0:
						if llr <= lls:
							K[llr-1,lls-1] = K[llr-1, lls-1] + G[r-1,s-1]*I
						else:
							K[lls-1,llr-1] = K[lls-1, llr-1] + G[r-1,s-1]*I
	K = np.round(K + np.triu(K, 1).T, 14)
	# print("\nLaplacian:\n", K)
	return K
def contribute_u(mesh, K0, coeff=1):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					  [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		qpt = sum(c)/3
		for s in range(1,4):
			phi_i = C[0, s-1] + C[1, s-1]*qpt[0] + C[2, s-1]*qpt[1]
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,s+1):
					phi_j = C[0, r-1] + C[1, r-1]*qpt[0] + C[2, r-1]*qpt[1]
					llr = ll[r-1]
					if llr > 0:
						if llr <= lls:
							K[llr-1,lls-1] = K[llr-1, lls-1] + phi_i*phi_j*A
						else:
							K[lls-1,llr-1] = K[lls-1, llr-1] + phi_i*phi_j*A
	K = np.round(K + np.triu(K, 1).T, 14)
	return K0 + coeff*K
def contribute_u3(mesh, K0, coeff=1):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					  [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		qpt = sum(c)/3
		for s in range(1,4):
			phi_i = C[0, s-1] + C[1, s-1]*qpt[0] + C[2, s-1]*qpt[1]
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,4):
					phi_j = C[0, r-1] + C[1, r-1]*qpt[0] + C[2, r-1]*qpt[1]
					phi_j = phi_j**3
					llr = ll[r-1]
					if llr > 0:
						K[llr-1,lls-1] = K[llr-1, lls-1] + phi_i*phi_j*A
	K = np.round(K + np.triu(K, 1).T, 14)
	return K0 + coeff*K
def contribute_dux(mesh, K0, coeff=1):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	# NONSYMMETRIC PORTION
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
		Gx = C[1,:]
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
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,4):
					phi_j = C[0, r-1] + C[1, r-1]*qpt[0] + C[2, r-1]*qpt[1]
					llr = ll[r-1]
					if llr > 0:
						K[llr-1,lls-1] = K[llr-1, lls-1] + Gx[s-1]*phi_j*A
	return K0 + coeff*K
def contribute_duy(mesh, K0, coeff=1):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	# NONSYMMETRIC PORTION
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
		Gy = C[2,:]
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
			lls = ll[s-1]
			if lls > 0:
				for r in range(1,4):
					phi_j = C[0, r-1] + C[1, r-1]*qpt[0] + C[2, r-1]*qpt[1]
					llr = ll[r-1]
					if llr > 0:
						K[llr-1,lls-1] = K[llr-1, lls-1] + Gy[s-1]*phi_j*A
	return K0 + coeff*K
def load(mesh, f, k, g, h, t=0):
	Nf = mesh['FNodePtrs'].shape[0]
	F  = np.zeros((Nf, 1))
	vo = np.ones((3,))
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		qpt = (1/3)*sum(c)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					 [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		fval = f(qpt[0], qpt[1], t)
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
				M[:,0], M[:,1:] = vo, c[:,0:]
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
				element = mesh['Elements'][k-1, :]
				edge = abs(mesh['Elements'][k-1, j])
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
def loadu(mesh, U0):
	Nf = mesh['FNodePtrs'].shape[0]
	F  = np.zeros((Nf, 1))
	vo = np.ones((3,))
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		qpt = (1/3)*sum(c)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					 [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		fval = 1
		I = (A*fval)/3
		for r in range(1, 4):
			llr = ll[r-1]
			if llr > 0:
				F[llr-1] = F[llr-1] + I*U[llr-1]
		
	return F
"""
BOUNDARY CONDITIONS
"""
def get_dirichlet_data(mesh, t=0):
	try:
		CNodePtrs = mesh['CNodePtrs'][:,0]
		x = mesh['Nodes'][CNodePtrs-1,0]
		x.reshape(len(x),)
		y = mesh['Nodes'][CNodePtrs-1,1]
		y.reshape(len(y),)
		answer = u(x, y, t)
		return answer
	except:
		return np.zeros_like((mesh['CNodePtrs'].shape[0],))
def get_neumann_data(mesh, t=0):
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
			g1x, g1y = ux(z1[0], z1[1], t=t), uy(z1[0], z1[1], t=t)
			g2x, g2y = ux(z2[0], z2[1], t=t), uy(z2[0], z2[1], t=t)
		except:
			g1x, g1y = 0, 0
			g2x, g2y = 0, 0
		g1, g2 = [g1x, g1y], [g2x, g2y]
		g = np.array([g1, g2])
		hx = k1*np.dot(g1,nv)
		hy = k2*np.dot(g2,nv)
		h[j-1,:] = [hx, hy]
	return h
def generate_dirichlet(mesh, t=0):
	c_node_ptrs = mesh['CNodePtrs']
	X, Y = mesh['Nodes'][c_node_ptrs-1,0], mesh['Nodes'][c_node_ptrs-1,1]
	# True dirichlet
	g = np.zeros_like(X)
	# g = u(X, Y, t)
	return g
def generate_neumann(mesh, t=0):
	Nb = mesh['FBndyEdges'].shape[0]
	# h = np.zeros((Nb, 2))
	h = np.ones_like((Nb, 2))
	return h
"""
MESH INITIALIZATION AND REFINEMENT for CIRCLE or ELLIPSE
"""
def generate(N, type_eq='Circle', a=2, b=3):
	for n in range(N+1):
		if n == 0:
			if type_eq.lower() == 'circle':
				try:
					mesh = load_obj('circle0')
				except:
					mesh = course_circle_mesh_dirichlet()
					save_obj(mesh, 'circle0')
				plot_circle_mesh(mesh, 0)
			elif type_eq.lower() == 'ellipse':
				mesh = course_elliptic_mesh_dirichlet(a, b)
				plot_elliptic_mesh(mesh, 0)
			else:
				print("\n\nIncorrect geometry.\n\n")
				return 0
			print(f"\nInitializing mesh.\nNumber of Elements: {mesh['Elements'].shape[0]}")
		elif n == N:
			print("Done generating mesh.\n")
			pass
		else:
			print(f"\nRefining mesh.\nNumber of refinements: {n}")
			if type_eq.lower() == 'circle':
				try:
					mesh = load_obj(f'circle{n}')
				except:
					mesh = refine(mesh)
					save_obj(mesh, f'circle{n}')
			else:
				mesh = refine(mesh)
			if type_eq.lower() == 'circle':
				plot_circle_mesh(mesh, n)
			elif type_eq.lower() == 'ellipse':
				plot_elliptic_mesh(mesh, n)
			print(f"Number of Elements: {mesh['Elements'].shape[0]}")
	return mesh
"""
CREATE MULTIPLE PLOTS with REFINEMENT and DEBUG
"""
def compute(N, type_eq='Circle', cycle=False, error=False, debug=False, gif=False):
	errors = {'Energy': {}, 'L2': {}}
	if debug == True:
		cycle = False
	if cycle == False:
		mesh = generate(N, type_eq=type_eq, a=1, b=2)
		print(f"Final Number of Elements: {mesh['Elements'].shape[0]}")
		print("\n\tComputing stiffness matrix.")
		K = laplacian(mesh)
		K = contribute_u(mesh, K)
		# K = contribute_dux(mesh, K)
		# K = contribute_duy(mesh, K)
		# g = get_dirichlet_data(mesh)
		g = generate_dirichlet(mesh)
		# h = get_neumann_data(mesh)
		h = generate_neumann(mesh)
		print("\tComputing load vector.")
		F = load(mesh, f, kappa, g, h, t=0)	
		print("\n\tSolving the system.")
		U = scipy.sparse.linalg.cg(K, F)[0].reshape(F.shape[0], 1)
		print("\n\tConstructing solution.")
		X, Y = mesh['Nodes'][:,0], mesh['Nodes'][:,1]
		Z = np.zeros((len(X), 1))
		FNodePtrs = mesh['FNodePtrs'][:,0]
		Z[FNodePtrs - 1] = U
		CNodePtrs = mesh['CNodePtrs'][:,0]
		Z[CNodePtrs - 1] = g.reshape(len(g), 1)
		Z = Z.T[0]
		Z = np.round(Z, 14)
		if debug == False:
			print("\n\tPlotting the solution.")
			plot_matrix_density(mesh, N, K)
			plot_init_solution(mesh, U, g, N)
		elif debug == True:
			print("\n***********\nLaplacian:\n", laplacian(mesh))
			print("u:\n", contribute_u(mesh, np.zeros_like(K)))
			print("K:\n", K)
			print("U:\n", U)
			print("F:\n", F, "\n***********")
			print("Z:\n", Z)
	else:
		for n in range(1, N+1):
			mesh = generate(n, type_eq=type_eq, a=1, b=2)
			print(f"\nFinal Number of Elements: {mesh['Elements'].shape[0]}")
			print("\n\tComputing stiffness matrix.")
			K = laplacian(mesh)
			K = contribute_u(mesh, K)
			# K = contribute_dux(mesh, K)
			# K = contribute_duy(mesh, K)
			# plot_matrix_density(mesh, n, K)
			# g = get_dirichlet_data(mesh)
			g = generate_dirichlet(mesh)
			# h = get_neumann_data(mesh)
			h = generate_neumann(mesh)
			print("\tComputing load vector.")
			F = load(mesh, f, kappa, g, h, t=0)	
			print("\n\tSolving the system.")
			U = scipy.sparse.linalg.cg(K, F)[0].reshape(F.shape[0], 1)
			if debug == False and error == True:
				print("\tCalculating Energy norm error.")
				err = energy_norm_error(mesh, kappa, ux, uy, U, g)
				nrm = energy_norm(mesh, kappa, ux, uy)
				errors['Energy'][f"{int(mesh['Elements'].shape[0])}"] = np.round(err/nrm, 6)
				X, Y = mesh['Nodes'][:,0], mesh['Nodes'][:,1]
				Z = np.zeros((len(X), 1))
				FNodePtrs = mesh['FNodePtrs'][:,0]
				Z[FNodePtrs - 1] = U
				CNodePtrs = mesh['CNodePtrs'][:,0]
				Z[CNodePtrs - 1] = g.reshape(len(g), 1)
				Z = Z.T[0]
				Z = np.round(Z, 14)
				print("\tCalculating Relative L2 error.")
				el = mesh['Elements'].shape[0]
				L2 = np.linalg.norm(Z-u(X,Y), ord=2)/np.linalg.norm(u(X,Y), ord=2)
				errors['L2'][el] = L2
			print("\n\tPlotting the solution.")
			plot_init_solution(mesh, U, g, n)
		if debug == False and error == True:
			"""
				L2 ERROR
			"""
			df = pd.DataFrame.from_dict(errors['L2'], orient='index').astype(float)
			df.columns = ['L2']
			fig = plt.figure(1)
			ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
			df.plot(loglog=True, figsize=(10,6), title=r'Relative $L_2$ Error', ax=ax)
			plt.xlabel('$\\log(\\#$ Elements$)$')
			plt.ylabel('$\\log($Error$)$')
			plt.xlim(df.index[0], df.index[-1])
			plt.savefig(f'Relative_L2_Error.png')
			plt.show()
			plt.close()
			"""
				REL ENERGY NORM
				HW3P1 MATLAB VALUES
			# """
			matlab = [0.89646, 0.44277, 0.25889, 0.13817, 0.072165, 0.039454, 0.024981]
			# matlab = [1.2984, 0.75222, 0.41782, 0.2238, 0.11599, 0.059006, 0.029739]
			df = pd.DataFrame.from_dict(errors['Energy'], orient='index').astype(float)
			df.columns = ['Energy']
			def power_law(x, a, r):
			    return a*x**r
			df.index = [4**i for i in range(1, N+1)]
			xdata = [4**i for i in range(1, N+1)]
			ydata = df['Energy']
			popt1, pcov = curve_fit(power_law, xdata, ydata)
			df['FitEnergy'] = power_law(xdata, *popt1)
			plt.figure(2, figsize=(10, 6))
			if N <= 7:
				df['Matlab'] = matlab[:N]
				plt.loglog(xdata, df['Matlab'], 'ko', label='Matlab Norm Data')
				ydata = df['Matlab']
				popt2, pcov = curve_fit(power_law, xdata, ydata)
				df['FitMatlab'] = power_law(xdata, *popt2)
				plt.loglog(xdata, df['FitMatlab'], 'k--', label='Matlab Fit')
				plt.title(f'Relative Energy Norm Error\n' +
						  f'Matlab $y_M=bx^k$,  b={np.round(popt2[0], 2)}, k={np.round(popt2[1], 2)}\n' +
						  f'Approx. $y_A=ax^r$,  a={np.round(popt1[0], 2)}, r={np.round(popt1[1], 2)}')
			else:
				plt.title(f'Relative Energy Norm Error\n $y=ax^r$,  a={np.round(popt1[0], 2)}, r={np.round(popt1[1], 2)}')
			plt.loglog(xdata, ydata, 'ro', mfc='none', label='Approx. Norm Data')
			plt.loglog(xdata, df['FitEnergy'], 'r.-.', mfc='none', label='Approx. Norm Fit')
			plt.xlabel('$\\log(\\#$ of Elements$)$')
			plt.ylabel('$\\log($Energy Norm$)$')
			plt.legend()
			plt.xlim(xdata[0], xdata[-1])
			plt.savefig('Relative_Energy_Norm_Error.png')
			plt.show()
	if debug == False and gif == True:
		"""
		SURFACE REVOLUTION GIF CREATOR
		"""
		print("\nCreating GIF.")
		gif_creator(mesh, U, g)
	return mesh, U
"""
REFINE MESH DATA STRUCTURE
"""
N = 6
mesh, U = compute(N, type_eq='Circle', cycle=False, error=False, debug=False, gif=False)

"""
TIME EVOLUTION
"""
def uv(mesh):
	K = laplacian(mesh)
	K = contribute_u(mesh, 0*K)
	return K
def mass(mesh, dt):
	m = dt*laplacian(mesh)
	m = contribute_u(mesh, m, 1+dt)
	# m = contribute_u(mesh, m, 1)
	# m = contribute_u3(mesh, m, dt)
	return m
print("Initial condition has been created.")
print("Generating time evolution images.")
dt = 0.01
m = mass(mesh, dt)
for n in tqdm(range(1, 1000)):
	t = np.round(n*dt, 2)
	# print(f"t = {t},\tComputing load vector.")
	g = generate_dirichlet(mesh, t=t+dt)
	h = generate_neumann(mesh, t=t+dt)
	F = loadu(mesh, U) + dt*load(mesh, f, kappa, g, h, t=t+dt)
	# print("\nSolving the system.")
	U = scipy.sparse.linalg.cg(m, F)[0].reshape(F.shape[0], 1)
	# print("Plotting solutions.")
	plot_init_solution(mesh, U, g, N, t=t)


