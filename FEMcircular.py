import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from PIL import Image
import os, sys


def concatenate_pics(plist, N):
	images = [Image.open(i) for i in plist]
	widths, heights = zip(*(i.size for i in images))
	total_width = sum(widths)
	max_height = max(heights)
	new_im = Image.new('RGB', (total_width, max_height))
	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset,0))
		x_offset += im.size[0]
	new_im.save(f'merged{N}.jpg')
	for _ in plist:
		os.remove(_)

def gif_creator(plist):
	plist[0].save(f"Solution GIF.gif",
		save_all=True, append_images=plist[1:],
		optimize=True, duration=50, loop=0)
def f(x, y, t=0):
	r2 = x**2 + y**2
	"""
		e(-2*pi*r^2)*cos(2*pi*r^2)
	"""
	# answer = -8*np.pi*np.exp(-2*np.pi*r2)*((4*np.pi*r2 - 1)*np.sin(2*np.pi*r2) - np.cos(2*np.pi*r2))
	# sech^2(r^2)
	# def sech(x):
	# 	return 1/np.cosh(x)
	# answer = -4*sech(r2)**4*(-4*r2 + 2*r2*np.cosh(2*r2) - np.sinh(2*r2))
	# x^2-y^2/(r2 + 1)
	answer = -(4*(3*x**2 + x**4 - y**2*(3 + y**2)))/(1 + r2)**3
	# Modified e(-r^2)*cos(r^2)
	# answer = 4*np.pi*np.exp(-np.pi*r2)*((15*np.pi*r2 + 1)*np.cos(4*np.pi*r2) - 4*(2*np.pi*r2 - 1)*np.sin(4*np.pi*r2))
	# answer = np.sin(np.pi*r2-2*np.pi*5*t)
	return  answer

def kappa(x, y):
	return 1
def u(x,y):
	r2 = x**2 + y**2
	# return np.exp(-2*np.pi*r2)*np.cos(2*np.pi*r2)
	# return (1/np.cosh(r2))**2
	return (x**2-y**2)/(r2 + 1)
	# return np.exp(-np.pi*r2)*np.cos(4*np.pi*r2)
	# return np.zeros_like(x)
def ux(x,y):
	# return -2*2*np.pi*x*np.cos(x**2 + y**2)
	return 0
def uy(x,y):
	# return -2*2*np.pi*y*np.cos(x**2 + y**2)
	return 0

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
		n *= +1
	else:
		n *= -1
	n /= np.linalg.norm(n)
	return n
def get_tri_node_indices(mesh):
	Nt = len(mesh['Elements'])
	ElList = np.zeros((Nt, 3), dtype=int)
	for k in range(1, Nt+1):
		coords, indices = get_nodes(mesh, k)
		ElList[k-1] = indices.T
	return ElList

def plot_circle_mesh(mesh, N):
	plt.close()
	X, Y= mesh['Nodes'][:,0], mesh['Nodes'][:,1]
	Z = np.zeros((len(X), 1))
	triangles = get_tri_node_indices(mesh)
	trimesh = mtri.Triangulation(X,Y,triangles=triangles)
	plt.figure(figsize=(6,6))
	theta = np.linspace(0, 2*np.pi, 1000, endpoint=True)
	x, y = np.cos(theta), np.sin(theta)
	plt.triplot(trimesh, marker='o', markersize=1, color='k', linewidth=1) #
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.plot(x,y,'k')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.tight_layout()
	plt.axis('equal')
	plt.grid(alpha=0.618)
	plt.savefig(f'mesh{N}.png')
	# plt.show()

def plot_init_solution(mesh, U, g, N):
	plt.close()
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
	# plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title(f"Approximate: N={N}")
	ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
	plt.savefig(f'approx{N}.png')
	plt.close()
	sol = u(X,Y)
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	# plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title(f"Exact: N={N}")
	ax.plot_trisurf(X, Y, sol, cmap=cm.Spectral_r, linewidth=0)
	plt.savefig(f'sol{N}.png')
	plt.close()
	fig, ax = plt.subplots(subplot_kw={"projection":"3d"}, figsize=(10,6))
	# plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title(f"Difference: N={N}")
	ax.plot_trisurf(trimesh, Z-sol, cmap=cm.jet, linewidth=0)
	plt.savefig(f'diff{N}.png')
	plt.close()
	# plt.show()
	plist = [f'approx{N}.png', f'sol{N}.png']
	concatenate_pics(plist, N)
	if N == 7:
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
		for ii in range(46):
			ax.view_init(elev=ii, azim=0)
			plt.savefig(f'gifs/approx_elevu{str(ii).zfill(2)}.png')
			image = Image.open(f'gifs/approx_elevu{str(ii).zfill(2)}.png')
			plist.append(image) 
		for ii in range(361):
			ax.view_init(elev=45, azim=ii)
			plt.savefig(f'gifs/approx_rot{str(ii).zfill(3)}.png')
			image = Image.open(f'gifs/approx_rot{str(ii).zfill(3)}.png')
			plist.append(image)
		degs = list(range(46))
		degs.reverse()
		for ii in degs:
			ax.view_init(elev=ii, azim=0)
			plt.savefig(f'gifs/approx_elevd{str(ii).zfill(2)}.png')
			image = Image.open(f'gifs/approx_elevd{str(ii).zfill(2)}.png')
			plist.append(image)
		plt.close()
		gif_creator(plist)

def plot_time_solution(mesh, U, g, t):
	plt.close()
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
	# plt.triplot(trimesh, marker='*', markersize=1, linewidth=1)
	plt.xlim(X.min(), X.max())
	plt.ylim(Y.min(), Y.max())
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title(f"Approximate: t={t}")
	ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
	plt.savefig(f'approx_t{str(int(100*t)).zfill(3)}.png')
	plt.close()

def rel_energy_norm(mesh, kappa, ux, uy, U, g):
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		u = np.zeros((3,1))
		for j in range(3):
			if ll[j]>0:
				u[j] = U[ll[j]-1]
			else:
				u[j] = g[-ll[j]-1]
		M = np.ones((3,3))
		# M[:,0] =

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
	# print("Nt_:", Nt_, ", Ne_:", Ne_, ", Nv_:", Nv_, ", Nf_:", Nf_, ", Nc_:", Nc_, ", Nb_:", Nb_)
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
	node_ptrs[1][0:length,:] = node_ptrs[0]
	Nv = Nv_
	Nc1 = sum(mesh['EdgeEls'][:,1] == 0)
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
	edge_els[(f_bndy_edges).astype(int),1] = -np.array([np.linspace(1, Nb, Nb+1, endpoint=True)]).T
	mesh['CNodePtrs'] = c_node_ptrs[1].astype(int)
	mesh['FNodePtrs'] = f_node_ptrs[1].astype(int)
	mesh['FBndyEdges'] = f_bndy_edges.astype(int)
	mesh['Elements'] = elements.astype(int)
	mesh['Edges'] = edges.astype(int)
	mesh['EdgeEls'] = edge_els.astype(int)
	mesh['EdgeCFlags'] = edge_cflags.astype(int)
	mesh['NodePtrs'] = node_ptrs[1].astype(int)
	mesh['Nodes'] = nodes[1]
	return mesh
def stiffness(mesh, N):
	Nf = mesh['FNodePtrs'].shape[0]
	K = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	# SYMMETRIC LAPLACIAN
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
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
	
	# MATRIX DENSITY
	# if N > 1:
	# 	plt.close()
	# 	plt.figure(figsize=(10,6))
	# 	plt.spy(K)
	# 	plt.title(f'Stiffness Matrix: N={N}')
	# 	plt.savefig(f'stiffness{N}.png')
	# 	plt.close()
	# 	plt.show()
	return K
def get_dirichlet_data(mesh):
	try:
		CNodePtrs = mesh['CNodePtrs'][:,0]
		x = mesh['Nodes'][CNodePtrs-1,0]
		x.reshape(len(x),)
		y = mesh['Nodes'][CNodePtrs-1,1]
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
def generate_dirichlet(mesh):
	c_node_ptrs = mesh['CNodePtrs']
	X, Y = mesh['Nodes'][c_node_ptrs-1,0], mesh['Nodes'][c_node_ptrs-1,1]
	# Polynomial
	g = u(X,Y)
	# Modiefied ecos
	# g = u(1,1)*np.ones_like(c_node_ptrs)
	# True dirichlet
	# g = np.zeros_like(X)
	return g
def generate_neumann(mesh):
	Nb = mesh['FBndyEdges'].shape[0]
	h = np.zeros((Nb, 2))
	return h
def load(mesh, f, k, g, h, t=0):
	Nf = mesh['FNodePtrs'].shape[0]
	F  = np.zeros((Nf, 1))
	vo = np.ones((3,))
	Nt = mesh['Elements'].shape[0]
	for k in range(1, Nt+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		qpt   = (1/3)*sum(c)
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					 [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		if t == 0:
			fval = f(qpt[0], qpt[1])
		else:
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


N = 4
mesh = course_circle_mesh_dirichlet()
for _ in range(N+1):
	# print("N =",_)
	# plot_circle_mesh(mesh, _+1)
	K = stiffness(mesh, _+1)
	g = get_dirichlet_data(mesh)
	# g = generate_dirichlet(mesh)
	h = get_neumann_data(mesh)
	# h = generate_neumann(mesh)
	F = load(mesh, f, kappa, g, h)
	U = np.linalg.lstsq(K,F, rcond=0)[0]
	# plot_init_solution(mesh, U, g, _+1)
	if _ != N:
		mesh = refine(mesh)

"""
TIME EVOLUTION
"""
def mass(mesh):
	Nf = mesh['FNodePtrs'].shape[0]
	mass = np.zeros((Nf, Nf))
	vo = np.ones((3,))
	# SYMMETRIC
	for k in range(1, len(mesh['Elements'])+1):
		c, indices = get_nodes(mesh, k)
		ll = mesh['NodePtrs'][indices].T[0]
		M = np.zeros((3,3), dtype=float)
		M[:,0], M[:,1:] = vo, c
		C = np.round(np.linalg.inv(M), 14)
		G = C[1:3,:].T@C[1:3,:]
		J = np.array([[c[1,0]-c[0,0], c[2,0]-c[0,0]],
					  [c[1,1]-c[0,1], c[2,1]-c[0,1]]])
		A = 0.5*abs(np.linalg.det(J))
		for j in range(1, 4):
			edge = abs(mesh['Elements'][k-1,j-1])
			# print(mesh['EdgeEls'])
			eptr = -mesh['EdgeEls'][edge-1, 1]
			# print(eptr)
			if eptr < 0:
				ii = mesh['Edges'][edge-1,:]
				c = mesh['Nodes'][ii-1,:]
				ll = mesh['NodePtrs'][ii-1]
				# print(ll)
				midpoint = ((c[0,:] + c[1,:])/2).T
				for s in range(1,3):
					phi_i = sum([C[0, s-1]] + [C[i+1,s-1]*midpoint[i] for i in range(2)])
					lls = ll[s-1]
					if lls > 0:
						for r in range(1,s+1):
							phi_j = sum([C[0, r-1]] + [C[i+1,r-1]*midpoint[i] for i in range(2)])
							llr = ll[r-1]
							if llr > 0:
								mass[llr-1,lls-1] = mass[llr-1, lls-1] + phi_i*phi_j*A
	return mass


t, dt = 0, 0.01
plot_time_solution(mesh, U, g, t)
for t in np.round(np.linspace(t, 1, 101, endpoint=True), 14):
	M = np.linalg.pinv(mass(mesh))
	# g = get_dirichlet_data(mesh)
	F = load(mesh, f, kappa, g, h, t)
	U += dt*M@(K@U+F)
	if abs(U.max()) > 1E3:
		break 
	if int(100*t)% 1000 and t!= 0:
		plot_time_solution(mesh, U, g, t)

