"""Headless CUDA rasterization of PyBullet URDF transforms."""
import numpy as np, torch, trimesh, pybullet as p
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation

class CudaScene:
    def __init__(self,rig):
        self.rig=rig;self.ctx=dr.RasterizeCudaContext(device='cuda');self.records=[];self.bounds=[]
        verts=[];normals=[];triangles=[];record_ids=[];isrobot=[];offset=0
        for bi in range(p.getNumBodies(physicsClientId=rig.cid)):
            body=p.getBodyUniqueId(bi,physicsClientId=rig.cid)
            for shape in p.getVisualShapeData(body,physicsClientId=rig.cid):
                _,link,kind,dims,path,origin,quat,*_=shape
                try:
                    if kind==p.GEOM_MESH:
                        path=path.decode();
                        # Most URDF DAE visuals have a matching STL alongside.
                        if path.lower().endswith('.dae'):
                            from pathlib import Path
                            alternatives=[Path(path).with_suffix('.STL'),Path(path).with_suffix('.stl')]
                            path=str(next((x for x in alternatives if x.exists()),path))
                        m=trimesh.load(path,force='mesh',process=True)
                        m.apply_scale(dims)
                    elif kind==p.GEOM_BOX:m=trimesh.creation.box(extents=dims)
                    elif kind==p.GEOM_SPHERE:m=trimesh.creation.icosphere(subdivisions=2,radius=dims[0])
                    elif kind==p.GEOM_CYLINDER:m=trimesh.creation.cylinder(radius=dims[1],height=dims[0],sections=24)
                    elif kind==p.GEOM_CAPSULE:m=trimesh.creation.capsule(radius=dims[1],height=dims[0],count=[12,16])
                    else:continue
                except Exception as e:
                    raise RuntimeError(f'Could not load URDF visual {path}: {e}') from e
                mat=np.eye(4);mat[:3,:3]=Rotation.from_quat(quat).as_matrix();mat[:3,3]=origin
                m.apply_transform(mat)
                v=np.asarray(m.vertices,dtype=np.float32);f=np.asarray(m.faces,dtype=np.int32);nn=np.asarray(m.vertex_normals,dtype=np.float32)
                rid=len(self.records);self.records.append((body,link))
                self.bounds.append(trimesh.bounds.corners(m.bounds))
                verts.append(np.c_[v,np.ones(len(v),np.float32)]);normals.append(nn)
                record_ids.append(np.full(len(v),rid,dtype=np.int64));isrobot.append(np.full((len(v),1),body in rig.bodies,dtype=np.float32));triangles.append(f+offset);offset+=len(v)
        self.v=torch.tensor(np.concatenate(verts),device='cuda');self.n=torch.tensor(np.concatenate(normals),device='cuda')
        self.f=torch.tensor(np.concatenate(triangles),device='cuda',dtype=torch.int32)
        self.rids=torch.tensor(np.concatenate(record_ids),device='cuda');self.robot=torch.tensor(np.concatenate(isrobot),device='cuda')
        view=np.array(rig.view).reshape(4,4,order='F');proj=np.array(rig.proj).reshape(4,4,order='F')
        self.mvp=torch.tensor(proj@view,dtype=torch.float32,device='cuda')
        eye=np.linalg.inv(view)[:3,3];self.eye=torch.tensor(eye,dtype=torch.float32,device='cuda')


    def fit_camera(self,qs,qa):
        rig=self.rig;points=[]
        ids=set(np.linspace(0,len(qs)-1,min(20,len(qs))).astype(int).tolist())
        for q in [qs,qa]:
            flat=np.nan_to_num(q.reshape(len(q),-1))
            ids.update(np.argmin(flat,axis=0).tolist());ids.update(np.argmax(flat,axis=0).tolist())
        for q in [qs,qa]:
            for t in sorted(ids):
                rig.set_q(q[t])
                for (body,link),bounds in zip(self.records,self.bounds):
                    if body not in rig.bodies:continue
                    if link<0:pos,quat=p.getBasePositionAndOrientation(body,physicsClientId=rig.cid)
                    else:
                        state=p.getLinkState(body,link,computeForwardKinematics=True,physicsClientId=rig.cid);pos,quat=state[4],state[5]
                    points.extend(bounds@Rotation.from_quat(quat).as_matrix().T+pos)
        points=np.asarray(points);lo=points.min(0);hi=points.max(0);mid=(lo+hi)/2
        for bi in range(p.getNumBodies(physicsClientId=rig.cid)):
            body=p.getBodyUniqueId(bi,physicsClientId=rig.cid)
            if body not in rig.bodies:
                pos,quat=p.getBasePositionAndOrientation(body,physicsClientId=rig.cid)
                p.resetBasePositionAndOrientation(body,[pos[0],pos[1],lo[2]-.06],quat,physicsClientId=rig.cid)
        view=np.array(p.computeViewMatrixFromYawPitchRoll(mid.tolist(),3,45,-20,0,2)).reshape(4,4,order='F')
        camera=(np.c_[points,np.ones(len(points))]@view.T)[:,:3];camera[:,2]+=3
        ty=np.tan(np.deg2rad(21));tx=ty*rig.w/rig.h
        distance=max(1.25,float(np.max(np.maximum(np.abs(camera[:,0])/tx,np.abs(camera[:,1])/ty)*1.1+camera[:,2])))
        rig.target=mid;rig.view=p.computeViewMatrixFromYawPitchRoll(mid.tolist(),distance,45,-20,0,2)
        view=np.array(rig.view).reshape(4,4,order='F');proj=np.array(rig.proj).reshape(4,4,order='F')
        self.mvp=torch.tensor(proj@view,dtype=torch.float32,device='cuda')
        self.eye=torch.tensor(np.linalg.inv(view)[:3,3],dtype=torch.float32,device='cuda')

    @torch.no_grad()
    def render(self,command=False):
        rig=self.rig;trans=[]
        cache={}
        for body,link in self.records:
            key=(body,link)
            if key not in cache:
                if link<0:pos,quat=p.getBasePositionAndOrientation(body,physicsClientId=rig.cid)
                else:
                    state=p.getLinkState(body,link,computeForwardKinematics=True,physicsClientId=rig.cid);pos,quat=state[4],state[5]
                m=np.eye(4,dtype=np.float32);m[:3,:3]=Rotation.from_quat(quat).as_matrix();m[:3,3]=pos;cache[key]=m
            trans.append(cache[key])
        matrices=torch.tensor(np.stack(trans),device='cuda')
        mt=matrices[self.rids]
        world=(mt@self.v.unsqueeze(-1)).squeeze(-1)
        normal=(mt[:,:3,:3]@self.n.unsqueeze(-1)).squeeze(-1)
        clip=(world@self.mvp.T)[None]
        rast,_=dr.rasterize(self.ctx,clip,self.f,resolution=[rig.h,rig.w])
        attr=torch.cat([world[:,:3],normal,self.robot],dim=1)[None]
        pix,_=dr.interpolate(attr,rast,self.f)
        pos=pix[...,:3];n=torch.nn.functional.normalize(pix[...,3:6],dim=-1);robot=pix[...,6:7]
        l=torch.tensor([-.4,-.4,1.0],device='cuda');l=torch.nn.functional.normalize(l,dim=0)
        view=torch.nn.functional.normalize(self.eye-pos,dim=-1);half=torch.nn.functional.normalize(view+l,dim=-1)
        diff=(n*l).sum(-1,keepdim=True).clamp(0,1);spec=(n*half).sum(-1,keepdim=True).clamp(0,1).pow(32)
        tint=[.83,.37,.22] if command else [.15,.67,.65]
        tint=torch.tensor(tint,device='cuda')
        base=tint*(.38+.62*diff)+spec*.35
        rim=(1-(n*view).sum(-1,keepdim=True).abs()).pow(3)
        base+=rim*torch.tensor([.07,.1,.13],device='cuda')
        # Ground grid is shaded procedurally, avoiding aliasing from tiny boxes.
        floor=torch.tensor([.06,.087,.125],device='cuda').expand_as(base).clone()
        radial=((pos[...,:2]-torch.tensor(rig.target[:2],dtype=torch.float32,device='cuda'))**2).sum(-1,keepdim=True)
        floor*=.85+.15*torch.exp(-radial)
        grid=(torch.remainder(pos[...,0:1]+.003,.25)<.006)|(torch.remainder(pos[...,1:2]+.003,.25)<.006)
        floor+=grid*.014
        color=torch.where(robot>.5,base,floor)
        bg=torch.tensor([13/255,20/255,30/255],device='cuda')
        color=torch.where(rast[...,3:4]>0,color,bg)
        color=dr.antialias(color, rast, clip,self.f)
        im=(color[0].clamp(0,1)*255).byte().flip(0).cpu().numpy()
        return im
