"""URDF mesh replay in an isolated PyBullet client; optional CUDA rasterization."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


def resolve_mesh(name, source, packages):
    if name.startswith('package://'):
        package, tail = name[10:].split('/', 1)
        candidates = [Path(packages[package]) / tail] if package in packages else []
        candidates += [parent / package / tail for parent in source.parents]
        candidates += [parent / tail for parent in source.parents]
    elif name.startswith('file://'):
        candidates = [Path(name[7:])]
    else:
        candidates = [source.parent / name]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f'URDF mesh {name!r}; provide --package-root PACKAGE=PATH with its mesh directory')


def prepare_urdf(source, target, joint_names, packages, geometry):
    tree = ET.parse(source)
    root = tree.getroot()
    joints = root.findall('joint')
    by_name = {j.get('name'): j for j in joints}
    unknown = set(joint_names) - set(by_name)
    if unknown:
        raise ValueError(f'Joint names absent from URDF: {sorted(unknown)}')
    by_child = {j.find('child').get('link'): j for j in joints}
    visible = set()
    for name in joint_names:
        child = by_name[name].find('child').get('link')
        visible.add(child)
        while child in by_child:
            child = by_child[child].find('parent').get('link')
            visible.add(child)
    for _ in joints:
        for joint in joints:
            if joint.get('type') in ('fixed', 'prismatic') and joint.find('parent').get('link') in visible:
                visible.add(joint.find('child').get('link'))
    outgoing = {}
    for joint in joints:
        parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
        if parent in visible and child in visible:
            origin = joint.find('origin')
            vec = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            outgoing.setdefault(parent, []).append(vec)
    for link in root.findall('link'):
        for collision in link.findall('collision'):
            link.remove(collision)
        if link.get('name') not in visible or geometry == 'links':
            for visual in link.findall('visual'):
                link.remove(visual)
        if link.get('name') not in visible:
            continue
        if geometry == 'meshes':
            for mesh in link.findall('.//visual/geometry/mesh'):
                mesh.set('filename', str(resolve_mesh(mesh.get('filename'), source, packages)))
        else:
            for vec in outgoing.get(link.get('name'), []):
                length = float(np.linalg.norm(vec))
                if not .02 < length < 1.2:
                    continue
                visual = ET.SubElement(link, 'visual')
                rotation, _ = Rotation.align_vectors([vec / length], [[0, 0, 1]])
                ET.SubElement(visual, 'origin', xyz=' '.join(map(str, vec / 2)), rpy=' '.join(map(str, rotation.as_euler('xyz'))))
                geom = ET.SubElement(visual, 'geometry')
                ET.SubElement(geom, 'cylinder', radius='.032', length=str(length))
            geom = ET.SubElement(ET.SubElement(link, 'visual'), 'geometry')
            ET.SubElement(geom, 'sphere', radius='.044')
        if link.find('inertial') is None:
            inertial = ET.SubElement(link, 'inertial')
            ET.SubElement(inertial, 'mass', value='.01')
            ET.SubElement(inertial, 'inertia', ixx='.0001', iyy='.0001', izz='.0001', ixy='0', ixz='0', iyz='0')
    tree.write(target)
    return [(j.get('name'), j.find('mimic').attrib) for j in joints if j.find('mimic') is not None]


def joint_value(item, vector):
    if isinstance(item, int):
        return float(vector[item])
    return float(vector[int(item['dim'])]) * float(item.get('scale', 1.)) + float(item.get('offset', 0.))


class RobotRenderer:
    def __init__(self, urdf, config, states, actions, width=648, height=708,
                 packages=None, geometry='meshes', backend='cpu'):
        self.w, self.h = width, height
        self.robot = 'custom'
        self.cid = p.connect(p.DIRECT)
        self.temporary = tempfile.TemporaryDirectory(prefix='internw0-urdf-')
        self.bodies, self.mapping, self.mimics = [], [], []
        source = Path(urdf).expanduser().resolve()
        instances = config.get('instances') or [{'joints': config.get('joints', config)}]
        joint_names = {name for instance in instances for name in instance['joints']}
        styled = Path(self.temporary.name) / 'robot.urdf'
        try:
            mimics = prepare_urdf(source, styled, joint_names, packages or {}, geometry)
            for instance in instances:
                body = p.loadURDF(str(styled), basePosition=instance.get('base_position', [0, 0, 0]),
                                  baseOrientation=p.getQuaternionFromEuler(instance.get('base_rpy', [0, 0, 0])),
                                  useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.cid)
                names = {p.getJointInfo(body, j, physicsClientId=self.cid)[1].decode(): j
                         for j in range(p.getNumJoints(body, physicsClientId=self.cid))}
                self.bodies.append(body)
                for name, item in instance['joints'].items():
                    self.mapping.append((body, names[name], item))
                for name, value in instance.get('reference_joints', {}).items():
                    p.resetJointState(body, names[name], float(value), physicsClientId=self.cid)
                for name, mimic in mimics:
                    self.mimics.append((body, names[name], names[mimic['joint']], float(mimic.get('multiplier', 1)), float(mimic.get('offset', 0))))
            points = []
            for values in (states, actions):
                if values is None:
                    continue
                for i in np.linspace(0, len(values) - 1, min(24, len(values))).astype(int):
                    self.set_q(values[i])
                    for body, joint, _ in self.mapping:
                        points.append(p.getLinkState(body, joint, computeForwardKinematics=True, physicsClientId=self.cid)[4])
            points = np.asarray(points)
            low, high = points.min(0), points.max(0)
            self.target = (low + high) / 2
            distance = max(1.25, float(np.max(high - low)) * 2.2)
            self.view = p.computeViewMatrixFromYawPitchRoll(self.target.tolist(), distance, 45, -20, 0, 2)
            self.proj = p.computeProjectionMatrixFOV(42, width / height, .03, 20)
            floor = p.createVisualShape(p.GEOM_BOX, halfExtents=[8, 8, .02], rgbaColor=[.095, .125, .17, 1], physicsClientId=self.cid)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor, basePosition=[0, 0, float(low[2]) - .09], physicsClientId=self.cid)
            self.last_color = None
            self.cuda_scene = None
            if backend == 'cuda':
                from .cuda_render import CudaScene
                self.cuda_scene = CudaScene(self)
                self.cuda_scene.fit_camera(states, actions if actions is not None else states)
            self.provenance = {'urdf_filename': source.name, 'urdf_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                               'geometry': geometry, 'backend': backend, 'instances': instances}
        except BaseException:
            self.close()
            raise

    def set_q(self, vector):
        for body, joint, item in self.mapping:
            value = joint_value(item, vector)
            if not np.isfinite(value):
                raise ValueError('Non-finite recorded joint value')
            p.resetJointState(body, joint, value, physicsClientId=self.cid)
        for body, joint, reference, scale, offset in self.mimics:
            value = p.getJointState(body, reference, physicsClientId=self.cid)[0]
            p.resetJointState(body, joint, value * scale + offset, physicsClientId=self.cid)

    def render(self, vector, command=False):
        self.set_q(vector)
        if self.cuda_scene is not None:
            return self.cuda_scene.render(command)
        if self.last_color != command:
            color = [1, .49, .32, 1] if command else [.24, .83, .83, 1]
            for body in self.bodies:
                for joint in [-1] + list(range(p.getNumJoints(body, physicsClientId=self.cid))):
                    p.changeVisualShape(body, joint, rgbaColor=color, specularColor=[.25, .25, .25], physicsClientId=self.cid)
            self.last_color = command
        _, _, rgba, _, segmentation = p.getCameraImage(self.w, self.h, self.view, self.proj,
            lightDirection=[-3, -2, 5], lightColor=[1, .97, .92], lightAmbientCoeff=.5,
            lightDiffuseCoeff=.65, lightSpecularCoeff=.25, shadow=1,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid)
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
        rgb[np.asarray(segmentation).reshape(self.h, self.w) < 0] = [13, 20, 30]
        return rgb

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)
        self.temporary.cleanup()
