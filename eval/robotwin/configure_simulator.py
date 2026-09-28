"""Apply the environment settings required by the pinned RoboTwin simulator."""
from importlib.metadata import distribution, version
from pathlib import Path


def configure():
    for package, expected in [('sapien', '3.0.0b1'), ('mplib', '0.2.1')]:
        if version(package) != expected:
            raise RuntimeError(f'{package} must be {expected}; run this with the simulator Python')
    changes = [
        ('sapien', 'sapien/wrapper/urdf_loader.py',
         'with open(urdf_file, "r") as f:', 'with open(urdf_file, "r", encoding="utf-8") as f:'),
        ('sapien', 'sapien/wrapper/urdf_loader.py',
         'with open(srdf_file, "r") as f:', 'with open(srdf_file, "r", encoding="utf-8") as f:'),
        ('mplib', 'mplib/planner.py',
         'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:',
         'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'),
    ]
    prepared = {}
    for package, relative, before, after in changes:
        path = Path(distribution(package).locate_file(relative))
        source = prepared.get(path, path.read_text())
        if after not in source:
            if source.count(before) != 1:
                raise RuntimeError(f'Unexpected simulator package source: {path}')
            source = source.replace(before, after)
        prepared[path] = source
    for path, source in prepared.items():
        if path.read_text() != source:
            path.write_text(source)
    print('RoboTwin simulator environment configured')


if __name__ == '__main__':
    configure()
