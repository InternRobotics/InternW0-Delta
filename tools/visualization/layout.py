"""Recorded camera, paired URDF views, and synchronized physical signal chart."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH, HEIGHT = 1920, 1080
BG, PANEL, WHITE, MUTED = '#080d15', '#0d141e', '#edf3fa', '#8b9cb2'
CYAN, ORANGE, GREEN = '#4fddd8', '#ff986c', '#55d79f'


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def label(draw, xy, value, size=24, color=WHITE, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=color)


def shorten(value, length):
    return value if len(value) <= length else value[:length - 1] + '…'


class ReplayLayout:
    def __init__(self, dataset, episode, source_frames, times, state, action, dimension, reference, camera):
        self.frames, self.times = source_frames, np.asarray(times)
        image = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(image)
        label(draw, (52, 24), 'InternW0-delta  /  DATA REPLAY', 20, CYAN, True)
        label(draw, (50, 70), 'Recorded motion and URDF joint replay', 44, bold=True)
        label(draw, (53, 130), shorten(dataset, 108), 23, MUTED)
        label(draw, (54, 181), 'RECORDED VIDEO', 18, MUTED, True)
        label(draw, (1007, 181), 'MEASURED STATE', 18, CYAN, True)
        label(draw, (1447, 181), 'COMMANDED POSE', 18, ORANGE, True)
        for box in ((50, 218, 962, 690), (996, 218, 1428, 690), (1440, 218, 1872, 690)):
            draw.rounded_rectangle(box, 12, fill=PANEL)
        label(draw, (53, 706), shorten(camera, 60), 22, MUTED)
        label(draw, (1004, 706), shorten(reference, 57), 19, MUTED)
        draw.rounded_rectangle((50, 762, 1872, 993), 14, fill=PANEL)
        label(draw, (76, 777), f'Joint dimension {dimension} · recorded position (rad)', 21, bold=True)
        series = [(np.asarray(state)[:, dimension], CYAN, 'state')]
        if action is not None:
            series.append((np.asarray(action)[:, dimension], ORANGE, 'command'))
        self.chart = (128, 1839, 823, 954)
        x0, x1, y0, y1 = self.chart
        vals = np.concatenate([s for s, _, _ in series])
        vals = vals[np.isfinite(vals)]
        low, high = float(vals.min()), float(vals.max())
        margin = max(.05, (high - low) * .1)
        low, high = low - margin, high + margin
        xs = self._x(np.arange(len(times)))
        for value in np.linspace(low, high, 4):
            y = y1 - (value - low) / (high - low) * (y1 - y0)
            draw.line((x0, y, x1, y), fill='#232e3f')
            label(draw, (65, y - 8), f'{value:.2g}', 15, MUTED)
        for i in np.linspace(0, len(times) - 1, 5).astype(int):
            label(draw, (xs[i] - 20, y1 + 10), f'{times[i]:.1f}s', 16, MUTED)
        for n, (values, color, title) in enumerate(series):
            points = [(float(x), y1 - (float(v) - low) / (high - low) * (y1 - y0)) for x, v in zip(xs, values)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            label(draw, (1410 + n * 220, 777), title, 20, color)
        label(draw, (52, 1021), f'EPISODE {episode:06d}  ·  source timing  ·  joint replay reference', 19, MUTED)
        self.static = image

    def _x(self, indices):
        t = self.times[indices]
        return 128 + (t - self.times[0]) / max(1e-9, self.times[-1] - self.times[0]) * (1839 - 128)

    def frame(self, index, video, measured, command, timestamp, replay=False):
        image = self.static.copy()
        picture = ImageOps.contain(video, (912, 472))
        image.paste(picture, (50 + (912 - picture.width) // 2, 218 + (472 - picture.height) // 2))
        image.paste(Image.fromarray(measured).resize((432, 472), Image.Resampling.LANCZOS), (996, 218))
        if command is not None:
            image.paste(Image.fromarray(command).resize((432, 472), Image.Resampling.LANCZOS), (1440, 218))
        draw = ImageDraw.Draw(image)
        if command is None:
            label(draw, (1470, 425), 'No joint target', 22, MUTED)
        accent = GREEN
        draw.rectangle((50, 218, 961, 689), outline=accent, width=3)
        status = 'OBSERVATION'
        draw.rectangle((64, 231, 306, 274), fill=PANEL)
        label(draw, (77, 240), status, 21, accent, True)
        draw.rectangle((693, 638, 947, 678), fill=PANEL)
        label(draw, (704, 646), f'{timestamp:.2f}s  |  ' + ('0.25x' if replay else '1x'), 22)
        if replay:
            draw.rectangle((644, 231, 947, 274), fill=PANEL)
            label(draw, (655, 240), 'SLOW MOTION REPLAY', 21, ORANGE)
        x = float(self._x(index))
        draw.line((x, 819, x, 959), fill=WHITE, width=2)
        return image
