"""Sequential nearest-timestamp video access with bounded frame memory."""
import av


class VideoReader:
    def __init__(self, path, first_timestamp=0):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        if self.stream.time_base and first_timestamp > 0:
            self.container.seek(int(first_timestamp / self.stream.time_base), stream=self.stream, backward=True)
        self.frames = iter(self.container.decode(self.stream))
        self.previous = self.current = None
        self.finished = False
        self.last_query = float("-inf")
        self.max_error = 0.

    def frame_at(self, timestamp):
        if timestamp < self.last_query:
            raise ValueError("Video timestamps must be monotonic")
        self.last_query = timestamp
        while not self.finished and (self.current is None or self.current[0] < timestamp):
            try:
                frame = next(self.frames)
            except StopIteration:
                self.finished = True
                break
            if frame.time is None:
                raise ValueError("Video frame has no presentation timestamp")
            self.previous, self.current = self.current, (float(frame.time), frame.to_image())
        candidates = [item for item in (self.previous, self.current) if item is not None]
        if not candidates:
            raise ValueError("Video contains no readable frames")
        chosen = min(candidates, key=lambda item: abs(item[0] - timestamp))
        self.max_error = max(self.max_error, abs(chosen[0] - timestamp))
        return chosen[1]

    def close(self):
        self.container.close()
