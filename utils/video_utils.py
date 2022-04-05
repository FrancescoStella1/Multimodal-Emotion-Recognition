import numpy as np
import os, sys

import cv2


class VideoManager:
    """Class aimed at managing frame extraction from video clips."""
    def __init__(self, filepath, skip_frames):
        try:
            self.curr_frame = 0
            self.file = cv2.VideoCapture(filepath)
            self.filepath = filepath
            self.skip_frames = skip_frames
            self.subj = filepath.split(os.sep)[-1][1:3]
            self.trial = filepath.split(os.sep)[-1][9:11]
            
        except Exception as e:
            print("[ VideoManager __init__() ] error: ", repr(e), file=sys.stderr)

    def __del__(self):
        """Customized destructor, releases the cv2 video capture."""
        self.file.release()

    def _reset_with_values(self, filepath):
        """Changes attributes of a VideoManager object in order to set it properly for processing
        the new video clip in filepath."""
        try:
            self.curr_frame = 0
            self.file.release()
            self.file = cv2.VideoCapture(filepath)
            self.filepath = filepath
            self.subj = filepath.split(os.sep)[-1][1:3]
            self.trial = filepath.split(os.sep)[-1][9:11]

        except Exception as e:
            print("[ VideoManager ] error while resetting values: ", repr(e), file=sys.stderr)

    def read_frame(self):
        r"""
        Reads the next frame after skipping the number of frames defined for the VideoManager object.        

        Returns
        -------
        frame: array_like (or None)
            Frame extracted from the video (or None).
        """
        if self.curr_frame is None: return None
        #print("Current frame: ", self.curr_frame)
        self.file.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame)
        isFrame, frame = self.file.read()
        self.curr_frame += self.skip_frames
        if not isFrame: 
            self.curr_frame = None
            frame = None
        return frame

    def read_frame_at(self, index):
        r"""
        Reads a frame with specific index from the file associated with the
        VideoManager object.

        Parameters
        ----------
        index: int
            Index of the frame to read.

        Returns
        -------
            frame: array_like
                Frame at the specified index (or None).
        """
        try:
            self.file.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.curr_frame = index
            isFrame, frame = self.file.read()
            if not isFrame:
                frame = None
            return frame

        except Exception as e:
            print("Error while reading frame at index {}: ".format(index), repr(e), file=sys.stderr)

    def read_all_frames(self):
        r"""
        Reads all frames from the file associated with the VideoManager object.
        
        Returns
        -------
        frames: array_like
            Frames extracted from the video.
        """
        print("Reading all frames..")
        frames = None
        isFrame = True
        self.curr_frame = 0
        while isFrame:
            self.file.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame)
            isFrame, frame = self.file.read()
            if not isFrame: break
            frame = np.reshape(frame, (1, frame.shape[0], frame.shape[1], frame.shape[2]))
            if frames is None: frames = frame 
            else: frames = np.row_stack((frames, frame))
            self.curr_frame += self.skip_frames
        self.curr_frame = None
        return frames

    def get_subject(self):
        return self.subj

    def get_trial(self):
        return self.trial
