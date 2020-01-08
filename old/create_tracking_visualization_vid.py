import os
from PIL import Image
import moviepy
from moviepy.editor import *

clip = ImageSequenceClip('visualizations/matching_visualization', fps=4)
clip.write_videofile("visualizations/matching_visualization_vid.mp4", fps=4)
