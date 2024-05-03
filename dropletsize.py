import cv2 as cv
import numpy as np
from os.path import *
import datetime

def get_aerodynamic_size(droplet_diameter, C = 0.09):
    # C - float, mass concentration of salt in droplet
    # get the particle size
    particle_size = droplet_diameter*C
    rho_p = 2.17 # g/cm^3

    # stokes settling velocity
    shape_factor = 1 # source from the FMAG manual

    # aerodynamic diameter
    aerodynamic_diameter = particle_size*(rho_p/shape_factor)**(1/2)

    return aerodynamic_diameter

def convert_paper_size_to_actual_size(droplet_diameter):
    return 1*droplet_diameter

class ScannedSPOTImage:
    def __init__(self, file_path, scale=1e-3):
        # filepath - os.path, file with the image
        # scale - float, scale for the image in units of [um/pix]

        # initialize the filepath and scale
        self.file_path = file_path
        self.scale = scale

        # get the name of the file
        file_name = basename(file_path)

        # parse the name of the file
        info_array = file_name[:-4].split("_")
        
        # set the droplet diameter
        self.droplet_diameter = int(info_array[0].replace("dd", ""))

        # set the trial number
        self.trial_number = int(info_array[1][1])

        # set the location number
        self.location_number = int(info_array[1][2])

        # set the date collected
        self.date_collected = datetime.datetime.strptime(info_array[2], r"%m%d%Y").date()

    def size_particles(self, thresholds = (0, 50)):
        # read the image
        image = cv.imread(self.file_path, cv.IMREAD_GRAYSCALE)

        # apply gaussian blur to reduce noise
        _, thresholded_image = cv.threshold(image, 0, thresholds[1], cv.THRESH_BINARY)

        # apply the Canny edge detection
        edges = cv.Canny(255 - thresholded_image, 255 - thresholds[1], 255 - thresholds[0])

        # find the contours
        contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # initialize the contour size array
        sizes = np.zeros((len(contours), ))

        # iterate through the contours
        for i, contour in enumerate(contours):
            # Calculate the diameter using the area of the contour
            area = cv.contourArea(contour)
            diameter = convert_paper_size_to_actual_size(np.sqrt(4 * area / np.pi))

            # add it to the size array
            sizes[i] = diameter*self.scale

        # set as class attribute
        self.sizes = sizes

        # show the contours image
        window = cv.namedWindow("contour", cv.WINDOW_NORMAL)
        contours_image = cv.resize(cv.drawContours(image, contours, -1, (255, 0, 0)), (1200, int(0.23*1200)))
        cv.imshow('contour', contours_image)
        cv.waitKey(0)

    def __str__(self):
        # format the string
        string = "SPOT Image with droplet diameter: {}, Trial-Location number: {}-{} ({})".format({self.droplet_diameter,
                                                                                                   self.trial_number,
                                                                                                   self.location_number,
                                                                                                   self.date_collected})
        
        return string