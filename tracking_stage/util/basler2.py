'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import cv2

# conecting to the first available camera
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Image format control
# camera.Width = 2040
# camera.Height = 1530
# camera.CenterX.SetValue(True)
# camera.CenterY.SetValue(True)

camera.Width = 1020
camera.Height = 764
camera.OffsetX.SetValue(900)
camera.OffsetY.SetValue(390)

# Analog control
#camera.Gain.SetValue(23)#(19.2)
#camera.Gamma.SetValue(0.82001)#(0.73372)

camera.Gain.SetValue(23.9)
camera.Gamma.SetValue(0.66667)#(0.73372)

# Acquisition control
camera.ExposureTime.SetValue(5000)#1500)#(1290.0)
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(1200.00480)



# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 

converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


def getBasler():
    '''
    Get the basler camera object
    '''
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
    return img

def closeBasler():
    '''
    Close the basler camera object
    '''
    # camera.StopGrabbing()
    camera.Close()