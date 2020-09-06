"""
Image distribution
==================

"""
import seaborn_image as isns

img = isns.load_image("polymer")

_ = isns.imghist(img, cmap="ice", dx=15, units="nm")
