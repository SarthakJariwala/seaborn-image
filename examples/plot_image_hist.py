"""
Image distribution
==================

"""
import seaborn_image as isns

img = isns.load_image("polymer")

_ = isns.imghist(img, cmap="YlGnBu_r", dx=15, units="nm")
