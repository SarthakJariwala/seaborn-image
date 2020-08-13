import matplotlib.pyplot as plt


class ImageGrid(object):

    def __init__(
        self,
        mosaic=None,
        rows=1,
        cols=1,
        height=5,
        aspect=1,
    ):

        self.mosaic = mosaic
        self.rows = rows
        self.cols = cols
        self.height = height
        self.aspect = aspect

        self.fig = plt.figure(figsize=(self.height * self.aspect, self.height))

        if self.mosaic is not None:
            self.rows = None
            self.cols = None
            self.mosaic_ax = self.fig.subplot_mosaic(self.mosaic)

        if self.rows is not None:
            self.fig.subplots(self.rows, self.cols)

        return
