from matplotlib import pyplot as plt
class CategoricalDistributionsManifold:
    def __init__(self, dim):
        self.dim = dim
        self.points = []
        self.ax = None
    
    def draw(self):
        # 1D
        self.set_axis()

    
    def set_axis(self):
        min_limit = 0
        max_limit = 1
        ax = plt.subplot(111, projection="3d")
        plt.setp(
            ax,
            xlim=(min_limit, max_limit),
            ylim=(min_limit, max_limit),
            zlim=(min_limit, max_limit),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        )
        self.ax = ax


