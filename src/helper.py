import numpy as np

# code borrowed from MisGAN
def plot_grid(ax, image, image_size, bbox=None, gap=1, gap_value=1, nrow=4, ncol=8, title=None):
    image = image.cpu().numpy().squeeze(1)
    LEN = image_size
    grid = np.empty((nrow * (LEN + gap) - gap, ncol * (LEN + gap) - gap))
    grid.fill(gap_value)

    for i, x in enumerate(image):
        if i >= nrow * ncol:
            break
        p0 = (i // ncol) * (LEN + gap)
        p1 = (i % ncol) * (LEN + gap)
        grid[p0:(p0 + LEN), p1:(p1 + LEN)] = x

    ax.set_axis_off()
    ax.imshow(grid, cmap='binary_r', interpolation='none', aspect='equal')
            
    if title:
        ax.set_title(title)


# for visualisation purpose
def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau