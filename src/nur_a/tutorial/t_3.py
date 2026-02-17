"""
Scripts for tutorial session 3
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %% question 1
# q1 - image load
IMG = mpimg.imread("../../media/dog_image.jpg")
GREY_IMG = np.dot(IMG[..., :3], [0.640, 0.595, 0.155])  # Rec601 luminanc


# %%
# q1 - func
def img_svd(img=GREY_IMG):
    """svd image compression"""
    mat_u, mat_w, mat_vt = np.linalg.svd(GREY_IMG, full_matrices=False)
    return mat_u, mat_w, mat_vt


def img_svd_reconstitute(img=GREY_IMG, n=0):
    """reconstitute from mats"""
    mat_u, mat_w, mat_vt = img_svd(img)
    # recon via np.dot(mat_u, np.dot(mat_w_zeros, mat_vh))
    if n == 0:
        mat_w_zeros = np.zeros((img.shape[0], img.shape[1]))
        np.fill_diagonal(mat_w_zeros, mat_w)
    else:
        mat_u = mat_u[:, :n]
        mat_w = mat_w[:n]
        mat_vt = mat_vt[:n, :]

    img_recon = mat_u @ np.diag(mat_w) @ mat_vt

    return img_recon


def img_svd_reconstitute_elements(img=GREY_IMG, ns: list = [5, 10, 50]):
    """reconstitute from the first n elements of the mats"""
    mat_u, mat_w, mat_vt = img_svd(img)
    img_size = img.size
    img_recon = []
    compression = 0

    ns_custom = np.arange(0, 400, 10)

    for n in ns_custom:
        # find the depth of elements needed to get a good compression ratio
        if compression < 1.00:
            # first n elements of the mats
            mat_u_n = mat_u[:, :n]
            mat_w_n = mat_w[:n]
            mat_vt_n = mat_vt[:n, :]

            img_recon_n = mat_u_n @ np.diag(mat_w_n) @ mat_vt_n

            img_recon_n_size = mat_u_n.size + n + mat_vt_n.size

            img_recon.append(img_recon_n)

            compression = img_recon_n_size / img_size
            depth = n

    print(f"Compression rate {compression} at {depth} elements.")

    return img_recon


def img_svd_plot(img=GREY_IMG):
    """wrapper for img svd plot gen"""
    img_recon = img_svd_reconstitute(img)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # source img
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Source img")
    ax[0].axis("off")

    # reconstituted img
    ax[1].imshow(img_recon, cmap="gray")
    ax[1].set_title("Reconstituted img")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


# %%
# q1 - func call
img_svd_reconstitute_elements()
img_svd_plot()
# %% question 2
