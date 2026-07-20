from __future__ import annotations
from skimage.transform import rescale
import numpy as np
from time import perf_counter
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass

__all__= [
    "track_one_T",
    "segment_single_img",
    "find_com",
    "update_pos_points",
]

# def track_one_T(labels: np.ndarray, scale: int, pts, radius: float=15):
#     radius = radius / scale
    
#     objects = btrack.utils.segmentation_to_objects(labels)
#     with btrack.BayesianTracker(verbose=False) as tracker:
#         tracker.configure(Path(__file__) / "particle_config.json")
#         tracker.max_search_radius = radius

#         # append the objects to be tracked
#         tracker.append(objects)

#         # set the tracking volume
#         tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))

#         # track them (in interactive mode)
#         tracker.track(step_size=100)

#         # generate hypotheses and run the global optimizer
#         tracker.optimize()

#         # tracker.export(tracks_out, obj_type='obj_type_1')
#         tracks = tracker.tracks
#     # all_tracks = pd.concat([pd.DataFrame(t.to_dict()) for t in tracks])
#     tracked = btrack.utils.update_segmentation(labels, tracks)
    
#     # return tracked
#     pts = np.atleast_2d(pts)
    
#     new_aim = []
#     for pt in pts:
#         pt = (np.array(pt)/scale).astype(int)
#         label = tracked[0, pt[0], pt[1]]
#         if np.sum(tracked[1] == label) != 0:
#             new_aim.append((np.array(ndi.center_of_mass(tracked[1]==label))*scale))
#         else:
#             new_aim.append((pt*scale).astype(int)) 
            
#     return new_aim

# def track_one_T(labels: np.ndarray, scale: int, pts, radius: float = 5, threshold=60, tracked=None, use_same_img=False):
#     radius = radius / scale
#     if not use_same_img:
#         objects = btrack.utils.segmentation_to_objects(labels)
#         with btrack.BayesianTracker(verbose=False) as tracker:
#             tracker.configure("particle_config.json")
#             tracker.max_search_radius = radius

#             # append the objects to be tracked
#             tracker.append(objects)

#             # set the tracking volume
#             tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))

#             # track them (in interactive mode)
#             tracker.track(step_size=100)

#             # generate hypotheses and run the global optimizer
#             tracker.optimize()

#             # tracker.export(tracks_out, obj_type='obj_type_1')
#             tracks = tracker.tracks
#         # all_tracks = pd.concat([pd.DataFrame(t.to_dict()) for t in tracks])
#         tracked = btrack.utils.update_segmentation(labels, tracks)

#     # return tracked
#     pts = np.atleast_2d(pts)

#     new_aim = []
#     for pt in pts:
#         pt = (np.array(pt) / scale).astype(int)
#         label = tracked[0, pt[0], pt[1]]
#         if label == 0:
#             # cellpose broke somehow, just leave it still and hope
#             # for the best
#             new_label = tracked[1, pt[0], pt[1]]
#             if new_label == 0:
#                 id = np.unique(tracked[1])
#                 id = id[id != 0]  # Ignore background

#                 coms = np.array([center_of_mass(tracked[1] == i) for i in id])
#                 dists = np.linalg.norm(coms - pt, axis=1)

#                 min_dist_idx = np.argmin(dists)
#                 min_dist = dists[min_dist_idx]

#                 if min_dist > 3*threshold/scale:
#                     print(f"lost tracking of a cell {pt*scale}, no potential cells within threshold")
#                     new_aim.append((pt * scale).astype(int))
#                 else:
#                     new_id = id[min_dist_idx]
#                     print(f"lost tracking of a cell {pt*scale}, moved to closest point within threshold")
#                     new_aim.append(
#                         (np.array(ndi.center_of_mass(tracked[1] == new_id)) * scale)
#                     )
                
#             else:
#                 # send_slack_message(f"maybe lost tracking of a cell {pt*scale}")
#                 print(f"maybe lost tracking of a cell {pt*scale}")
#                 new_aim.append(
#                     (np.array(ndi.center_of_mass(tracked[1] == new_label)) * scale)
#                 )

#         elif np.sum(tracked[1] == label) != 0:
#             new_aim.append((np.array(ndi.center_of_mass(tracked[1] == label)) * scale))
#         else:
#             new_aim.append((pt * scale).astype(int))

#     return tracked, new_aim


import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass
import multiprocessing as mp


def _new_cellpose_model():
    """Create Cellpose without making it a package import-time dependency."""
    try:
        from cellpose.models import Cellpose
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Cellpose/PyTorch could not be loaded. Install the "
            "'raman-mda-engine[tracking]' extra in a binary-consistent "
            "environment. On Windows, do not mix Conda MKL/OpenMP runtimes "
            "with a pip PyTorch wheel."
        ) from exc

    return Cellpose(model_type="cyto2", gpu=False)


def run_tracking(labels, radius):
    try:
        import btrack
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "BTrack could not be loaded. Install the "
            "'raman-mda-engine[tracking]' extra."
        ) from exc

    objects = btrack.utils.segmentation_to_objects(labels)
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure("particle_config.json")
        tracker.max_search_radius = radius
        tracker.volume = ((0, labels.shape[-2]), (0, labels.shape[-1]), (-1e5, 1e5))
        tracker.append(objects)
        tracker.track(step_size=100)
        tracker.optimize()
        tracks = tracker.tracks

        if len(tracks) == 0:
            return labels
        
    return btrack.utils.update_segmentation(labels, tracks)

def track_with_timeout(labels, radius, timeout_sec=30):
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        async_result = pool.apply_async(run_tracking, (labels, radius))
        try:
            return async_result.get(timeout=timeout_sec)
        except mp.context.TimeoutError:
            print(f"BTrack timed out after {timeout_sec}s. Using fallback.")
            return None
        except Exception as e:
            print(f"BTrack failed in worker: {e!r}. Using fallback.")
            return None

def track_one_T(labels: np.ndarray, scale: int, pts, radius: float = 5, threshold=60, tracked=None, use_same_img=False):
    radius = radius / scale
    if not use_same_img:
        tracked = track_with_timeout(labels, radius, timeout_sec=300)
        if tracked is None:
            # fallback: use raw segmentation with no tracking
            tracked = labels.copy()
            # tracked = np.stack([tracked]*2)  # fake 2 timepoints if needed

    pts = np.atleast_2d(pts)
    new_aim = []

    for pt in pts:
        pt = (np.array(pt) / scale).astype(int)
        label = tracked[0, pt[0], pt[1]]
        if label == 0:
            new_label = tracked[1, pt[0], pt[1]]
            if new_label == 0:
                id = np.unique(tracked[1])
                id = id[id != 0]
                coms = np.array([center_of_mass(tracked[1] == i) for i in id])
                dists = np.linalg.norm(coms - pt, axis=1)
                if len(dists) == 0 or np.min(dists) > 3 * threshold / scale:
                    print(f"lost tracking of a cell {pt * scale}, no potential cells within threshold")
                    new_aim.append((pt * scale).astype(int))
                else:
                    new_id = id[np.argmin(dists)]
                    print(f"lost tracking of a cell {pt * scale}, moved to closest point within threshold")
                    new_aim.append((np.array(center_of_mass(tracked[1] == new_id)) * scale))
            else:
                print(f"maybe lost tracking of a cell {pt * scale}")
                new_aim.append((np.array(center_of_mass(tracked[1] == new_label)) * scale))
        elif np.sum(tracked[1] == label) != 0:
            new_aim.append((np.array(center_of_mass(tracked[1] == label)) * scale))
        else:
            new_aim.append((pt * scale).astype(int))

    return tracked, new_aim

def mask_outside_circle(img, circle_center=(540, 740), circle_radius=400):
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]

    dist_sq = (X - circle_center[0])**2 + (Y - circle_center[1])**2

    mask = dist_sq <= circle_radius**2

    masked_img = img.copy()
    masked_img[~mask] = img.min()

    return masked_img


def segment_single_img(img: np.ndarray, scale: int = 4, crop=True):
    model = _new_cellpose_model()
    channels = [[0, 0]]

    # seg_imgs = img[::scale, ::scale]
    if crop:
        img = mask_outside_circle(img)
    seg_imgs = rescale(img, 1 / scale, anti_aliasing=True)
    seg_imgs = (seg_imgs - seg_imgs.min()) / (seg_imgs.max() - seg_imgs.min())
    
    masks, flow, styles = model.cp.eval(
        seg_imgs,
        batch_size=1024,
        channels=channels,
        diameter=50/scale,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        # tile=False,
        # tile_overlap=0
    )
    
    return masks


def find_com(img: np.ndarray, pt_xy: np.ndarray, scale: int=4, dist_thres: float=80, plot=False)->np.ndarray:
    """
    Find the COM of a moving object of interest by creating masks with cellpose.
    Return the closest match if no overlap is found, and return the input if the closest match is too far away.
    
    Parameters
    ----------
    pt_xy : (int, int)
        the coordinates of the scan point
    scale: int
        the downscaling scalar to improve segmentation speed
    dist_thres: float
        a distance threshold above which a point is not going to be considered to be matched with a label
    plot: bool
        Show a snapshot of the image and its cellposed mask alone with pt_xy
        
    Returns
    -------
    new_aim: (int, int)
        The new coordinates of the scan points
    """
    t_start = perf_counter()
    t0 = perf_counter()

    model = _new_cellpose_model()
    channels = [[0, 0]]

    # seg_imgs = img[::scale, ::scale]
    seg_imgs = rescale(img, 1/scale, anti_aliasing=True)
    seg_imgs = (seg_imgs - seg_imgs.min())/(seg_imgs.max() - seg_imgs.min())

    masks, flow, styles = model.cp.eval(
        seg_imgs,
        batch_size=1024,
        channels=channels,
        diameter=40/scale,
        flow_threshold=0.6,
        cellprob_threshold=-2,
        normalize=False,
        tile=False,
        tile_overlap=0
    )
    
    pt_xy = np.atleast_2d(pt_xy)
    
    new_aim = []
    for pt in pt_xy:
        pt = (np.array(pt)/scale).astype(int)

        label = masks[pt[0], pt[1]]
        if label != 0:
            print('label found (exact match)')
            new_aim.append((np.array(ndi.center_of_mass(masks==label))*scale).astype(int))
        else:
            distances = []
            for i in range(1, masks.max()+1):
                distances.append(np.linalg.norm(pt - ndi.center_of_mass(masks==i)))
            if np.min(distances) <= dist_thres/scale:
                print('label found (closest match)')
                new_aim.append((np.array(ndi.center_of_mass(masks==np.argmin(distances)+1))*scale).astype(int))
            else:
                print('label not found')
                new_aim.append((pt*scale).astype(int))
                
    new_aim = np.asarray(new_aim)

    if plot:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
        ax[0].imshow(seg_imgs, cmap='gray')
        ax[0].scatter(pt_xy[:, 1]/scale, pt_xy[:, 0]/scale, c='r', marker='x')
        ax[1].imshow(masks)
        ax[1].scatter(pt_xy[:, 1]/scale, pt_xy[:, 0]/scale, c='r', marker='x')
        ax[1].scatter(new_aim[:, 1]/scale, new_aim[:, 0]/scale, c='k', marker='x')
            
    return new_aim

# from qtpy.QtCore import QTimer, QEventLoop

# def run_on_main_thread(fn):
#     loop = QEventLoop()
#     def wrapped():
#         try:
#             fn()
#         finally:
#             loop.quit()
#     QTimer.singleShot(0, wrapped)
#     loop.exec_()  # blocks until loop.quit() is called
   
def update_pos_points(P, new_pts, points_layer, p_idx=1):
    """
    Parameters
    ----------
    P : int
        The current position step
    new_pts : (N, 2) array
        the new points to set
    points_layer : poitns layer
        the layer to update
    p_idx : int
        Which dimension is position
    """
    new = np.copy(points_layer.data)
    # new[:, -1] = 600
    new[new[:, p_idx] == P,-2:] = new_pts
    points_layer.data = new
    # run_on_main_thread(lambda: setattr(points_layer, 'data', new))