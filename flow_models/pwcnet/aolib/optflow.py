import numpy as np, scipy.io, scipy.ndimage, aolib.img as ig, aolib.util as ut, os

def warp_flow(im2, vx, vy):
  """ Assumes im1[y, x] = im2[y + flow[y, x, 1], x + flow[y, x, 0]] """
  
  # this code is confusing because we assume vx and vy are the negative
  # of where to send each pixel, as in the results by ce's siftflow code
  y, x = np.mgrid[:im2.shape[0], :im2.shape[1]]
  dy = (y + vy).flatten()[np.newaxis, :]
  dx = (x + vx).flatten()[np.newaxis, :]
  # this says: a recipe for making im1 is to make a new image where im[y, x] = im2[y + flow[y, x, 1], x + flow[y, x, 0]]
  return np.concatenate([scipy.ndimage.map_coordinates(im2[..., i], np.concatenate([dy, dx])).reshape(im2.shape[:2] + (1,)) \
                         for i in range(im2.shape[2])], axis = 2)

# given a flow that warps im2 -> im1, returns a warped version that warps im1 to im2
def invert_warp(im1, vx, vy):
  y, x = np.mgrid[:im1.shape[0], :im1.shape[1]]
  im_coords = np.concatenate([x[..., np.newaxis], y[..., np.newaxis]], axis = 2)
  new_flow = warp_flow(im_coords, vx, vy) - im_coords
  return warp_flow(im1, -new_flow[...,0], -new_flow[...,1])

def test_invert_warp():
  im1 = ig.draw_rects(ig.make(200, 200), [(100, 50, 100, 100)], fills = (0, 0, 255))
  im2 = ig.draw_rects(ig.make(200, 200), [(50, 50, 100, 100)], fills = (255, 0, 0))
  vx = -50 + np.zeros(im1.shape[:2])
  vy = np.zeros(im1.shape[:2])
  ig.show([[im1, im2], warp_flow(im2, vx, vy), invert_warp(im1, vx, vy)])

  
#def color_flow(flow, max_flow = None, show = True):
def color_flow(flow, max_flow = -1, show = True):
  flow = flow.copy()
  # based on flowToColor.m by Deqing Sun, orignally based on code by Daniel Scharstein
  UNKNOWN_FLOW_THRESH = 1e9
  UNKNOWN_FLOW = 1e10
  height, width, nbands = flow.shape
  assert nbands == 2
  u, v = flow[:,:,0], flow[:,:,1]
  maxu = -999.
  maxv = -999.
  minu = 999.
  minv = 999.
  maxrad = -1.

  idx_unknown = ut.lor(np.abs(u) > UNKNOWN_FLOW_THRESH, np.abs(v) > UNKNOWN_FLOW_THRESH)
  u[idx_unknown] = 0
  v[idx_unknown] = 0
  
  maxu = max(maxu, np.max(u))
  maxv = max(maxv, np.max(v))
  
  minu = min(minu, np.min(u))
  minv = min(minv, np.min(v))

  rad = np.sqrt(u**2 + v**2)
  maxrad = max(maxrad, np.max(rad))

  if show:
    print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

  if max_flow > 0:
    maxrad = max_flow

  u = u/(maxrad + np.spacing(1))
  v = v/(maxrad + np.spacing(1))
  
  im = compute_color(u, v)
  im[idx_unknown] = 0
  return im

def compute_color(u, v):
  # from same source as color_flow; please see above comment
  nan_idx = ut.lor(np.isnan(u), np.isnan(v))
  u[nan_idx] = 0
  v[nan_idx] = 0
  colorwheel = make_color_wheel()
  ncols = colorwheel.shape[0]
  
  rad = np.sqrt(u**2 + v**2)

  a = np.arctan2(-v, -u)/np.pi
  
  #fk = (a + 1)/2. * (ncols-1) + 1
  fk = (a + 1)/2. * (ncols-1)

  k0 = np.array(np.floor(fk), 'l')

  k1 = k0 + 1
  k1[k1 == ncols] = 1

  f = fk - k0

  im = np.zeros(u.shape + (3,))
  
  for i in range(colorwheel.shape[1]):
    tmp = colorwheel[:, i]
    col0 = tmp[k0]/255.
    col1 = tmp[k1]/255.
    col = (1-f)*col0 + f*col1

    idx = rad <= 1
    col[idx] = 1 - rad[idx]*(1-col[idx])
    col[ut.lnot(idx)] *= 0.75
    im[:, :, i] = np.uint8(np.floor(255*col*(1-nan_idx)))

  return im

def make_color_wheel():
  # same source as color_flow

  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR

  #colorwheel = zeros(ncols, 3) # r g b
  # matlab correction
  colorwheel = np.zeros((1+ncols, 4)) # r g b

  col = 0
  #RY
  colorwheel[1:1+RY, 1] = 255
  colorwheel[1:1+RY, 2] = np.floor(255*np.arange(0, 1+RY-1)/RY).T
  col = col+RY

  #YG
  colorwheel[col+1:col+1+YG, 1] = 255 - np.floor(255*np.arange(0,1+YG-1)/YG).T
  colorwheel[col+1:col+1+YG, 2] = 255
  col = col+YG

  #GC
  colorwheel[col+1:col+1+GC, 2] = 255
  colorwheel[col+1:col+1+GC, 3] = np.floor(255*np.arange(0,1+GC-1)/GC).T
  col = col+GC

  #CB
  colorwheel[col+1:col+1+CB, 2] = 255 - np.floor(255*np.arange(0,1+CB-1)/CB).T
  colorwheel[col+1:col+1+CB, 3] = 255
  col = col+CB

  #BM
  colorwheel[col+1:col+1+BM, 3] = 255
  colorwheel[col+1:col+1+BM, 1] = np.floor(255*np.arange(0,1+BM-1)/BM).T
  col = col+BM

  #MR
  colorwheel[col+1:col+1+MR, 3] = 255 - np.floor(255*np.arange(0,1+MR-1)/MR).T
  colorwheel[col+1:col+1+MR, 1] = 255  

  # 1-based to 0-based indices
  return colorwheel[1:, 1:]
 

def ldof(im1, im2, max_dim = None, rescale = True):
  im1_shape = im1.shape
  with ut.temp_file('.ppm') as fname1, ut.temp_file('.ppm') as fname2:
    if max_dim is not None:
      s = max_dim / float(max(im1.shape[:2]))
      im1 = ig.scale(im1, s)
      im2 = ig.scale(im2, s)
    else:
      s = 1.
    ig.save(fname1, im1)
    ig.save(fname2, im2)
    ut.sys_check('../lib/ldof/ldof "%s" "%s"' % (fname1, fname2))
    out_fname = fname1.replace('.ppm', 'LDOF.flo')
    flo = np.array(np.fromfile(out_fname, np.float32), 'float32')[3:].reshape((im1.shape[0], im1.shape[1], 2))
    os.remove(out_fname)
    if rescale and max_dim is not None:
      flo = (1./s) * ig.scale(flo, im1_shape)
    return flo


def deepflow(im1, im2, max_dim = None, rescale = True, match = False, shape = None):
  if max_dim is not None:
    im1_shape = im1.shape
  with ut.temp_file('.png') as fname1, \
           ut.temp_file('.png') as fname2, \
           ut.temp_file('.flo') as flow_file:
    if max_dim is not None:
      s = max_dim / float(max(im1.shape[:2]))
      im1 = ig.scale(im1, s)
      im2 = ig.scale(im2, s)
    else:
      s = 1.

    if type(im1) != type('') or type(im2) != type(im2):
      ig.save(fname1, im1)
      ig.save(fname2, im2)
      new_shape = im1.shape
    else:
      fname1, fname2 = im1, im2
      new_shape = shape

    if match:
      # ut.sys_check(('../lib/deepmatching_1.2.2_c++/deepmatching-static "%s" "%s" | '
      #               '../lib/DeepFlow_release2.0/deepflow2-static "%s" "%s" "%s" -match') \
      #              % (fname1, fname2, fname1, fname2, flow_file))
      ut.sys_check(('../lib/deepmatching_1.2.2_c++/deepmatching-static "%s" "%s" -downscale 2 | '
                    '../lib/DeepFlow_release2.0/deepflow2-static "%s" "%s" "%s" -match') \
                   % (fname1, fname2, fname1, fname2, flow_file))
    else:
      ut.sys_check('../lib/DeepFlow_release2.0/deepflow2-static "%s" "%s" "%s"' % (fname1, fname2, flow_file))

    flo = np.array(np.fromfile(flow_file, np.float32), 'float32')[3:].reshape((new_shape[0], new_shape[1], 2))
    if rescale and max_dim is not None:
      flo = (1./s) * ig.scale(flo, im1_shape)
    return flo

def flow_im(im, flo, dim, scale = 2.5):
  # Similar to the representation in Donahue et. al.'s LRCN paper
  flo_scaled = flo * (dim/float(flo.shape[0]))
  flo_scaled = ig.scale(flo_scaled, (dim, dim))
  flo_byte = np.uint8(np.clip((flo_scaled * scale) + 128, 0, 255))
  # Make the last channel be a grayscale version of the image itself
  both = np.zeros((dim, dim, 3))
  both[:, :, :2] = flo_byte
  both[:, :, 2] = ig.scale(ig.luminance(im), (dim, dim))
  return np.uint8(both)

  
