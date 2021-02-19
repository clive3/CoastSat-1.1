from coastsat.SDS_preprocess import *

# Main function to preprocess S1 satellite image
def preprocess_sar(fn, satname):

    fn10 = fn[0]
    data = gdal.Open(fn10, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im10 = np.stack(bands, 2)
    im10 = im10 / 10000  # TOA scaled to 10000

    # if image contains only zeros (can happen with S2), skip the image
    if sum(sum(sum(im10))) < 1:
        im_ms = []
        georef = []
        # skip the image by giving it a full cloud_mask
        cloud_mask = np.ones((im10.shape[0], im10.shape[1])).astype('bool')
        return im_ms, georef, cloud_mask, [], [], []

    # size of 10m bands
    nrows = im10.shape[0]
    ncols = im10.shape[1]

    # read 20m band (SWIR1)
    fn20 = fn[1]
    data = gdal.Open(fn20, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im20 = np.stack(bands, 2)
    im20 = im20[:, :, 0]
    im20 = im20 / 10000  # TOA scaled to 10000

    # resize the image using bilinear interpolation (order 1)
    im_swir = transform.resize(im20, (nrows, ncols), order=1, preserve_range=True,
                               mode='constant')
    im_swir = np.expand_dims(im_swir, axis=2)

    # append down-sampled SWIR1 band to the other 10m bands
    im_ms = np.append(im10, im_swir, axis=2)

    # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
    fn60 = fn[2]
    data = gdal.Open(fn60, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    im60 = np.stack(bands, 2)
    im_QA = im60[:, :, 0]
    cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue)
    # resize the cloud mask using nearest neighbour interpolation (order 0)
    cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                  mode='constant')
    # check if -inf or nan values on any band and create nodata image
    im_nodata = np.zeros(cloud_mask.shape).astype(bool)
    for k in range(im_ms.shape[2]):
        im_inf = np.isin(im_ms[:, :, k], -np.inf)
        im_nan = np.isnan(im_ms[:, :, k])
        im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
    # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
    # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
    im_zeros = np.ones(im_nodata.shape).astype(bool)
    im_zeros = np.logical_and(np.isin(im_ms[:, :, 1], 0), im_zeros)  # Green
    im_zeros = np.logical_and(np.isin(im_ms[:, :, 3], 0), im_zeros)  # NIR
    im_20_zeros = transform.resize(np.isin(im20, 0), (nrows, ncols), order=0,
                                   preserve_range=True, mode='constant').astype(bool)
    im_zeros = np.logical_and(im_20_zeros, im_zeros)  # SWIR1
    # add to im_nodata
    im_nodata = np.logical_or(im_zeros, im_nodata)
    # dilate if image was merged as there could be issues at the edges
    if 'merged' in fn10:
        im_nodata = morphology.dilation(im_nodata, morphology.square(5))

    # update cloud mask with all the nodata pixels
    cloud_mask = np.logical_or(cloud_mask, im_nodata)

    # the extra image is the 20m SWIR band
    im_extra = im20

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata

