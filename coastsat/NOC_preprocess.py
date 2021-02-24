from coastsat.SDS_preprocess import *

from utils.print_utils import printWarning, printProgress


def preprocess_sar(file_name, satname):

    data = gdal.Open(file_name, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    sar_stack = np.stack(bands, 2)

    return sar_stack, georef


def get_reference_shoreline_median(inputs):

    site_name = inputs['site_name']
    median_dir_path = inputs['median_dir_path']

    # check if reference shoreline already exists in the corresponding folder
    file_name = site_name + '_reference_shoreline.pkl'
    # if it exist, load it and return it
    if file_name in os.listdir(median_dir_path):
        printProgress('reference shoreline loaded')
        with open(os.path.join(median_dir_path, site_name + '_reference_shoreline.pkl'), 'rb') as f:
            refsl = pickle.load(f)
        return refsl
    else:
        printWarning('no reference shoreline found')
        return np.zeros(1)


def preprocess_S2(filenames):
    
    # read NIR image
    filename_10m = filenames[0]
    data = gdal.Open(filename_10m, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    image_10m = np.stack(bands, 2)
    image_10m = image_10m / 10000
    image_NIR = image_10m[:,:,3]
    image_ms = image_10m[:, :, [1, 2, 3]]

    # size of 10m image
    nrows = image_NIR.shape[0]
    ncols = image_NIR.shape[1]
    
    # read pan image
    filename_SWIR = filenames[1]
    data = gdal.Open(filename_SWIR, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    image_SWIR = data.GetRasterBand(1).ReadAsArray()
    image_SWIR = image_SWIR / 10000

    # resize the SWIR using bi-linear interpolation (order 1)
    image_SWIR = transform.resize(image_SWIR, (nrows, ncols),
                                  order=1, preserve_range=True,
                                  mode='constant')

    # pansharpen SWIR
    image_SWIR_ps = pansharpen_SWIR(image_SWIR, image_ms)

    image_SWIR_ps = np.expand_dims(image_SWIR_ps, axis=2)

    # add pansharpened SWIR band to the other 10m bands
    image_10m_ps = np.append(image_10m, image_SWIR_ps, axis=2)

    return image_10m_ps, georef


def pansharpen_SWIR(image_SWIR, image_ms):
    
    print(f'@@@ {image_SWIR.shape}')
    print(f'@@@ {image_ms.shape}')
    
    # reshape image into vector
    image_vec = image_ms.reshape(image_ms.shape[0] * image_ms.shape[1] * image_ms.shape[2])

    print(f'@@@ {image_vec.shape}')

    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pca = pca.fit_transform(image_vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_swir = image_SWIR.reshape(image_ms.shape[0] * image_ms.shape[1])
    vec_pca[:, 0] = hist_match(vec_swir, vec_pca[:, 0])
    vec_SWIR_ps = pca.inverse_transform(vec_pca)

    # reshape vector into image

    image_SWIR_ps = vec_SWIR_ps.reshape(image_SWIR.shape[0], image_SWIR.shape[1])

    return image_SWIR_ps
